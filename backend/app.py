from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = tf.keras.models.load_model('best_chest_xray_model.h5')

# Get the base MobileNetV2 sub-model
base_model = model.get_layer('mobilenetv2_1.00_224')

# Build the model with dummy input
print("Building model with dummy input...")
dummy_input = tf.random.normal((1, 224, 224, 3))
_ = model.predict(dummy_input, verbose=0)
print("Model built successfully.")

# GRAD-CAM function (higher threshold for focus)
def get_gradcam_heatmap(img_array, model, base_model):
    possible_layers = ['out_relu', 'Conv_1_relu', 'block_16_project_relu']
    conv_layer = None
    for layer_name in possible_layers:
        try:
            conv_layer = base_model.get_layer(layer_name)
            print(f"Using layer: {layer_name}")
            break
        except ValueError:
            continue
    
    if conv_layer is None:
        raise ValueError("No suitable conv layer found")
    
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        tape.watch(model.inputs)
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[0, 0]
    
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise ValueError("Gradients are None")
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads[:, tf.newaxis, tf.newaxis]), axis=-1)
    
    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    else:
        heatmap = tf.zeros_like(heatmap)
    
    # Upsample
    heatmap = tf.image.resize(heatmap[tf.newaxis, ..., tf.newaxis], [224, 224], method='bilinear')[0, :, :, 0]
    
    # Stats
    heatmap_np = heatmap.numpy()
    orig_max = np.max(heatmap_np)
    orig_mean = np.mean(heatmap_np)
    print(f"Heatmap stats (pre-boost) - Max: {orig_max:.3f}, Mean: {orig_mean:.3f}")
    
    # Conservative boost only if very low
    boosted = False
    if orig_max < 0.15:
        boost_factor = min(1.3, 1.0 / orig_max if orig_max > 0 else 1.0)  # Milder cap
        heatmap_np = np.clip(heatmap_np * boost_factor, 0, 1)
        boosted = True
        print(f"Mild boost by {boost_factor:.1f}x, new max: {np.max(heatmap_np):.3f}")
    
    # Aggressive threshold: <0.2 to 0 (focus on strong hotspots only)
    low_mask = heatmap_np < 0.2
    heatmap_np[low_mask] = 0
    hotspot_pct = (1 - np.mean(low_mask)) * 100  # % above threshold
    print(f"Applied threshold <0.2 to 0. Boosted: {boosted}, Hotspots cover {hotspot_pct:.1f}% of image")
    
    return heatmap_np

# Overlay with selective, subtle application (no filter effect)
def overlay_heatmap(heatmap, original_img):
    original_np = np.array(original_img, dtype=np.uint8)  # RGB uint8
    
    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    
    # Create mask for high activations only (>0.3 for color)
    high_mask = (heatmap_resized > 0.3).astype(np.uint8) * 255
    low_mask = 255 - high_mask  # For low areas (no color)
    
    # Apply HOT colormap (black-low to red-high, less blue flood)
    heatmap_normalized = np.uint8(255 * heatmap_resized)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_HOT)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)
    
    # Selective blend: High areas get color, low areas stay original
    high_blend = cv2.addWeighted(original_np, 0.6, heatmap_colored_rgb, 0.4, 0)
    low_blend = original_np  # No overlay for low
    
    # Combine using masks
    superimposed = np.where(high_mask[..., np.newaxis] > 0, high_blend, low_blend)
    
    # Final subtle global alpha on the whole (0.9 original + 0.1 color for faint tint)
    superimposed = cv2.addWeighted(original_np, 0.9, superimposed.astype(np.uint8), 0.1, 0)
    
    print("Selective subtle overlay applied (HOT colormap, thresholded hotspots only, 0.9 original + 0.1 color)")
    return superimposed.astype(np.uint8)

# Fallback: Very subtle tint on grayscale
def get_fallback_image(original_img):
    original_np = np.array(original_img, dtype=np.uint8)
    gray = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # Minimal uniform low heatmap (faint warm tint)
    uniform_heatmap = np.full((224, 224), 0.05)  # Extremely low
    uniform_heatmap[uniform_heatmap < 0.05] = 0
    heatmap_normalized = np.uint8(255 * uniform_heatmap)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_HOT)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)
    
    fallback = cv2.addWeighted(gray_3ch, 0.95, heatmap_colored_rgb, 0.05, 0)  # Almost invisible tint
    return fallback

# Preprocess
def preprocess_image(image_file):
    image_bytes = image_file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    return img, np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        original_img, img_array = preprocess_image(file)
        
        probability = model.predict(img_array, verbose=0)[0][0]
        diagnosis = "PNEUMONIA" if probability >= 0.5 else "NORMAL"
        confidence = f"{(probability * 100 if diagnosis == 'PNEUMONIA' else (1 - probability) * 100):.1f}%"
        
        heatmap_data = None
        try:
            heatmap = get_gradcam_heatmap(img_array, model, base_model)
            overlaid_img = overlay_heatmap(heatmap, original_img)
            
            success, buffer = cv2.imencode('.png', overlaid_img)
            if not success:
                raise ValueError("Encoding failed")
            
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            heatmap_data = f"data:image/png;base64,{img_base64}"
            
            print(f"Full GRAD-CAM heatmap generated: {len(img_base64)} characters")
            
        except Exception as grad_error:
            print(f"GRAD-CAM failed: {grad_error}")
            fallback_img = get_fallback_image(original_img)
            success, buffer = cv2.imencode('.png', fallback_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            heatmap_data = f"data:image/png;base64,{img_base64}"
            print(f"Fallback minimal tint generated: {len(img_base64)} characters")
        
        return jsonify({
            'diagnosis': diagnosis,
            'confidence': confidence,
            'heatmap_image': heatmap_data
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)