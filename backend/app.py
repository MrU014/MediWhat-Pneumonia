from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io, base64, cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

print("Loading model...")
model = tf.keras.models.load_model("pneumonia_model_balanced.h5")

# CRITICAL: Build the model first
print("Building model...")
dummy_input = tf.random.normal((1, 224, 224, 3))
_ = model(dummy_input)
print("✅ Model ready")

# Get the base model
base_model = model.layers[0]

# Find last conv layer ONCE at startup
last_conv_layer = None
for layer in reversed(base_model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer.name
        break

print(f"✅ Using conv layer for Grad-CAM: {last_conv_layer}")

def preprocess(file):
    img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return img, np.expand_dims(arr.astype(np.float32), axis=0)

# ------------------ WORKING Grad-CAM -------------------
def gradcam(img_array):
    """
    Generate Grad-CAM heatmap using the correct approach
    """
    # Convert to tensor
    img_tensor = tf.convert_to_tensor(img_array)
    
    # Create a model that maps from input to conv output AND final prediction
    # KEY: Use base_model.input (the Functional model has proper input)
    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer).output, base_model.output]
    )
    
    # Watch the gradients
    with tf.GradientTape() as tape:
        # Get conv outputs and predictions
        last_conv_output, base_predictions = grad_model(img_tensor)
        
        # Pass base output through the rest of the model
        x = model.layers[1](base_predictions)  # GlobalAveragePooling2D
        x = model.layers[2](x, training=False)  # Dropout
        x = model.layers[3](x)  # Dense
        x = model.layers[4](x, training=False)  # BatchNormalization
        x = model.layers[5](x, training=False)  # Dropout
        predictions = model.layers[6](x)  # Final Dense
        
        # Get the score for pneumonia class
        class_channel = predictions[:, 0]
    
    # Compute gradient of class score with respect to conv output
    grads = tape.gradient(class_channel, last_conv_output)
    
    # Global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps by gradient importance
    last_conv_output = last_conv_output[0].numpy()
    pooled_grads = pooled_grads.numpy()
    
    # Compute weighted combination
    heatmap = np.zeros(last_conv_output.shape[:2], dtype=np.float32)
    for i in range(pooled_grads.shape[0]):
        heatmap += pooled_grads[i] * last_conv_output[:, :, i]
    
    # ReLU
    heatmap = np.maximum(heatmap, 0)
    
    # Normalize
    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Resize to input size
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # Smooth
    heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)
    
    return heatmap

def overlay_heat(heat, img):
    """Create overlay of heatmap on original image"""
    img_array = np.array(img).astype("uint8")
    
    # Apply JET colormap
    heat_uint8 = np.uint8(255 * heat)
    heat_colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_colored = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlayed = cv2.addWeighted(img_array, 0.6, heat_colored, 0.4, 0)
    
    return overlayed

def explain(diag, conf, heat):
    """Generate explanation"""
    h, w = heat.shape
    
    # Analyze lung regions
    left_region = heat[int(.2*h):int(.8*h), int(.2*w):int(.45*w)]
    right_region = heat[int(.2*h):int(.8*h), int(.55*w):int(.8*w)]
    
    left_intensity = np.mean(left_region)
    right_intensity = np.mean(right_region)
    
    lung_side = "left" if left_intensity > right_intensity else "right"
    overall_intensity = np.mean(heat)
    
    if diag == "PNEUMONIA":
        if overall_intensity > 0.5:
            intensity_desc = "strong"
        elif overall_intensity > 0.3:
            intensity_desc = "moderate"
        else:
            intensity_desc = "mild"
        
        return (f"⚠️ Pneumonia detected with {conf} confidence. "
                f"The model shows {intensity_desc} activation in the {lung_side} lung region. "
                f"This is NOT a medical diagnosis - please consult a healthcare professional.")
    else:
        return (f"✅ No pneumonia detected with {conf} confidence. "
                f"The X-ray appears normal with no significant abnormalities. "
                f"This is NOT a medical diagnosis - please consult a healthcare professional.")

# ------------------ API -------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Preprocess
        img, arr = preprocess(file)
        
        # Predict
        prediction = model.predict(arr, verbose=0)[0][0]
        
        # Format results
        diag = "PNEUMONIA" if prediction >= 0.5 else "NORMAL"
        confidence_value = prediction * 100 if diag == "PNEUMONIA" else (1 - prediction) * 100
        conf = f"{confidence_value:.1f}%"
        
        print(f"Prediction: {diag} ({conf}) - raw score: {prediction:.4f}")
        
        # Generate Grad-CAM
        try:
            heatmap = gradcam(arr)
            overlay = overlay_heat(heatmap, img)
            
            # Encode to base64
            _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            img_uri = f"data:image/png;base64,{img_base64}"
            
            explanation = explain(diag, conf, heatmap)
            
            print("✅ Grad-CAM generated successfully")
            
        except Exception as e:
            print(f"⚠️ Grad-CAM failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback
            img_array = np.array(img)
            _, buffer = cv2.imencode(".png", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            img_uri = f"data:image/png;base64,{img_base64}"
            
            explanation = f"{diag} ({conf}). Heatmap unavailable."
        
        return jsonify({
            "diagnosis": diag,
            "confidence": conf,
            "raw_score": float(prediction),
            "heatmap_image": img_uri,
            "explanation": explanation
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🏥 Pneumonia Detection API")
    print("="*80)
    print("Endpoints:")
    print("  POST /predict - Upload X-ray for analysis")
    print("  GET  /health  - Check API status")
    print("="*80 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')