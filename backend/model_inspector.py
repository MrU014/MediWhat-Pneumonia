"""
Quick model structure inspector
"""
import tensorflow as tf
import numpy as np

print("Loading model...")
model = tf.keras.models.load_model("pneumonia_model_balanced.h5")

# Build the model by running a dummy prediction
print("Building model...")
dummy = tf.random.normal((1, 224, 224, 3))
_ = model(dummy)

print("\n" + "="*80)
print("MODEL STRUCTURE")
print("="*80)

print("\nTop-level layers:")
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} ({layer.__class__.__name__})")

print("\n" + "="*80)
print("BASE MODEL (MobileNetV2) STRUCTURE")
print("="*80)

base = model.layers[0]  # Should be MobileNetV2
print(f"\nBase model: {base.name} ({base.__class__.__name__})")
print(f"Total layers in base: {len(base.layers)}")

# Find all Conv2D layers
conv_layers = []
for i, layer in enumerate(base.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        conv_layers.append((i, layer.name, layer.output_shape))

print(f"\nFound {len(conv_layers)} Conv2D layers:")
print("\nFirst 5 conv layers:")
for idx, name, shape in conv_layers[:5]:
    print(f"  [{idx}] {name}: {shape}")

print("\nLast 5 conv layers:")
for idx, name, shape in conv_layers[-5:]:
    print(f"  [{idx}] {name}: {shape}")

print("\n" + "="*80)
print("INPUT/OUTPUT INFO")
print("="*80)
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
print(f"Base model output shape: {base.output_shape}")

print("\n" + "="*80)
print("TESTING GRAD-CAM APPROACH")
print("="*80)

# Test the working approach
last_conv_layer_name = conv_layers[-1][1]
print(f"\nUsing last conv layer: {last_conv_layer_name}")

# Create grad model
grad_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[base.get_layer(last_conv_layer_name).output, model.output]
)

print("Testing grad model...")
conv_out, pred = grad_model(dummy)
print(f"✅ Conv output shape: {conv_out.shape}")
print(f"✅ Prediction: {pred.numpy()[0][0]:.4f}")

print("\n✅ Model structure analyzed successfully!")
print("\nKey findings:")
print(f"1. Last conv layer: {last_conv_layer_name}")
print(f"2. Conv output shape: {conv_out.shape}")
print(f"3. Use model.inputs (not model.input) for Sequential models")