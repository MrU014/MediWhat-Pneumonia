"""
Training Script for BALANCED Dataset
Since data is already 1:1, we don't need heavy class weights
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from pathlib import Path

print("="*80)
print("PNEUMONIA TRAINING - BALANCED DATASET")
print("="*80)

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
INITIAL_LR = 0.001

# Use BALANCED dataset
BASE_DIR = Path('..') / 'chest_xray_balanced'
TRAIN_DIR = str(BASE_DIR / 'train')
VAL_DIR = str(BASE_DIR / 'val')
TEST_DIR = str(BASE_DIR / 'test')

if not os.path.exists(TRAIN_DIR):
    print(f"❌ ERROR: Balanced dataset not found!")
    print(f"   Run: python balance_dataset.py first!")
    exit(1)

print(f"✅ Using balanced dataset: {Path(TRAIN_DIR).absolute()}")

# Moderate augmentation (data is balanced, don't need extreme augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print("\n📂 Loading balanced datasets...")

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\n📊 Dataset Statistics:")
print(f"   Training: {train_generator.samples}")
print(f"   Validation: {val_generator.samples}")
print(f"   Test: {test_generator.samples}")

# Check balance
normal_count = len(os.listdir(os.path.join(TRAIN_DIR, 'NORMAL')))
pneumonia_count = len(os.listdir(os.path.join(TRAIN_DIR, 'PNEUMONIA')))

print(f"\n⚖️  Training Set Balance:")
print(f"   NORMAL: {normal_count}")
print(f"   PNEUMONIA: {pneumonia_count}")
print(f"   Ratio: 1:{pneumonia_count/normal_count:.2f}")

if abs(pneumonia_count/normal_count - 1.0) > 0.1:
    print("   ⚠️  WARNING: Data not balanced! Run balance_dataset.py again")
else:
    print("   ✅ Data is balanced!")

# NO class weights needed since data is balanced!
print("\n   Note: Not using class weights (data is balanced)")

# Build model
print("\n🏗️  Building model...")

base_model = MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', 
                 kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid', name='predictions')
], name='PneumoniaDetector_Balanced')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]
)

print("\n📋 Model Summary:")
model.summary()

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_chest_xray_model.h5',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Phase 1: Train with frozen base
print("\n" + "="*80)
print("PHASE 1: Training with frozen MobileNetV2 base")
print("="*80)

history1 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Phase 2: Fine-tune
print("\n" + "="*80)
print("PHASE 2: Fine-tuning with unfrozen layers")
print("="*80)

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR / 10),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]
)

history2 = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
print("\n" + "="*80)
print("FINAL EVALUATION")
print("="*80)

test_results = model.evaluate(test_generator, verbose=1)

print("\n📊 Test Results:")
print(f"   Loss: {test_results[0]:.4f}")
print(f"   Accuracy: {test_results[1]:.4f}")
print(f"   Precision: {test_results[2]:.4f}")
print(f"   Recall: {test_results[3]:.4f}")
print(f"   AUC: {test_results[4]:.4f}")

# Calculate balanced accuracy
test_generator.reset()
predictions = model.predict(test_generator, verbose=0)
labels = test_generator.labels

preds_binary = (predictions > 0.5).astype(int).flatten()

tp = np.sum((preds_binary == 1) & (labels == 1))
tn = np.sum((preds_binary == 0) & (labels == 0))
fp = np.sum((preds_binary == 1) & (labels == 0))
fn = np.sum((preds_binary == 0) & (labels == 1))

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
balanced_acc = (sensitivity + specificity) / 2

print(f"\n🎯 Balanced Metrics:")
print(f"   Sensitivity (Recall): {sensitivity:.4f}")
print(f"   Specificity: {specificity:.4f}")
print(f"   Balanced Accuracy: {balanced_acc:.4f}")
print(f"   True Positives: {tp}")
print(f"   True Negatives: {tn}")
print(f"   False Positives: {fp}")
print(f"   False Negatives: {fn}")

# Save
model.save('pneumonia_model_balanced.h5')
print(f"\n✅ Models saved!")

# Sample predictions
print("\n🧪 Sample Predictions:")
test_generator.reset()
x_batch, y_batch = next(test_generator)
predictions = model.predict(x_batch[:20], verbose=0)

correct = 0
for i in range(20):
    actual = "PNEUMONIA" if y_batch[i] == 1 else "NORMAL"
    pred_prob = predictions[i][0]
    predicted = "PNEUMONIA" if pred_prob >= 0.5 else "NORMAL"
    match = "✅" if actual == predicted else "❌"
    
    if actual == predicted:
        correct += 1
    
    print(f"{match} {i+1:2d}: {actual:10s} → {predicted:10s} ({pred_prob:.4f})")

print(f"\nSample accuracy: {correct}/20 = {correct*5}%")

# Variance check
print("\n🔍 Model Variance Check:")
random_preds = [model.predict(np.random.rand(1, 224, 224, 3).astype(np.float32), verbose=0)[0][0] 
                for _ in range(20)]
variance = np.var(random_preds)
mean = np.mean(random_preds)

print(f"   Mean: {mean:.4f}")
print(f"   Variance: {variance:.6f}")
print(f"   Range: [{np.min(random_preds):.4f}, {np.max(random_preds):.4f}]")

if variance > 0.01 and 0.3 < mean < 0.7:
    print("   ✅ EXCELLENT! Model shows good variance and balance!")
elif variance > 0.01:
    print("   ✅ Good variance but check if biased")
else:
    print("   ⚠️  Low variance - might still be stuck")

print("\n" + "="*80)
print("TRAINING COMPLETE! 🎉")
print("="*80)
print("\nNext steps:")
print("1. python model_diagnostic.py")
print("2. python app.py")
print("3. Test with frontend!")
print("="*80)