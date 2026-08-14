"""
Comprehensive Testing & Statistics Script
Tests all aspects of the Pneumonia Detection system and generates detailed reports
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import requests
from PIL import Image
import io
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, matthews_corrcoef
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("COMPREHENSIVE PNEUMONIA DETECTION SYSTEM TEST")
print("=" * 100)
print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

# Configuration
BASE_DIR = Path('..')
DATASET_DIR = BASE_DIR / 'chest_xray_balanced'
MODEL_PATH = "pneumonia_model_balanced.h5"
API_URL = "http://localhost:5000"
OUTPUT_DIR = Path('test_results')
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize results dictionary
test_results = {
    'timestamp': datetime.now().isoformat(),
    'tests_passed': 0,
    'tests_failed': 0,
    'warnings': []
}

# ============================================================================
# TEST 1: DATASET STATISTICS
# ============================================================================

def test_dataset_statistics():
    """Analyze dataset composition and balance"""
    print("\n" + "=" * 100)
    print("TEST 1: DATASET STATISTICS")
    print("=" * 100)
    
    stats = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = DATASET_DIR / split
        
        if not split_dir.exists():
            print(f"❌ ERROR: {split} directory not found at {split_dir}")
            test_results['tests_failed'] += 1
            return None
        
        normal_dir = split_dir / 'NORMAL'
        pneumonia_dir = split_dir / 'PNEUMONIA'
        
        normal_files = list(normal_dir.glob('*.jpeg')) + list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.png'))
        pneumonia_files = list(pneumonia_dir.glob('*.jpeg')) + list(pneumonia_dir.glob('*.jpg')) + list(pneumonia_dir.glob('*.png'))
        
        normal_count = len(normal_files)
        pneumonia_count = len(pneumonia_files)
        total = normal_count + pneumonia_count
        
        stats[split] = {
            'normal': normal_count,
            'pneumonia': pneumonia_count,
            'total': total,
            'ratio': pneumonia_count / normal_count if normal_count > 0 else 0,
            'balance_percentage': (min(normal_count, pneumonia_count) / max(normal_count, pneumonia_count) * 100) if max(normal_count, pneumonia_count) > 0 else 0
        }
    
    # Display statistics
    print("\n📊 Dataset Composition:")
    print("-" * 100)
    print(f"{'Split':<15} {'NORMAL':<15} {'PNEUMONIA':<15} {'Total':<15} {'Ratio':<15} {'Balance':<15}")
    print("-" * 100)
    
    for split, data in stats.items():
        print(f"{split.upper():<15} {data['normal']:<15} {data['pneumonia']:<15} {data['total']:<15} "
              f"1:{data['ratio']:.2f}{'':<8} {data['balance_percentage']:.1f}%{'':<8}")
    
    # Calculate totals
    total_normal = sum(s['normal'] for s in stats.values())
    total_pneumonia = sum(s['pneumonia'] for s in stats.values())
    grand_total = sum(s['total'] for s in stats.values())
    
    print("-" * 100)
    print(f"{'TOTAL':<15} {total_normal:<15} {total_pneumonia:<15} {grand_total:<15}")
    print("-" * 100)
    
    # Validation checks
    print("\n✓ Validation Checks:")
    
    # Check training balance
    train_balance = stats['train']['balance_percentage']
    if train_balance >= 95:
        print(f"  ✅ Training set is well balanced ({train_balance:.1f}%)")
        test_results['tests_passed'] += 1
    else:
        print(f"  ⚠️  Training set balance is suboptimal ({train_balance:.1f}%)")
        test_results['warnings'].append(f"Training balance: {train_balance:.1f}%")
    
    # Check minimum samples
    if stats['test']['total'] >= 500:
        print(f"  ✅ Test set has sufficient samples ({stats['test']['total']})")
        test_results['tests_passed'] += 1
    else:
        print(f"  ⚠️  Test set is small ({stats['test']['total']} samples)")
        test_results['warnings'].append(f"Small test set: {stats['test']['total']} samples")
    
    # Visualize dataset distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart
    splits = list(stats.keys())
    normal_counts = [stats[s]['normal'] for s in splits]
    pneumonia_counts = [stats[s]['pneumonia'] for s in splits]
    
    x = np.arange(len(splits))
    width = 0.35
    
    axes[0].bar(x - width/2, normal_counts, width, label='NORMAL', color='#4caf50', alpha=0.8)
    axes[0].bar(x + width/2, pneumonia_counts, width, label='PNEUMONIA', color='#f44336', alpha=0.8)
    axes[0].set_xlabel('Dataset Split', fontsize=12)
    axes[0].set_ylabel('Number of Images', fontsize=12)
    axes[0].set_title('Dataset Distribution by Split', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s.upper() for s in splits])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Pie chart for total distribution
    labels = ['NORMAL', 'PNEUMONIA']
    sizes = [total_normal, total_pneumonia]
    colors = ['#4caf50', '#f44336']
    explode = (0.05, 0.05)
    
    axes[1].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Overall Dataset Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_statistics.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Dataset visualization saved to {OUTPUT_DIR / 'dataset_statistics.png'}")
    plt.close()
    
    test_results['dataset_stats'] = stats
    return stats

# ============================================================================
# TEST 2: MODEL ARCHITECTURE & PARAMETERS
# ============================================================================

def test_model_architecture():
    """Analyze model architecture and parameters"""
    print("\n" + "=" * 100)
    print("TEST 2: MODEL ARCHITECTURE & PARAMETERS")
    print("=" * 100)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        test_results['tests_failed'] += 1
        return None
    
    try:
        # Load model
        print("\n📦 Loading model...")
        model = keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
        
        # Build model
        dummy = tf.random.normal((1, 224, 224, 3))
        _ = model(dummy, training=False)
        
        # Model summary
        print("\n📋 Model Architecture:")
        print("-" * 100)
        
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"Model Name: {model.name}")
        print(f"Input Shape: {model.input_shape}")
        print(f"Output Shape: {model.output_shape}")
        print(f"Total Layers: {len(model.layers)}")
        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Non-trainable: {non_trainable_params:,}")
        
        # Layer details
        print("\n📝 Layer Details:")
        print("-" * 100)
        print(f"{'Layer':<30} {'Type':<30} {'Output Shape':<20} {'Params':<15}")
        print("-" * 100)
        
        for i, layer in enumerate(model.layers):
            params = layer.count_params()
            try:
                output_shape = str(layer.output_shape)
            except AttributeError:
                output_shape = "N/A"
            print(f"{layer.name:<30} {layer.__class__.__name__:<30} {output_shape:<20} {params:>12,}")
        
        print("-" * 100)
        
        # Model size
        model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"\n💾 Model File Size: {model_size_mb:.2f} MB")
        
        # Validation checks
        print("\n✓ Validation Checks:")
        
        if model.input_shape[1:3] == (224, 224):
            print("  ✅ Input shape is correct (224x224)")
            test_results['tests_passed'] += 1
        else:
            print(f"  ❌ Input shape is incorrect: {model.input_shape}")
            test_results['tests_failed'] += 1
        
        if model.output_shape[-1] == 1:
            print("  ✅ Output shape is correct (binary classification)")
            test_results['tests_passed'] += 1
        else:
            print(f"  ❌ Output shape is incorrect: {model.output_shape}")
            test_results['tests_failed'] += 1
        
        if model_size_mb < 50:
            print(f"  ✅ Model size is efficient ({model_size_mb:.2f} MB)")
            test_results['tests_passed'] += 1
        else:
            print(f"  ⚠️  Model is large ({model_size_mb:.2f} MB)")
            test_results['warnings'].append(f"Large model: {model_size_mb:.2f} MB")
        
        test_results['model_info'] = {
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'non_trainable_params': int(non_trainable_params),
            'size_mb': float(model_size_mb),
            'layers': len(model.layers)
        }
        
        return model
        
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        test_results['tests_failed'] += 1
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# TEST 3: MODEL PERFORMANCE EVALUATION
# ============================================================================

def test_model_performance(model):
    """Comprehensive model performance testing"""
    print("\n" + "=" * 100)
    print("TEST 3: MODEL PERFORMANCE EVALUATION")
    print("=" * 100)
    
    if model is None:
        print("❌ Skipping: Model not loaded")
        test_results['tests_failed'] += 1
        return
    
    # Create test generator
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        str(DATASET_DIR / 'test'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"\n📊 Test Set: {test_generator.samples} images")
    print(f"   NORMAL: {test_generator.class_indices}")
    
    # Get predictions
    print("\n🔄 Generating predictions...")
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    y_pred_prob = predictions.flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    y_true = test_generator.labels
    
    # Calculate metrics
    print("\n📈 Calculating metrics...")
    
    # Basic metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity and sensitivity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Display results
    print("\n" + "=" * 100)
    print("PERFORMANCE METRICS")
    print("=" * 100)
    
    print(f"\n🎯 Classification Metrics:")
    print(f"   Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Balanced Accuracy:  {balanced_accuracy:.4f} ({balanced_accuracy*100:.2f}%)")
    print(f"   Precision:          {precision:.4f}")
    print(f"   Recall (Sensitivity): {recall:.4f}")
    print(f"   Specificity:        {specificity:.4f}")
    print(f"   F1-Score:           {f1:.4f}")
    print(f"   Matthews Correlation: {mcc:.4f}")
    print(f"   ROC AUC:            {roc_auc:.4f}")
    print(f"   PR AUC:             {pr_auc:.4f}")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"   True Negatives (TN):   {tn:>6}")
    print(f"   False Positives (FP):  {fp:>6}")
    print(f"   False Negatives (FN):  {fn:>6}")
    print(f"   True Positives (TP):   {tp:>6}")
    
    print(f"\n📉 Error Analysis:")
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f"   False Positive Rate: {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
    print(f"   False Negative Rate: {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)")
    
    # Validation checks
    print("\n✓ Performance Validation:")
    
    if accuracy >= 0.85:
        print(f"  ✅ Excellent accuracy ({accuracy*100:.2f}%)")
        test_results['tests_passed'] += 1
    elif accuracy >= 0.75:
        print(f"  ⚠️  Acceptable accuracy ({accuracy*100:.2f}%)")
        test_results['warnings'].append(f"Accuracy: {accuracy*100:.2f}%")
    else:
        print(f"  ❌ Poor accuracy ({accuracy*100:.2f}%)")
        test_results['tests_failed'] += 1
    
    if balanced_accuracy >= 0.85:
        print(f"  ✅ Excellent balanced accuracy ({balanced_accuracy*100:.2f}%)")
        test_results['tests_passed'] += 1
    else:
        print(f"  ⚠️  Balanced accuracy needs improvement ({balanced_accuracy*100:.2f}%)")
        test_results['warnings'].append(f"Balanced accuracy: {balanced_accuracy*100:.2f}%")
    
    if roc_auc >= 0.90:
        print(f"  ✅ Excellent ROC AUC ({roc_auc:.4f})")
        test_results['tests_passed'] += 1
    elif roc_auc >= 0.80:
        print(f"  ⚠️  Acceptable ROC AUC ({roc_auc:.4f})")
        test_results['warnings'].append(f"ROC AUC: {roc_auc:.4f}")
    else:
        print(f"  ❌ Poor ROC AUC ({roc_auc:.4f})")
        test_results['tests_failed'] += 1
    
    # Classification report
    print("\n" + "=" * 100)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 100)
    class_names = ['NORMAL', 'PNEUMONIA']
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # Visualizations
    print("\n📊 Generating visualizations...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_xlabel('Predicted', fontsize=12)
    
    # 2. Normalized Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax2,
                cbar_kws={'label': 'Proportion'})
    ax2.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Actual', fontsize=12)
    ax2.set_xlabel('Predicted', fontsize=12)
    
    # 3. ROC Curve
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate', fontsize=12)
    ax3.set_ylabel('True Positive Rate', fontsize=12)
    ax3.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax3.legend(loc="lower right")
    ax3.grid(True, alpha=0.3)
    
    # 4. Precision-Recall Curve
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(recall_curve, precision_curve, color='green', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.4f})')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('Recall', fontsize=12)
    ax4.set_ylabel('Precision', fontsize=12)
    ax4.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax4.legend(loc="lower left")
    ax4.grid(True, alpha=0.3)
    
    # 5. Prediction Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(y_pred_prob[y_true == 0], bins=50, alpha=0.6, label='NORMAL', color='green')
    ax5.hist(y_pred_prob[y_true == 1], bins=50, alpha=0.6, label='PNEUMONIA', color='red')
    ax5.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax5.set_xlabel('Predicted Probability', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Metrics Comparison Bar Chart
    ax6 = fig.add_subplot(gs[1, 2])
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'Balanced\nAccuracy']
    metrics_values = [accuracy, precision, recall, f1, specificity, balanced_accuracy]
    colors_bar = ['#667eea' if v >= 0.85 else '#ffa726' if v >= 0.75 else '#ef5350' for v in metrics_values]
    bars = ax6.barh(metrics_names, metrics_values, color=colors_bar, alpha=0.8)
    ax6.set_xlim([0, 1])
    ax6.set_xlabel('Score', fontsize=12)
    ax6.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax6.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, metrics_values)):
        ax6.text(value + 0.02, i, f'{value:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # 7. Class-wise Performance
    ax7 = fig.add_subplot(gs[2, 0])
    class_metrics = {
        'NORMAL': {
            'Precision': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'Recall': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'F1': 2 * (tn / (tn + fn)) * (tn / (tn + fp)) / ((tn / (tn + fn)) + (tn / (tn + fp))) if (tn + fn) > 0 and (tn + fp) > 0 else 0
        },
        'PNEUMONIA': {
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    }
    
    x_pos = np.arange(3)
    width = 0.35
    normal_values = list(class_metrics['NORMAL'].values())
    pneumonia_values = list(class_metrics['PNEUMONIA'].values())
    
    ax7.bar(x_pos - width/2, normal_values, width, label='NORMAL', color='#4caf50', alpha=0.8)
    ax7.bar(x_pos + width/2, pneumonia_values, width, label='PNEUMONIA', color='#f44336', alpha=0.8)
    ax7.set_ylabel('Score', fontsize=12)
    ax7.set_title('Class-wise Performance', fontsize=14, fontweight='bold')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)
    ax7.set_ylim([0, 1.1])
    
    # 8. Error Analysis
    ax8 = fig.add_subplot(gs[2, 1])
    error_types = ['True\nNegatives', 'False\nPositives', 'False\nNegatives', 'True\nPositives']
    error_counts = [tn, fp, fn, tp]
    error_colors = ['#4caf50', '#ff9800', '#f44336', '#4caf50']
    bars = ax8.bar(error_types, error_counts, color=error_colors, alpha=0.8)
    ax8.set_ylabel('Count', fontsize=12)
    ax8.set_title('Confusion Matrix Breakdown', fontsize=14, fontweight='bold')
    ax8.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, error_counts):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(y_true)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 9. Performance Summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = f"""
    PERFORMANCE SUMMARY
    {'=' * 40}
    
    Overall Metrics:
    • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
    • Balanced Accuracy: {balanced_accuracy:.4f}
    • ROC AUC: {roc_auc:.4f}
    • PR AUC: {pr_auc:.4f}
    
    Classification Performance:
    • Precision: {precision:.4f}
    • Recall: {recall:.4f}
    • F1-Score: {f1:.4f}
    • MCC: {mcc:.4f}
    
    Clinical Metrics:
    • Sensitivity: {sensitivity:.4f}
    • Specificity: {specificity:.4f}
    • FPR: {false_positive_rate:.4f}
    • FNR: {false_negative_rate:.4f}
    
    Test Set: {len(y_true)} images
    Threshold: 0.50
    """
    
    ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Comprehensive Model Performance Analysis', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(OUTPUT_DIR / 'performance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Performance visualization saved to {OUTPUT_DIR / 'performance_analysis.png'}")
    plt.close()
    
    # Store results
    test_results['performance_metrics'] = {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'mcc': float(mcc),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'error_rates': {
            'fpr': float(false_positive_rate),
            'fnr': float(false_negative_rate)
        }
    }

# ============================================================================
# TEST 4: API HEALTH CHECK
# ============================================================================

def test_api_health():
    """Test Flask API availability and health"""
    print("\n" + "=" * 100)
    print("TEST 4: API HEALTH CHECK")
    print("=" * 100)
    
    print(f"\n🌐 Testing API at {API_URL}")
    
    # Test health endpoint
    try:
        print("\n1. Testing /health endpoint...")
        response = requests.get(f"{API_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {response.status_code}")
            print(f"   ✅ Response: {data}")
            
            if data.get('status') == 'healthy' and data.get('model_loaded'):
                print("   ✅ API is healthy and model is loaded")
                test_results['tests_passed'] += 1
            else:
                print("   ⚠️  API returned unexpected data")
                test_results['warnings'].append("API health check: unexpected response")
        else:
            print(f"   ❌ Status: {response.status_code}")
            test_results['tests_failed'] += 1
            
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to API at {API_URL}")
        print(f"   ⚠️  Make sure the backend server is running (python app.py)")
        test_results['tests_failed'] += 1
        test_results['api_available'] = False
        return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        test_results['tests_failed'] += 1
        test_results['api_available'] = False
        return
    
    # Test with a sample image
    try:
        print("\n2. Testing /predict endpoint with sample image...")
        
        # Find a test image
        test_normal = list((DATASET_DIR / 'test' / 'NORMAL').glob('*.jpeg'))
        test_pneumonia = list((DATASET_DIR / 'test' / 'PNEUMONIA').glob('*.jpeg'))
        
        if test_normal and test_pneumonia:
            # Test with NORMAL image
            test_file = test_normal[0]
            print(f"   Testing with NORMAL image: {test_file.name}")
            
            with open(test_file, 'rb') as f:
                files = {'file': (test_file.name, f, 'image/jpeg')}
                response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Prediction successful")
                print(f"   ✅ Diagnosis: {data.get('diagnosis')}")
                print(f"   ✅ Confidence: {data.get('confidence')}")
                print(f"   ✅ Heatmap generated: {'heatmap_image' in data}")
                test_results['tests_passed'] += 1
            else:
                print(f"   ❌ Prediction failed with status: {response.status_code}")
                test_results['tests_failed'] += 1
            
            # Test with PNEUMONIA image
            test_file = test_pneumonia[0]
            print(f"\n   Testing with PNEUMONIA image: {test_file.name}")
            
            with open(test_file, 'rb') as f:
                files = {'file': (test_file.name, f, 'image/jpeg')}
                response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Prediction successful")
                print(f"   ✅ Diagnosis: {data.get('diagnosis')}")
                print(f"   ✅ Confidence: {data.get('confidence')}")
                print(f"   ✅ Heatmap generated: {'heatmap_image' in data}")
                test_results['tests_passed'] += 1
            else:
                print(f"   ❌ Prediction failed with status: {response.status_code}")
                test_results['tests_failed'] += 1
                
        else:
            print("   ⚠️  No test images found, skipping prediction test")
            test_results['warnings'].append("No test images for API testing")
            
    except Exception as e:
        print(f"   ❌ Error during prediction test: {e}")
        test_results['tests_failed'] += 1
    
    test_results['api_available'] = True

# ============================================================================
# TEST 5: PREDICTION CONSISTENCY
# ============================================================================

def test_prediction_consistency(model):
    """Test model prediction consistency and variance"""
    print("\n" + "=" * 100)
    print("TEST 5: PREDICTION CONSISTENCY & VARIANCE")
    print("=" * 100)
    
    if model is None:
        print("❌ Skipping: Model not loaded")
        test_results['tests_failed'] += 1
        return
    
    print("\n🔄 Testing prediction consistency...")
    
    # Test 1: Same image, multiple predictions
    print("\n1. Repeatability Test (same image, 10 predictions):")
    
    test_img_path = list((DATASET_DIR / 'test' / 'NORMAL').glob('*.jpeg'))[0]
    img = Image.open(test_img_path).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    predictions = []
    for i in range(10):
        pred = model.predict(img_array, verbose=0)[0][0]
        predictions.append(pred)
    
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    pred_variance = np.var(predictions)
    
    print(f"   Mean prediction: {pred_mean:.6f}")
    print(f"   Std deviation: {pred_std:.8f}")
    print(f"   Variance: {pred_variance:.10f}")
    print(f"   Range: [{np.min(predictions):.6f}, {np.max(predictions):.6f}]")
    
    if pred_std < 1e-5:
        print("   ✅ Excellent consistency (deterministic predictions)")
        test_results['tests_passed'] += 1
    elif pred_std < 1e-3:
        print("   ✅ Good consistency")
        test_results['tests_passed'] += 1
    else:
        print(f"   ⚠️  High variance detected (std={pred_std:.6f})")
        test_results['warnings'].append(f"Prediction variance: {pred_std:.6f}")
    
    # Test 2: Random noise sensitivity
    print("\n2. Noise Sensitivity Test:")
    
    original_pred = model.predict(img_array, verbose=0)[0][0]
    
    noisy_predictions = []
    for noise_level in [0.01, 0.05, 0.1]:
        noise = np.random.normal(0, noise_level, img_array.shape).astype(np.float32)
        noisy_img = np.clip(img_array + noise, 0, 1)
        noisy_pred = model.predict(noisy_img, verbose=0)[0][0]
        noisy_predictions.append(noisy_pred)
        diff = abs(noisy_pred - original_pred)
        print(f"   Noise level {noise_level:.2f}: prediction={noisy_pred:.4f}, diff={diff:.4f}")
    
    max_diff = max([abs(p - original_pred) for p in noisy_predictions])
    
    if max_diff < 0.1:
        print(f"   ✅ Model is robust to noise (max diff: {max_diff:.4f})")
        test_results['tests_passed'] += 1
    else:
        print(f"   ⚠️  Model is sensitive to noise (max diff: {max_diff:.4f})")
        test_results['warnings'].append(f"Noise sensitivity: {max_diff:.4f}")
    
    # Test 3: Batch vs single prediction
    print("\n3. Batch Processing Consistency:")
    
    # Get 5 test images
    test_images = list((DATASET_DIR / 'test' / 'NORMAL').glob('*.jpeg'))[:5]
    
    # Single predictions
    single_preds = []
    for img_path in test_images:
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        pred = model.predict(img_array, verbose=0)[0][0]
        single_preds.append(pred)
    
    # Batch prediction
    batch_arrays = []
    for img_path in test_images:
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        batch_arrays.append(img_array)
    
    batch_input = np.array(batch_arrays).astype(np.float32)
    batch_preds = model.predict(batch_input, verbose=0).flatten()
    
    # Compare
    differences = [abs(s - b) for s, b in zip(single_preds, batch_preds)]
    max_batch_diff = max(differences)
    
    print(f"   Max difference: {max_batch_diff:.8f}")
    
    if max_batch_diff < 1e-5:
        print("   ✅ Perfect consistency between single and batch predictions")
        test_results['tests_passed'] += 1
    else:
        print(f"   ⚠️  Minor differences detected")
        test_results['warnings'].append(f"Batch consistency: {max_batch_diff:.8f}")
    
    test_results['consistency_metrics'] = {
        'repeatability_std': float(pred_std),
        'repeatability_variance': float(pred_variance),
        'noise_sensitivity': float(max_diff),
        'batch_consistency': float(max_batch_diff)
    }

# ============================================================================
# TEST 6: INFERENCE SPEED
# ============================================================================

def test_inference_speed(model):
    """Test model inference speed"""
    print("\n" + "=" * 100)
    print("TEST 6: INFERENCE SPEED BENCHMARK")
    print("=" * 100)
    
    if model is None:
        print("❌ Skipping: Model not loaded")
        test_results['tests_failed'] += 1
        return
    
    import time
    
    # Prepare test image
    test_img_path = list((DATASET_DIR / 'test' / 'NORMAL').glob('*.jpeg'))[0]
    img = Image.open(test_img_path).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    # Warm-up
    print("\n🔥 Warming up model...")
    for _ in range(5):
        _ = model.predict(img_array, verbose=0)
    
    # Single image inference
    print("\n⏱️  Single Image Inference (100 iterations):")
    
    times = []
    for _ in range(100):
        start = time.time()
        _ = model.predict(img_array, verbose=0)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    mean_time = np.mean(times)
    median_time = np.median(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"   Mean: {mean_time:.2f} ms")
    print(f"   Median: {median_time:.2f} ms")
    print(f"   Std Dev: {std_time:.2f} ms")
    print(f"   Min: {min_time:.2f} ms")
    print(f"   Max: {max_time:.2f} ms")
    print(f"   Throughput: {1000/mean_time:.1f} images/second")
    
    # Batch inference
    print("\n⏱️  Batch Inference (batches of 32, 10 iterations):")
    
    batch_input = np.repeat(img_array, 32, axis=0)
    
    batch_times = []
    for _ in range(10):
        start = time.time()
        _ = model.predict(batch_input, verbose=0)
        end = time.time()
        batch_times.append((end - start) * 1000)
    
    mean_batch_time = np.mean(batch_times)
    time_per_image = mean_batch_time / 32
    
    print(f"   Mean batch time: {mean_batch_time:.2f} ms")
    print(f"   Time per image: {time_per_image:.2f} ms")
    print(f"   Throughput: {32000/mean_batch_time:.1f} images/second")
    
    # Validation
    print("\n✓ Performance Validation:")
    
    if mean_time < 500:  # Less than 500ms
        print(f"   ✅ Excellent inference speed ({mean_time:.2f} ms)")
        test_results['tests_passed'] += 1
    elif mean_time < 1000:
        print(f"   ✅ Acceptable inference speed ({mean_time:.2f} ms)")
        test_results['tests_passed'] += 1
    else:
        print(f"   ⚠️  Slow inference speed ({mean_time:.2f} ms)")
        test_results['warnings'].append(f"Inference time: {mean_time:.2f} ms")
    
    test_results['inference_speed'] = {
        'single_image_ms': {
            'mean': float(mean_time),
            'median': float(median_time),
            'std': float(std_time),
            'min': float(min_time),
            'max': float(max_time)
        },
        'batch_ms': {
            'mean_batch': float(mean_batch_time),
            'per_image': float(time_per_image)
        },
        'throughput': {
            'single': float(1000/mean_time),
            'batch': float(32000/mean_batch_time)
        }
    }

# ============================================================================
# TEST 7: GRAD-CAM FUNCTIONALITY
# ============================================================================

def test_gradcam_functionality(model):
    """Test Grad-CAM heatmap generation"""
    print("\n" + "=" * 100)
    print("TEST 7: GRAD-CAM FUNCTIONALITY TEST")
    print("=" * 100)
    
    if model is None:
        print("❌ Skipping: Model not loaded")
        test_results['tests_failed'] += 1
        return
    
    print("\n🔥 Testing Grad-CAM heatmap generation...")
    
    try:
        # Find the last conv layer
        base_model = model.layers[0]
        last_conv_layer = None
        
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break
        
        if last_conv_layer is None:
            print("   ❌ No Conv2D layer found in model")
            test_results['tests_failed'] += 1
            return
        
        print(f"   ✅ Found last conv layer: {last_conv_layer}")
        
        # Test with sample image
        test_img_path = list((DATASET_DIR / 'test' / 'PNEUMONIA').glob('*.jpeg'))[0]
        img = Image.open(test_img_path).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        # Generate heatmap
        print("\n   Generating Grad-CAM heatmap...")
        
        img_tensor = tf.convert_to_tensor(img_array)
        
        grad_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=[base_model.get_layer(last_conv_layer).output, base_model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_output, base_predictions = grad_model(img_tensor)
            
            x = model.layers[1](base_predictions)
            x = model.layers[2](x, training=False)
            x = model.layers[3](x)
            x = model.layers[4](x, training=False)
            x = model.layers[5](x, training=False)
            predictions = model.layers[6](x)
            
            class_channel = predictions[:, 0]
        
        grads = tape.gradient(class_channel, last_conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        last_conv_output = last_conv_output[0].numpy()
        pooled_grads = pooled_grads.numpy()
        
        heatmap = np.zeros(last_conv_output.shape[:2], dtype=np.float32)
        for i in range(pooled_grads.shape[0]):
            heatmap += pooled_grads[i] * last_conv_output[:, :, i]
        
        heatmap = np.maximum(heatmap, 0)
        
        if np.max(heatmap) != 0:
            heatmap = heatmap / np.max(heatmap)
        
        print(f"   ✅ Heatmap generated successfully")
        print(f"   ✅ Heatmap shape: {heatmap.shape}")
        print(f"   ✅ Heatmap range: [{np.min(heatmap):.4f}, {np.max(heatmap):.4f}]")
        print(f"   ✅ Heatmap mean: {np.mean(heatmap):.4f}")
        
        # Validate heatmap
        if heatmap.shape == (7, 7):
            print("   ✅ Correct heatmap shape for MobileNetV2")
            test_results['tests_passed'] += 1
        else:
            print(f"   ⚠️  Unexpected heatmap shape: {heatmap.shape}")
            test_results['warnings'].append(f"Heatmap shape: {heatmap.shape}")
        
        if np.max(heatmap) > 0:
            print("   ✅ Heatmap contains activation")
            test_results['tests_passed'] += 1
        else:
            print("   ❌ Heatmap is empty")
            test_results['tests_failed'] += 1
        
        test_results['gradcam_working'] = True
        
    except Exception as e:
        print(f"   ❌ Grad-CAM generation failed: {e}")
        import traceback
        traceback.print_exc()
        test_results['tests_failed'] += 1
        test_results['gradcam_working'] = False

# ============================================================================
# GENERATE FINAL REPORT
# ============================================================================

def generate_final_report():
    """Generate comprehensive final report"""
    print("\n" + "=" * 100)
    print("GENERATING FINAL REPORT")
    print("=" * 100)
    
    # Save JSON report
    report_path = OUTPUT_DIR / 'test_report.json'
    with open(report_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n✓ JSON report saved to {report_path}")
    
    # Generate text report
    report_text_path = OUTPUT_DIR / 'test_report.txt'
    with open(report_text_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("PNEUMONIA DETECTION SYSTEM - COMPREHENSIVE TEST REPORT\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Test Date: {test_results['timestamp']}\n")
        f.write(f"Tests Passed: {test_results['tests_passed']}\n")
        f.write(f"Tests Failed: {test_results['tests_failed']}\n")
        f.write(f"Warnings: {len(test_results['warnings'])}\n\n")
        
        if 'dataset_stats' in test_results:
            f.write("=" * 100 + "\n")
            f.write("DATASET STATISTICS\n")
            f.write("=" * 100 + "\n\n")
            for split, stats in test_results['dataset_stats'].items():
                f.write(f"{split.upper()}:\n")
                f.write(f"  NORMAL: {stats['normal']}\n")
                f.write(f"  PNEUMONIA: {stats['pneumonia']}\n")
                f.write(f"  Total: {stats['total']}\n")
                f.write(f"  Ratio: 1:{stats['ratio']:.2f}\n")
                f.write(f"  Balance: {stats['balance_percentage']:.1f}%\n\n")
        
        if 'model_info' in test_results:
            f.write("=" * 100 + "\n")
            f.write("MODEL INFORMATION\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Total Parameters: {test_results['model_info']['total_params']:,}\n")
            f.write(f"Trainable Parameters: {test_results['model_info']['trainable_params']:,}\n")
            f.write(f"Non-trainable Parameters: {test_results['model_info']['non_trainable_params']:,}\n")
            f.write(f"Model Size: {test_results['model_info']['size_mb']:.2f} MB\n")
            f.write(f"Number of Layers: {test_results['model_info']['layers']}\n\n")
        
        if 'performance_metrics' in test_results:
            f.write("=" * 100 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("=" * 100 + "\n\n")
            pm = test_results['performance_metrics']
            f.write(f"Accuracy: {pm['accuracy']:.4f} ({pm['accuracy']*100:.2f}%)\n")
            f.write(f"Balanced Accuracy: {pm['balanced_accuracy']:.4f} ({pm['balanced_accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {pm['precision']:.4f}\n")
            f.write(f"Recall: {pm['recall']:.4f}\n")
            f.write(f"Specificity: {pm['specificity']:.4f}\n")
            f.write(f"F1-Score: {pm['f1_score']:.4f}\n")
            f.write(f"MCC: {pm['mcc']:.4f}\n")
            f.write(f"ROC AUC: {pm['roc_auc']:.4f}\n")
            f.write(f"PR AUC: {pm['pr_auc']:.4f}\n\n")
            
            cm = pm['confusion_matrix']
            f.write("Confusion Matrix:\n")
            f.write(f"  True Negatives: {cm['tn']}\n")
            f.write(f"  False Positives: {cm['fp']}\n")
            f.write(f"  False Negatives: {cm['fn']}\n")
            f.write(f"  True Positives: {cm['tp']}\n\n")
        
        if test_results['warnings']:
            f.write("=" * 100 + "\n")
            f.write("WARNINGS\n")
            f.write("=" * 100 + "\n\n")
            for warning in test_results['warnings']:
                f.write(f"⚠️  {warning}\n")
            f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("TEST SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        total_tests = test_results['tests_passed'] + test_results['tests_failed']
        success_rate = (test_results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
        
        f.write(f"Total Tests: {total_tests}\n")
        f.write(f"Passed: {test_results['tests_passed']}\n")
        f.write(f"Failed: {test_results['tests_failed']}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n\n")
        
        if test_results['tests_failed'] == 0:
            f.write("✅ ALL TESTS PASSED!\n")
        elif success_rate >= 80:
            f.write("✅ SYSTEM IS OPERATIONAL (with minor issues)\n")
        else:
            f.write("❌ SYSTEM HAS SIGNIFICANT ISSUES\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n")
    
    print(f"✓ Text report saved to {report_text_path}")
    
    # Print final summary
    print("\n" + "=" * 100)
    print("TEST SUMMARY")
    print("=" * 100)
    
    total_tests = test_results['tests_passed'] + test_results['tests_failed']
    success_rate = (test_results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"✅ Passed: {test_results['tests_passed']}")
    print(f"❌ Failed: {test_results['tests_failed']}")
    print(f"⚠️  Warnings: {len(test_results['warnings'])}")
    print(f"\n📊 Success Rate: {success_rate:.1f}%")
    
    if test_results['tests_failed'] == 0:
        print("\n🎉 ALL TESTS PASSED! System is fully operational.")
    elif success_rate >= 80:
        print("\n✅ System is operational with minor issues.")
    else:
        print("\n❌ System has significant issues that need attention.")
    
    print("\n📁 Output Files:")
    print(f"   • {OUTPUT_DIR / 'test_report.json'}")
    print(f"   • {OUTPUT_DIR / 'test_report.txt'}")
    print(f"   • {OUTPUT_DIR / 'dataset_statistics.png'}")
    print(f"   • {OUTPUT_DIR / 'performance_analysis.png'}")
    
    print("\n" + "=" * 100)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main test execution"""
    try:
        # Test 1: Dataset Statistics
        dataset_stats = test_dataset_statistics()
        
        # Test 2: Model Architecture
        model = test_model_architecture()
        
        # Test 3: Model Performance
        if model is not None:
            test_model_performance(model)
        
        # Test 4: API Health
        test_api_health()
        
        # Test 5: Prediction Consistency
        if model is not None:
            test_prediction_consistency(model)
        
        # Test 6: Inference Speed
        if model is not None:
            test_inference_speed(model)
        
        # Test 7: Grad-CAM Functionality
        if model is not None:
            test_gradcam_functionality(model)
        
        # Generate Final Report
        generate_final_report()
        
        print("\n" + "=" * 100)
        print("TESTING COMPLETE!")
        print("=" * 100)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())