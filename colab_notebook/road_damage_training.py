# %% [markdown]
# # 🚗 Road Surface Damage Detection — YOLOv8 Training Notebook
# **Project:** Pothole, Crack & Manhole Detection System
# **Dataset:** [Road Damage Dataset](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes)
# **Model:** YOLOv8n / YOLOv8s (Ultralytics)
#
# ### Workflow:
# 1. Install dependencies
# 2. Mount Google Drive
# 3. Download dataset from Kaggle
# 4. Prepare & verify dataset
# 5. Train YOLOv8 model
# 6. Evaluate & visualize results
# 7. Export model to Google Drive

# %% [markdown]
# ## 📦 Step 1: Install Dependencies

# %%
# Install all required packages
!pip install ultralytics==8.2.0 --quiet
!pip install kaggle --quiet
!pip install opencv-python-headless --quiet
!pip install matplotlib seaborn pandas numpy --quiet
!pip install albumentations --quiet

# Verify GPU availability
import torch
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU found — training will be slow on CPU. Use Runtime > Change runtime type > T4 GPU")

# %%
# Import libraries
import os
import shutil
import json
import yaml
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import seaborn as sns

print("✅ All libraries imported successfully")

# %% [markdown]
# ## 📁 Step 2: Mount Google Drive

# %%
from google.colab import drive
drive.mount('/content/drive')

# Create project directory on Drive
DRIVE_PROJECT_DIR = "/content/drive/MyDrive/road_damage_detection"
os.makedirs(DRIVE_PROJECT_DIR, exist_ok=True)
os.makedirs(f"{DRIVE_PROJECT_DIR}/models", exist_ok=True)
os.makedirs(f"{DRIVE_PROJECT_DIR}/results", exist_ok=True)

print(f"✅ Google Drive mounted")
print(f"✅ Project directory: {DRIVE_PROJECT_DIR}")

# %% [markdown]
# ## 📥 Step 3: Download Dataset from Kaggle
# **Before running:** Upload your `kaggle.json` API key file.
# Get it from: https://www.kaggle.com/settings → API → Create New Token

# %%
# Upload Kaggle credentials
from google.colab import files

print("📤 Upload your kaggle.json file:")
uploaded = files.upload()

# Set up Kaggle credentials
os.makedirs('/root/.kaggle', exist_ok=True)
with open('/root/.kaggle/kaggle.json', 'wb') as f:
    f.write(uploaded['kaggle.json'])
os.chmod('/root/.kaggle/kaggle.json', 0o600)
print("✅ Kaggle credentials configured")

# %%
# Download dataset
DATASET_DIR = "/content/road_damage_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

print("⬇️  Downloading Road Damage Dataset from Kaggle...")
!kaggle datasets download -d lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes -p {DATASET_DIR} --unzip

print(f"✅ Dataset downloaded to {DATASET_DIR}")

# Explore dataset structure
print("\n📂 Dataset structure:")
for root, dirs, files_list in os.walk(DATASET_DIR):
    level = root.replace(DATASET_DIR, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 3:
        subindent = ' ' * 2 * (level + 1)
        for file in files_list[:5]:
            print(f'{subindent}{file}')

# %% [markdown]
# ## 🗂️ Step 4: Dataset Preparation

# %%
# === CONFIGURATION ===
PROJECT_DIR = "/content/road_damage_project"
DATA_DIR = f"{PROJECT_DIR}/data"
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10

CLASS_NAMES = ["pothole", "crack", "manhole"]
CLASS_COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}  # BGR

for split in ['train', 'val', 'test']:
    os.makedirs(f"{DATA_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/{split}/labels", exist_ok=True)

print(f"✅ Project directory created: {PROJECT_DIR}")

# %%
def find_image_label_pairs(dataset_dir):
    """Find all image-label pairs in the dataset directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    pairs = []
    
    for root, dirs, files_list in os.walk(dataset_dir):
        for file in files_list:
            if Path(file).suffix.lower() in image_extensions:
                img_path = Path(root) / file
                # Look for corresponding label file
                label_path = img_path.with_suffix('.txt')
                if not label_path.exists():
                    # Try labels directory
                    label_path = Path(str(img_path).replace('images', 'labels')).with_suffix('.txt')
                if label_path.exists():
                    pairs.append((str(img_path), str(label_path)))
    
    return pairs

def validate_yolo_label(label_path):
    """Validate YOLO format label file."""
    issues = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            return True, []  # Empty = background, valid
        
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                issues.append(f"Line {i+1}: Expected 5 values, got {len(parts)}")
                continue
            
            cls, x, y, w, h = parts
            cls = int(cls)
            x, y, w, h = float(x), float(y), float(w), float(h)
            
            if cls not in [0, 1, 2]:
                issues.append(f"Line {i+1}: Invalid class {cls}")
            if not (0 <= x <= 1 and 0 <= y <= 1):
                issues.append(f"Line {i+1}: Center out of bounds ({x}, {y})")
            if not (0 < w <= 1 and 0 < h <= 1):
                issues.append(f"Line {i+1}: Invalid dimensions ({w}, {h})")
    
    except Exception as e:
        issues.append(str(e))
    
    return len(issues) == 0, issues

# %%
# Find all image-label pairs
print("🔍 Scanning dataset for image-label pairs...")
all_pairs = find_image_label_pairs(DATASET_DIR)
print(f"✅ Found {len(all_pairs)} image-label pairs")

# Validate labels
print("\n🔍 Validating YOLO format labels...")
valid_pairs = []
invalid_count = 0

for img_path, label_path in all_pairs:
    is_valid, issues = validate_yolo_label(label_path)
    if is_valid:
        valid_pairs.append((img_path, label_path))
    else:
        invalid_count += 1
        if invalid_count <= 3:
            print(f"⚠️  Invalid: {label_path}")
            for issue in issues[:2]:
                print(f"     → {issue}")

print(f"\n✅ Valid pairs: {len(valid_pairs)}")
print(f"❌ Invalid pairs: {invalid_count}")

# %%
# Analyze class distribution
class_counts = {0: 0, 1: 0, 2: 0}
for _, label_path in valid_pairs:
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                if cls in class_counts:
                    class_counts[cls] += 1

print("\n📊 Class Distribution:")
print("-" * 40)
total = sum(class_counts.values())
for cls_id, count in class_counts.items():
    pct = count / total * 100 if total > 0 else 0
    bar = "█" * int(pct / 2)
    print(f"  {CLASS_NAMES[cls_id]:10s} (class {cls_id}): {count:5d} boxes ({pct:.1f}%) {bar}")
print("-" * 40)
print(f"  {'Total':10s}           : {total:5d} boxes")

# Visualize
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
colors = ['#E74C3C', '#2ECC71', '#3498DB']

ax[0].bar(CLASS_NAMES, [class_counts[i] for i in range(3)], color=colors, edgecolor='white', linewidth=1.5)
ax[0].set_title('Class Distribution (Annotations)', fontweight='bold', fontsize=13)
ax[0].set_ylabel('Number of Annotations')
ax[0].grid(axis='y', alpha=0.3)
for i, v in enumerate([class_counts[i] for i in range(3)]):
    ax[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

ax[1].pie([class_counts[i] for i in range(3)], labels=CLASS_NAMES, colors=colors,
          autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
ax[1].set_title('Class Distribution (%)', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/class_distribution.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Class distribution chart saved")

# %%
# Split dataset
random.seed(42)
random.shuffle(valid_pairs)

n_total = len(valid_pairs)
n_train = int(n_total * TRAIN_SPLIT)
n_val = int(n_total * VAL_SPLIT)
n_test = n_total - n_train - n_val

train_pairs = valid_pairs[:n_train]
val_pairs = valid_pairs[n_train:n_train + n_val]
test_pairs = valid_pairs[n_train + n_val:]

print(f"📊 Dataset Split Summary:")
print(f"   Training   : {len(train_pairs)} images ({len(train_pairs)/n_total*100:.1f}%)")
print(f"   Validation : {len(val_pairs)} images ({len(val_pairs)/n_total*100:.1f}%)")
print(f"   Testing    : {len(test_pairs)} images ({len(test_pairs)/n_total*100:.1f}%)")
print(f"   Total      : {n_total} images")

# %%
def copy_split(pairs, split_name):
    """Copy image-label pairs to split directory."""
    copied = 0
    for img_path, label_path in pairs:
        img_dest = f"{DATA_DIR}/{split_name}/images/{Path(img_path).name}"
        label_dest = f"{DATA_DIR}/{split_name}/labels/{Path(label_path).name}"
        shutil.copy2(img_path, img_dest)
        shutil.copy2(label_path, label_dest)
        copied += 1
    return copied

print("📋 Copying files to splits...")
for split, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
    n = copy_split(pairs, split)
    print(f"  ✅ {split}: {n} pairs copied")

# %%
# Create YAML configuration file
data_yaml = {
    'path': DATA_DIR,
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': 3,
    'names': CLASS_NAMES
}

yaml_path = f"{PROJECT_DIR}/road_damage.yaml"
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"✅ Dataset YAML saved: {yaml_path}")
print("\n📄 YAML contents:")
with open(yaml_path, 'r') as f:
    print(f.read())

# %%
# Visualize sample images with bounding boxes
def draw_boxes_on_image(img_path, label_path, class_names, colors):
    """Draw YOLO bounding boxes on image."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                # Convert to pixel coordinates
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                color = tuple(c/255 for c in colors.get(cls, (255,255,255)))
                
                # Draw rectangle
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                          linewidth=2.5, edgecolor=color, facecolor='none')
                plt.gca().add_patch(rect)
                
                # Add label
                label = class_names[cls] if cls < len(class_names) else f"cls{cls}"
                plt.text(x1, y1-5, label, color=color, fontsize=8,
                         fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', 
                         facecolor='black', alpha=0.6))
    return img

# Show sample images
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
sample_pairs = random.sample(train_pairs[:50], min(8, len(train_pairs)))

for ax, (img_path, label_path) in zip(axes.flatten(), sample_pairs):
    plt.sca(ax)
    img = draw_boxes_on_image(img_path, label_path, CLASS_NAMES, CLASS_COLORS)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(Path(img_path).stem[:20], fontsize=8)

plt.suptitle('Sample Training Images with Ground Truth Annotations', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/sample_annotations.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Sample annotation visualization saved")

# %% [markdown]
# ## 🚀 Step 5: Model Training

# %%
# === TRAINING CONFIGURATION ===
TRAINING_CONFIG = {
    'model': 'yolov8s.pt',      # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,                 # Reduce to 8 if OOM error
    'lr0': 0.01,
    'lrf': 0.001,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'conf': 0.25,
    'iou': 0.45,
    'augment': True,
    'mosaic': 1.0,
    'mixup': 0.1,
    'copy_paste': 0.1,
    'degrees': 15.0,
    'translate': 0.1,
    'scale': 0.5,
    'flipud': 0.3,
    'fliplr': 0.5,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'workers': 2,
    'patience': 20,
    'save_period': 10,
    'project': f"{PROJECT_DIR}/runs",
    'name': 'road_damage_yolov8s',
    'exist_ok': True,
    'pretrained': True,
    'optimizer': 'AdamW',
    'verbose': True,
    'seed': 42,
    'device': 0 if torch.cuda.is_available() else 'cpu',
}

print("🎯 Training Configuration:")
print("-" * 50)
for k, v in TRAINING_CONFIG.items():
    print(f"  {k:20s}: {v}")
print("-" * 50)

# %%
# Initialize model
print(f"\n🏗️  Loading pretrained YOLOv8 model: {TRAINING_CONFIG['model']}")
model = YOLO(TRAINING_CONFIG['model'])
print("✅ Model loaded successfully")
print(f"\n📊 Model Architecture Summary:")
print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
print(f"   Layers: {len(list(model.model.modules()))}")

# %%
# ============================================================
# TRAINING — This cell will take 20-60 mins on GPU
# ============================================================
print("🚀 Starting Training...")
print("=" * 60)
print(f"   Model: YOLOv8s | Epochs: {TRAINING_CONFIG['epochs']}")
print(f"   Image size: {TRAINING_CONFIG['imgsz']}px | Batch: {TRAINING_CONFIG['batch']}")
print(f"   Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print("=" * 60)

results = model.train(
    data=yaml_path,
    epochs=TRAINING_CONFIG['epochs'],
    imgsz=TRAINING_CONFIG['imgsz'],
    batch=TRAINING_CONFIG['batch'],
    lr0=TRAINING_CONFIG['lr0'],
    lrf=TRAINING_CONFIG['lrf'],
    momentum=TRAINING_CONFIG['momentum'],
    weight_decay=TRAINING_CONFIG['weight_decay'],
    warmup_epochs=TRAINING_CONFIG['warmup_epochs'],
    conf=TRAINING_CONFIG['conf'],
    iou=TRAINING_CONFIG['iou'],
    mosaic=TRAINING_CONFIG['mosaic'],
    mixup=TRAINING_CONFIG['mixup'],
    copy_paste=TRAINING_CONFIG['copy_paste'],
    degrees=TRAINING_CONFIG['degrees'],
    translate=TRAINING_CONFIG['translate'],
    scale=TRAINING_CONFIG['scale'],
    flipud=TRAINING_CONFIG['flipud'],
    fliplr=TRAINING_CONFIG['fliplr'],
    hsv_h=TRAINING_CONFIG['hsv_h'],
    hsv_s=TRAINING_CONFIG['hsv_s'],
    hsv_v=TRAINING_CONFIG['hsv_v'],
    workers=TRAINING_CONFIG['workers'],
    patience=TRAINING_CONFIG['patience'],
    save_period=TRAINING_CONFIG['save_period'],
    project=TRAINING_CONFIG['project'],
    name=TRAINING_CONFIG['name'],
    exist_ok=TRAINING_CONFIG['exist_ok'],
    pretrained=TRAINING_CONFIG['pretrained'],
    optimizer=TRAINING_CONFIG['optimizer'],
    verbose=TRAINING_CONFIG['verbose'],
    seed=TRAINING_CONFIG['seed'],
    device=TRAINING_CONFIG['device'],
)

print("\n✅ Training Complete!")
print(f"   Best mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
print(f"   Best mAP@50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")

# %% [markdown]
# ## 📈 Step 6: Training Visualization

# %%
# Load and plot training results
RESULTS_DIR = f"{TRAINING_CONFIG['project']}/{TRAINING_CONFIG['name']}"
results_csv = f"{RESULTS_DIR}/results.csv"

if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle('YOLOv8 Training Metrics — Road Damage Detection', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Color palette
    train_color = '#E74C3C'
    val_color = '#2ECC71'
    metric_color = '#3498DB'
    
    plot_configs = [
        ('train/box_loss', 'Train Box Loss', train_color),
        ('train/cls_loss', 'Train Class Loss', train_color),
        ('train/dfl_loss', 'Train DFL Loss', train_color),
        ('val/box_loss', 'Val Box Loss', val_color),
        ('val/cls_loss', 'Val Class Loss', val_color),
        ('val/dfl_loss', 'Val DFL Loss', val_color),
        ('metrics/mAP50(B)', 'mAP@50', metric_color),
        ('metrics/mAP50-95(B)', 'mAP@50-95', metric_color),
    ]
    
    for ax, (col, title, color) in zip(axes.flatten(), plot_configs):
        if col in df.columns:
            ax.plot(df['epoch'], df[col], color=color, linewidth=2, alpha=0.9)
            ax.fill_between(df['epoch'], df[col], alpha=0.15, color=color)
            ax.set_title(title, fontweight='bold', fontsize=10)
            ax.set_xlabel('Epoch')
            ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Mark best value
            if 'mAP' in col:
                best_idx = df[col].idxmax()
                ax.axvline(df['epoch'][best_idx], color='gold', linestyle='--', alpha=0.7)
                ax.scatter(df['epoch'][best_idx], df[col][best_idx], 
                          color='gold', s=80, zorder=5)
                ax.text(df['epoch'][best_idx], df[col][best_idx],
                       f' Best: {df[col][best_idx]:.4f}', fontsize=8, color='gold')
        else:
            ax.text(0.5, 0.5, f'{col}\nnot found', ha='center', va='center',
                   transform=ax.transAxes, color='gray')
            ax.set_title(title, fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{PROJECT_DIR}/training_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary stats
    print("\n📊 Training Summary:")
    print("-" * 50)
    for col in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 
                'metrics/precision(B)', 'metrics/recall(B)']:
        if col in df.columns:
            metric_name = col.split('/')[-1].replace('(B)', '')
            print(f"  Best {metric_name:20s}: {df[col].max():.4f}")
else:
    print("⚠️  results.csv not found. Training may not have completed.")

# %%
# Plot confusion matrix if available
confusion_matrix_path = f"{RESULTS_DIR}/confusion_matrix_normalized.png"
if os.path.exists(confusion_matrix_path):
    img = plt.imread(confusion_matrix_path)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Confusion Matrix (Normalized)', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()
    print("✅ Confusion matrix displayed")
else:
    print("⚠️  Confusion matrix not found at:", confusion_matrix_path)

# %% [markdown]
# ## 🧪 Step 7: Model Evaluation

# %%
# Load best model for evaluation
best_model_path = f"{RESULTS_DIR}/weights/best.pt"
print(f"🔍 Loading best model from: {best_model_path}")

if not os.path.exists(best_model_path):
    print("⚠️  best.pt not found. Using last.pt instead.")
    best_model_path = f"{RESULTS_DIR}/weights/last.pt"

eval_model = YOLO(best_model_path)
print("✅ Best model loaded")

# %%
# Run validation
print("🔍 Running validation on val set...")
val_results = eval_model.val(
    data=yaml_path,
    split='val',
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.6,
    verbose=True
)

print("\n📊 Validation Metrics:")
print("-" * 50)
metrics = val_results.results_dict
for key, val in metrics.items():
    print(f"  {key:35s}: {val:.4f}")

# %%
# Per-class metrics
print("\n📊 Per-Class Performance:")
print("-" * 60)
print(f"  {'Class':15s} {'Precision':>10s} {'Recall':>10s} {'mAP@50':>10s} {'mAP@50-95':>12s}")
print("-" * 60)

try:
    for i, cls_name in enumerate(CLASS_NAMES):
        precision = val_results.results_dict.get(f'metrics/precision(B)', 0)
        recall = val_results.results_dict.get(f'metrics/recall(B)', 0)
        map50 = val_results.results_dict.get(f'metrics/mAP50(B)', 0)
        map50_95 = val_results.results_dict.get(f'metrics/mAP50-95(B)', 0)
        print(f"  {cls_name:15s} {precision:>10.4f} {recall:>10.4f} {map50:>10.4f} {map50_95:>12.4f}")
except:
    print("  Per-class metrics not available in this format")

# %%
# Run inference on test images and display predictions
print("🔍 Running inference on test images...")

test_images_dir = f"{DATA_DIR}/test/images"
test_images = list(Path(test_images_dir).glob("*.jpg")) + \
              list(Path(test_images_dir).glob("*.jpeg")) + \
              list(Path(test_images_dir).glob("*.png"))

if len(test_images) == 0:
    print("⚠️  No test images found")
else:
    sample_test = random.sample(test_images, min(8, len(test_images)))
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle('YOLOv8 Test Predictions — Road Damage Detection', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    for ax, img_path in zip(axes.flatten(), sample_test):
        # Run inference
        pred_results = eval_model.predict(
            str(img_path), 
            conf=0.25, 
            iou=0.45,
            verbose=False
        )
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        ax.imshow(img)
        
        # Draw predictions
        colors_rgb = {0: (231/255, 76/255, 60/255),  
                      1: (46/255, 204/255, 113/255),  
                      2: (52/255, 152/255, 219/255)}   
        
        for box in pred_results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            color = colors_rgb.get(cls, (1, 1, 1))
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                      linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            label = f"{CLASS_NAMES[cls]} {conf:.2f}"
            ax.text(x1, max(y1-4, 0), label, fontsize=7, color='white',
                   fontweight='bold', bbox=dict(facecolor=color, alpha=0.8, pad=1))
        
        n_dets = len(pred_results[0].boxes)
        ax.set_title(f"{img_path.stem[:18]} ({n_dets} det.)", fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{PROJECT_DIR}/test_predictions.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Inference complete on {len(sample_test)} test images")

# %% [markdown]
# ## 💾 Step 8: Export Model to Google Drive

# %%
# Copy model and results to Google Drive
print("💾 Saving model and artifacts to Google Drive...")

# Copy best.pt
src_model = f"{RESULTS_DIR}/weights/best.pt"
dst_model = f"{DRIVE_PROJECT_DIR}/models/best.pt"
shutil.copy2(src_model, dst_model)
model_size_mb = os.path.getsize(dst_model) / 1e6
print(f"✅ best.pt saved → {dst_model} ({model_size_mb:.1f} MB)")

# Copy last.pt
src_last = f"{RESULTS_DIR}/weights/last.pt"
dst_last = f"{DRIVE_PROJECT_DIR}/models/last.pt"
if os.path.exists(src_last):
    shutil.copy2(src_last, dst_last)
    print(f"✅ last.pt saved → {dst_last}")

# Copy visualizations
for viz_file in ['class_distribution.png', 'sample_annotations.png', 
                  'training_curves.png', 'test_predictions.png']:
    src = f"{PROJECT_DIR}/{viz_file}"
    if os.path.exists(src):
        shutil.copy2(src, f"{DRIVE_PROJECT_DIR}/results/{viz_file}")
        print(f"✅ {viz_file} saved to Drive")

# Copy results CSV
if os.path.exists(results_csv):
    shutil.copy2(results_csv, f"{DRIVE_PROJECT_DIR}/results/training_results.csv")
    print(f"✅ training_results.csv saved to Drive")

# Save training config
config_path = f"{DRIVE_PROJECT_DIR}/training_config.json"
with open(config_path, 'w') as f:
    json.dump({k: str(v) for k, v in TRAINING_CONFIG.items()}, f, indent=2)
print(f"✅ Training config saved → {config_path}")

print(f"\n🎉 All artifacts saved to: {DRIVE_PROJECT_DIR}")

# %%
# Provide download links for local use
from google.colab import files

print("📥 Downloading model file to your computer...")
print("(This will start a file download in your browser)")

files.download(f"{DRIVE_PROJECT_DIR}/models/best.pt")
print("✅ Download initiated for best.pt")
print("\n📁 Place the downloaded best.pt file in:")
print("   road_damage_detection/model/best.pt")

# %% [markdown]
# ## ✅ Step 9: Final Summary

# %%
print("=" * 60)
print("🏁 TRAINING COMPLETE — SUMMARY")
print("=" * 60)
print(f"\n📦 Dataset:")
print(f"   Total images : {n_total}")
print(f"   Train        : {len(train_pairs)}")
print(f"   Validation   : {len(val_pairs)}")
print(f"   Test         : {len(test_pairs)}")
print(f"\n🧠 Model:")
print(f"   Architecture : YOLOv8s")
print(f"   Epochs       : {TRAINING_CONFIG['epochs']}")
print(f"   Input size   : {TRAINING_CONFIG['imgsz']}px")
print(f"\n📊 Results (Validation):")
try:
    print(f"   mAP@50       : {val_results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"   mAP@50-95    : {val_results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"   Precision    : {val_results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"   Recall       : {val_results.results_dict.get('metrics/recall(B)', 0):.4f}")
except:
    pass
print(f"\n📁 Artifacts saved to: {DRIVE_PROJECT_DIR}")
print(f"\n🔜 Next Steps:")
print(f"   1. Download best.pt to road_damage_detection/model/best.pt")
print(f"   2. cd road_damage_detection/backend && uvicorn main:app --reload")
print(f"   3. cd road_damage_detection/frontend && npm run dev")
print("=" * 60)
