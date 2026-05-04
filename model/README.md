# Model Directory

Place your trained `best.pt` file here after running the Google Colab training notebook.

## Instructions

1. Complete training in `colab_notebook/road_damage_training.py`
2. The notebook will automatically save `best.pt` to your Google Drive
3. Download it and place it here as `model/best.pt`

## Expected File

```
model/
└── best.pt    ← YOLOv8s trained weights (~22MB for YOLOv8s)
```

## Verify Model

```python
from ultralytics import YOLO
model = YOLO("model/best.pt")
print(model.info())  # Should show 3 classes: pothole, crack, manhole
```
