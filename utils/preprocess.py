"""
Image Preprocessing Utilities
Shared helpers for image validation, resizing, and format conversion.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
MAX_DIMENSION = 4096          # Reject images wider/taller than this
MIN_DIMENSION = 32            # Reject images smaller than this
YOLO_INPUT_SIZE = 640         # Standard YOLO input dimension


# ─────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────

class ImageValidationError(ValueError):
    """Raised when an image fails validation checks."""
    pass


def validate_image_array(img: np.ndarray) -> None:
    """
    Validate a decoded numpy image array.
    Raises ImageValidationError on failure.
    """
    if img is None or not isinstance(img, np.ndarray):
        raise ImageValidationError("Image could not be decoded.")

    if img.ndim not in (2, 3):
        raise ImageValidationError(f"Expected 2D/3D array, got shape {img.shape}.")

    h, w = img.shape[:2]
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        raise ImageValidationError(
            f"Image too small: {w}×{h}px. Minimum: {MIN_DIMENSION}px."
        )
    if h > MAX_DIMENSION or w > MAX_DIMENSION:
        raise ImageValidationError(
            f"Image too large: {w}×{h}px. Maximum: {MAX_DIMENSION}px."
        )


def validate_extension(filename: str) -> str:
    """
    Check that filename has an allowed extension.
    Returns the lowercase extension on success.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )
    return ext


# ─────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────

def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes to BGR numpy array."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageValidationError(
            "Could not decode image data. "
            "Ensure the file is a valid, uncorrupted image."
        )
    validate_image_array(img)
    return img


def resize_for_inference(
    img: np.ndarray,
    target: int = YOLO_INPUT_SIZE,
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Letterbox-resize image to target×target while preserving aspect ratio.
    Pads with pad_color to fill unused space.

    Returns:
        resized:  (target, target, 3) uint8 array
        scale:    scale factor applied (original → resized content region)
        padding:  (pad_top_or_left, pad_bottom_or_right) pixels added
    """
    h, w = img.shape[:2]
    scale = min(target / h, target / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target, target, 3), pad_color, dtype=np.uint8)
    pad_y = (target - new_h) // 2
    pad_x = (target - new_w) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, scale, (pad_y, pad_x)


def undo_letterbox(
    x1: float, y1: float, x2: float, y2: float,
    scale: float,
    pad: Tuple[int, int],
    orig_w: int,
    orig_h: int,
) -> Tuple[float, float, float, float]:
    """
    Convert letterboxed bounding box coordinates back to original image space.
    """
    pad_y, pad_x = pad
    x1 = max(0.0, (x1 - pad_x) / scale)
    y1 = max(0.0, (y1 - pad_y) / scale)
    x2 = min(float(orig_w), (x2 - pad_x) / scale)
    y2 = min(float(orig_h), (y2 - pad_y) / scale)
    return x1, y1, x2, y2


def normalize_to_float(img: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR image to float32 normalized to [0, 1]."""
    return img.astype(np.float32) / 255.0


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB (PIL / display)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rotate_image(img: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate image by given degrees (0, 90, 180, 270)."""
    degrees = int(degrees) % 360
    if degrees == 0:
        return img
    if degrees == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Arbitrary rotation
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -degrees, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))


def auto_orient(img: np.ndarray, exif_orientation: Optional[int] = None) -> np.ndarray:
    """
    Apply EXIF orientation correction if available.
    EXIF orientation values 1-8 per the spec.
    """
    if exif_orientation is None:
        return img
    rotations = {3: 180, 6: 270, 8: 90}
    flips = {2: (1, 0), 4: (1, 0), 5: (0, 1), 7: (0, 1)}
    if exif_orientation in rotations:
        img = rotate_image(img, rotations[exif_orientation])
    if exif_orientation in flips:
        h_flip, v_flip = flips[exif_orientation]
        img = cv2.flip(img, 1 if h_flip else 0)
    return img


# ─────────────────────────────────────────────────────
# Augmentation helpers (used in Colab training)
# ─────────────────────────────────────────────────────

def random_brightness_contrast(
    img: np.ndarray,
    alpha: float = 1.0,
    beta: int = 0,
) -> np.ndarray:
    """
    Adjust brightness (beta) and contrast (alpha).
    alpha: contrast multiplier (0.5=low, 1.0=original, 1.5=high)
    beta:  brightness offset in [-127, 127]
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def add_gaussian_noise(img: np.ndarray, std: float = 10.0) -> np.ndarray:
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def random_crop(
    img: np.ndarray,
    labels: Optional[np.ndarray] = None,
    crop_ratio: float = 0.85,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Random crop maintaining at least crop_ratio of original dimensions.
    Adjusts YOLO labels to new crop coordinates if provided.

    labels: array of shape (N, 5) in YOLO format [class, cx, cy, w, h]
    """
    h, w = img.shape[:2]
    new_h = int(h * crop_ratio + np.random.uniform(0, h * (1 - crop_ratio)))
    new_w = int(w * crop_ratio + np.random.uniform(0, w * (1 - crop_ratio)))
    y_off = np.random.randint(0, h - new_h + 1)
    x_off = np.random.randint(0, w - new_w + 1)

    cropped = img[y_off:y_off + new_h, x_off:x_off + new_w]

    if labels is not None and len(labels) > 0:
        adjusted = []
        for lbl in labels:
            cls, cx, cy, bw, bh = lbl
            # Convert to pixel coords, adjust, convert back
            px = (cx - bw / 2) * w - x_off
            py = (cy - bh / 2) * h - y_off
            px2 = (cx + bw / 2) * w - x_off
            py2 = (cy + bh / 2) * h - y_off
            # Clip to crop bounds
            px, py = max(0, px), max(0, py)
            px2, py2 = min(new_w, px2), min(new_h, py2)
            if px2 > px and py2 > py:
                new_cx = (px + px2) / 2 / new_w
                new_cy = (py + py2) / 2 / new_h
                new_bw = (px2 - px) / new_w
                new_bh = (py2 - py) / new_h
                adjusted.append([cls, new_cx, new_cy, new_bw, new_bh])
        labels = np.array(adjusted) if adjusted else np.zeros((0, 5))

    return cropped, labels
