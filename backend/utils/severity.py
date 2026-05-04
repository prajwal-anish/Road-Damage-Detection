"""
Damage Severity Scoring Module
Computes severity levels based on confidence, bounding box area, and class type.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


# ──────────────────────────────────────────────────────────────
# Severity Weights per Class
# Potholes are weighted more severely than cracks/manholes
# ──────────────────────────────────────────────────────────────
CLASS_SEVERITY_WEIGHTS = {
    0: 1.0,   # pothole  — most dangerous
    1: 0.75,  # crack    — moderate
    2: 0.4,   # manhole  — infrastructure, lower danger score
}

# Thresholds for severity levels (score 0–1)
SEVERITY_THRESHOLDS = {
    "LOW":      (0.0,  0.25),
    "MEDIUM":   (0.25, 0.50),
    "HIGH":     (0.50, 0.75),
    "CRITICAL": (0.75, 1.01),
}

SEVERITY_COLORS = {
    "LOW":      "#2ECC71",
    "MEDIUM":   "#F39C12",
    "HIGH":     "#E67E22",
    "CRITICAL": "#E74C3C",
}

SEVERITY_DESCRIPTIONS = {
    "LOW": "Minor surface irregularity, monitor periodically",
    "MEDIUM": "Noticeable damage, schedule maintenance within 30 days",
    "HIGH": "Significant damage, prioritize repair within 7 days",
    "CRITICAL": "Severe hazard, immediate repair required",
}


@dataclass
class SeverityLevel:
    label: str
    score: float
    color: str
    description: str

    @classmethod
    def from_score(cls, score: float) -> "SeverityLevel":
        score = max(0.0, min(1.0, score))
        for label, (low, high) in SEVERITY_THRESHOLDS.items():
            if low <= score < high:
                return cls(
                    label=label,
                    score=score,
                    color=SEVERITY_COLORS[label],
                    description=SEVERITY_DESCRIPTIONS[label],
                )
        return cls(
            label="CRITICAL",
            score=score,
            color=SEVERITY_COLORS["CRITICAL"],
            description=SEVERITY_DESCRIPTIONS["CRITICAL"],
        )


def compute_severity(
    confidence: float,
    area_ratio: float,
    class_id: int,
) -> Tuple[str, float]:
    """
    Compute damage severity from detection attributes.

    Formula:
        raw_score = (conf_weight * confidence) + (area_weight * norm_area) 
        score = raw_score * class_weight

    Args:
        confidence:  Model confidence [0, 1]
        area_ratio:  Bounding box area as fraction of image [0, 1]
        class_id:    YOLO class ID (0=pothole, 1=crack, 2=manhole)

    Returns:
        (severity_label, severity_score)
    """
    CONF_WEIGHT = 0.50
    AREA_WEIGHT = 0.50

    # Normalize area — use sigmoid-like normalization
    # area_ratio > 0.02 is considered large damage
    norm_area = min(area_ratio / 0.05, 1.0)

    raw_score = (CONF_WEIGHT * confidence) + (AREA_WEIGHT * norm_area)

    class_weight = CLASS_SEVERITY_WEIGHTS.get(class_id, 0.5)
    final_score = raw_score * class_weight

    # Ensure potholes with high confidence are never LOW severity
    if class_id == 0 and confidence > 0.7:
        final_score = max(final_score, 0.30)

    level = SeverityLevel.from_score(final_score)
    return level.label, round(final_score, 4)


def aggregate_severity(severity_scores: list[float]) -> Tuple[str, float]:
    """
    Aggregate multiple detection severities into an overall score.
    Uses 90th percentile to avoid outlier inflation.
    """
    if not severity_scores:
        return "LOW", 0.0

    import statistics
    if len(severity_scores) == 1:
        score = severity_scores[0]
    else:
        sorted_scores = sorted(severity_scores)
        p90_idx = int(len(sorted_scores) * 0.9)
        score = sorted_scores[min(p90_idx, len(sorted_scores) - 1)]

    level = SeverityLevel.from_score(score)
    return level.label, score


def get_repair_recommendation(severity: str, class_name: str) -> dict:
    """Return actionable repair recommendations based on severity and class."""
    recommendations = {
        ("CRITICAL", "pothole"): {
            "action": "Immediate road closure and emergency repair",
            "timeline": "Within 24 hours",
            "method": "Full-depth patching with hot mix asphalt",
            "priority": 1,
        },
        ("HIGH", "pothole"): {
            "action": "Urgent road patching required",
            "timeline": "Within 3-7 days",
            "method": "Hot mix asphalt patching",
            "priority": 2,
        },
        ("MEDIUM", "pothole"): {
            "action": "Schedule routine patching",
            "timeline": "Within 30 days",
            "method": "Cold mix asphalt patch",
            "priority": 3,
        },
        ("LOW", "pothole"): {
            "action": "Monitor and include in next maintenance cycle",
            "timeline": "Next scheduled maintenance",
            "method": "Crack sealing or surface treatment",
            "priority": 4,
        },
        ("CRITICAL", "crack"): {
            "action": "Immediate crack sealing and structural assessment",
            "timeline": "Within 48 hours",
            "method": "Routed and sealed or full-depth repair",
            "priority": 1,
        },
        ("HIGH", "crack"): {
            "action": "Crack sealing required soon",
            "timeline": "Within 14 days",
            "method": "Hot-applied crack sealant",
            "priority": 2,
        },
    }

    key = (severity, class_name)
    if key in recommendations:
        return recommendations[key]

    # Default recommendation
    return {
        "action": SEVERITY_DESCRIPTIONS.get(severity, "Schedule maintenance"),
        "timeline": "Next maintenance cycle",
        "method": "Standard road maintenance procedure",
        "priority": {"LOW": 4, "MEDIUM": 3, "HIGH": 2, "CRITICAL": 1}.get(severity, 3),
    }
