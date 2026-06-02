import math
import yaml
from pathlib import Path
from typing import Optional

CONFIG_PATH = Path(__file__).parent / "config.yaml"

CLASSES = ["connection_hu", "gps_lu", "idle", "wenco_error", "load_in_n05", "mismatch"]

CLASS_LABELS = {
    "connection_hu": "Связь HU",
    "gps_lu": "GPS LU",
    "idle": "Простой",
    "wenco_error": "Ошибка Wenco",
    "load_in_n05": "Загрузка в N05",
    "mismatch": "Некорректная привязка",
}


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def binarize_features(features: dict) -> dict:
    b = {}
    b["MAX_LAT_HU_AT_ENTRY_5000"] = float(features.get("MAX_LAT_HU_AT_ENTRY", 0) or 0) > 5000
    b["LU_OFFLINE_PCT_15"] = float(features.get("LU_OFFLINE_PCT", 0) or 0) > 15
    b["LU_ALL_NAN"] = bool(features.get("LU_ALL_NAN", False))
    b["N02_DUR_30"] = float(features.get("N02_DUR", 0) or 0) > 30
    n05_dur = float(features.get("N05_DUR", 1) or 1)
    n05_real = float(features.get("N05_REAL", 0) or 0)
    ratio = n05_real / n05_dur if n05_dur > 0 else 1.0
    b["N05_REAL_RATIO_LOW"] = ratio < 0.5
    b["HAS_VIMS"] = bool(features.get("HAS_VIMS", False))
    b["HAS_N06"] = bool(features.get("HAS_N06", False))
    b["MOTION_APPROACH"] = bool(features.get("HU_APPROACHED_LU", False))
    b["MOTION_STOP_NEAR"] = float(features.get("DIST_HU_LU_MIN", 9999) or 9999) < 20
    b["DIST_HU_LU_MIN_FAR"] = float(features.get("DIST_HU_LU_MIN", 9999) or 9999) > 50
    b["N11_OVERLAP_LOW"] = float(features.get("N11_OVERLAP_PCT", 100) or 100) < 50
    b["LOAD_CONFIRMED"] = bool(features.get("LOAD_CONFIRMED", False))
    b["SEQ_N05_N01_GAP"] = bool(features.get("SEQ_N05_N01_GAP", False))
    b["HU_SPEED_AT_N05_END_HIGH"] = float(features.get("HU_SPEED_AT_N05_END", 0) or 0) > 2
    n01_ratio = float(features.get("N01_DUR_RATIO", 1) or 1)
    b["N01_DUR_RATIO_LOW"] = n01_ratio < 0.3
    return b


def hard_rules(features: dict) -> Optional[str]:
    if features.get("HAS_VIMS"):
        return "wenco_error"
    if (
        features.get("N05_MISSING")
        and features.get("HAS_N06")
        and features.get("N01_SOURCE") == "WCS::GPS"
        and features.get("N11_LU_HOUR", 1) == 0
        and float(features.get("LU_OFFLINE_PCT", 100) or 100) < 5
    ):
        return "wenco_error"
    if features.get("LU_ALL_NAN"):
        return "gps_lu"
    dump_end = features.get("DUMP_END_TS")
    load_start = features.get("LOAD_START_TS")
    if dump_end and load_start:
        try:
            delta = (dump_end - load_start).total_seconds()
            if delta < 60:
                return "wenco_error"
        except Exception:
            pass
    return None


def classify(features: dict) -> dict:
    hard = hard_rules(features)
    if hard:
        return {c: (0.95 if c == hard else 0.01) for c in CLASSES}
    cfg = load_config()
    priors = cfg["priors"]
    feat_probs = cfg["feature_probs"]
    binary = binarize_features(features)
    scores = {}
    for cls in CLASSES:
        log_score = math.log(priors.get(cls, 1 / len(CLASSES)))
        for feat_name, feat_value in binary.items():
            if feat_name in feat_probs:
                p_true = feat_probs[feat_name].get(cls, 0.5)
                p = p_true if feat_value else (1 - p_true)
                log_score += math.log(max(p, 1e-9))
        scores[cls] = log_score
    max_s = max(scores.values())
    exp_s = {c: math.exp(s - max_s) for c, s in scores.items()}
    total = sum(exp_s.values())
    return {c: v / total for c, v in exp_s.items()}


def retrain(labeled_cases: list) -> dict:
    new_probs = {}
    cfg = load_config()
    feat_names = list(cfg["feature_probs"].keys())
    for feat in feat_names:
        new_probs[feat] = {}
        for cls in CLASSES:
            cls_cases = [c for c in labeled_cases if cls in (c.get("manual_class") or c.get("auto_class", ""))]
            if not cls_cases:
                new_probs[feat][cls] = cfg["feature_probs"].get(feat, {}).get(cls, 0.5)
                continue
            bin_features = [binarize_features(c.get("features", {})) for c in cls_cases]
            pos = sum(1 for bf in bin_features if bf.get(feat, False))
            new_probs[feat][cls] = (pos + 1) / (len(cls_cases) + 2)
    return new_probs
