import io
import pandas as pd
from calculator import Case

FEATURE_COLS = [
    "MAX_LAT_HU", "MAX_LAT_HU_AT_ENTRY", "LU_ALL_NAN", "LU_OFFLINE_PCT", "LU_MEDIAN_LAT",
    "HAS_N06", "N06_DUR", "HU_MOVES_IN_N06", "N11_LU_HOUR", "N11_OVERLAP_PCT", "HAS_VIMS",
    "N05_DUR", "N05_REAL", "N02_DUR", "N02_DISPLACEMENT",
    "DIST_HU_LU_START", "DIST_HU_LU_MIN", "DIST_HU_LU_AT_LOAD", "HU_APPROACHED_LU",
    "HU_SPEED_AT_N05_END", "HU_SPEED_AT_N01_START", "LU_SPEED_DURING_STOP",
    "N01_DUR", "N01_SOURCE", "N13_DUR", "TYPICAL_N13", "N01_DUR_RATIO", "N01_N13_DELTA", "QTY_SOURCE",
    "SEQ_N05_N01_GAP", "SEQ_N01_N04", "MATCH_N05_N11", "MATCH_N01_N13",
    "LOAD_DUR_OK", "LOAD_CONFIRMED", "MOTION_APPROACH", "MOTION_STOP_NEAR",
    "MOTION_N02_STILL", "MOTION_LU_GPS_OK",
]


def export_xlsx(cases: list) -> bytes:
    rows1 = []
    rows2 = []
    for c in cases:
        rows1.append({
            "CASE_ID": c.case_id,
            "HU": c.hu,
            "LU": c.lu,
            "ANOMALY_TYPE": c.anomaly_type,
            "N05_START": c.n05_start,
            "N05_END": c.n05_end,
            "N05_DUR": round(c.n05_dur, 1),
            "N05_REAL": round(c.features.get("N05_REAL", 0) or 0, 1),
            "N02_DUR": round(c.features.get("N02_DUR", 0) or 0, 1),
            "THRESHOLD": round(c.threshold, 1),
            "DELTA": round(c.delta, 1),
            "AUTO_CLASS": c.auto_class,
            "AUTO_CONF": round(c.auto_probs.get(c.auto_class, 0), 3),
            "MANUAL_CLASS": c.manual_class or None,
            "MANUAL_CONF": c.manual_conf or None,
            "NOTE": c.note or None,
            "LABELED": c.labeled,
            "LABEL_SOURCE": c.label_source or None,
        })
        feat_row = {"CASE_ID": c.case_id, "HU": c.hu, "LU": c.lu}
        for col in FEATURE_COLS:
            val = c.features.get(col)
            if isinstance(val, bool):
                val = int(val)
            feat_row[col] = val
        rows2.append(feat_row)
    df1 = pd.DataFrame(rows1)
    df2 = pd.DataFrame(rows2)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="Аномалии", index=False)
        df2.to_excel(writer, sheet_name="Признаки", index=False)
    return buf.getvalue()
