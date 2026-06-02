import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Case:
    case_id: str
    hu: str
    lu: str
    anomaly_type: str
    n05_start: pd.Timestamp
    n05_end: pd.Timestamp
    n05_dur: float
    threshold: float
    delta: float
    features: dict = field(default_factory=dict)
    auto_class: str = ""
    auto_probs: dict = field(default_factory=dict)
    manual_class: str = ""
    manual_conf: str = ""
    note: str = ""
    labeled: bool = False
    label_source: str = ""


def _haversine_m(n1, e1, n2, e2):
    return math.sqrt((n1 - n2) ** 2 + (e1 - e2) ** 2)


def _ts_overlap_pct(s1, e1, s2, e2):
    if pd.isna(s1) or pd.isna(e1) or pd.isna(s2) or pd.isna(e2):
        return 0.0
    overlap_start = max(s1, s2)
    overlap_end = min(e1, e2)
    if overlap_end <= overlap_start:
        return 0.0
    dur1 = (e1 - s1).total_seconds()
    if dur1 <= 0:
        return 0.0
    return 100.0 * (overlap_end - overlap_start).total_seconds() / dur1


def compute_cases(
    eq_status: pd.DataFrame,
    haul_cycle: pd.DataFrame,
    eq_coord: pd.DataFrame,
    location_desc: pd.DataFrame,
    equip_health: pd.DataFrame,
    cfg: dict,
) -> list:
    cases = []

    for df, cols in [
        (eq_status, ["START_TIMESTAMP", "END_TIMESTAMP"]),
        (haul_cycle, ["START_TIMESTAMP", "LOAD_START_TIMESTAMP", "DUMP_END_TIMESTAMP"]),
        (eq_coord, ["TIMESTAMP"]),
        (equip_health, ["TIMESTAMP", "GPS_LAST_UPDATE_TIMESTAMP", "DIO_LAST_UPDATE_TIMESTAMP"]),
    ]:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False)

    eq_coord["NORTHING"] = pd.to_numeric(eq_coord["NORTHING"], errors="coerce")
    eq_coord["EASTING"] = pd.to_numeric(eq_coord["EASTING"], errors="coerce")
    eq_coord["SPEED"] = pd.to_numeric(eq_coord["SPEED"], errors="coerce")
    eq_coord["PREVIOUS_LATENCY"] = pd.to_numeric(eq_coord["PREVIOUS_LATENCY"], errors="coerce")
    location_desc["NORTHING"] = pd.to_numeric(location_desc["NORTHING"], errors="coerce")
    location_desc["EASTING"] = pd.to_numeric(location_desc["EASTING"], errors="coerce")
    location_desc["RADIUS"] = pd.to_numeric(location_desc["RADIUS"], errors="coerce")

    n05_rows = eq_status[eq_status["STATUS_CODE"] == "N05"].copy()
    n05_rows["SPOT_DUR"] = (
        n05_rows["END_TIMESTAMP"] - n05_rows["START_TIMESTAMP"]
    ).dt.total_seconds()
    n05_rows = n05_rows[n05_rows["SPOT_DUR"] > 0]

    lu_type_map = {}
    for lu in eq_status["EQUIP_IDENT"].unique():
        if str(lu).startswith("EX") or "EX" in str(lu).upper():
            lu_type_map[lu] = "EX"
        else:
            lu_type_map[lu] = "WH"

    ex_mult = cfg.get("thresholds", {}).get("ex_multiplier", 2.0)
    wh_mult = cfg.get("thresholds", {}).get("wh_multiplier", 2.0)

    def get_lu_for_hu(hu, ts):
        hc = haul_cycle[
            (haul_cycle["HAULING_UNIT_IDENT"] == hu)
            & (haul_cycle["START_TIMESTAMP"] <= ts)
            & (haul_cycle["DUMP_END_TIMESTAMP"] >= ts)
        ]
        if not hc.empty:
            return hc.iloc[0]["LOADING_UNIT_IDENT"]
        prev = haul_cycle[
            (haul_cycle["HAULING_UNIT_IDENT"] == hu)
            & (haul_cycle["START_TIMESTAMP"] <= ts)
        ].sort_values("START_TIMESTAMP")
        if not prev.empty:
            return prev.iloc[-1]["LOADING_UNIT_IDENT"]
        return None

    n05_with_lu = n05_rows.copy()
    n05_with_lu["LU"] = n05_with_lu.apply(
        lambda r: get_lu_for_hu(r["EQUIP_IDENT"], r["START_TIMESTAMP"]), axis=1
    )
    n05_with_lu["LU_TYPE"] = n05_with_lu["LU"].map(lambda x: lu_type_map.get(x, "EX") if x else "EX")

    for lu_type, mult in [("EX", ex_mult), ("WH", wh_mult)]:
        subset = n05_with_lu[n05_with_lu["LU_TYPE"] == lu_type]["SPOT_DUR"]
        if len(subset) > 1:
            thresh = subset.mean() + mult * subset.std()
        elif len(subset) == 1:
            thresh = subset.iloc[0] * 1.5
        else:
            thresh = float("inf")
        n05_with_lu.loc[n05_with_lu["LU_TYPE"] == lu_type, "THRESHOLD"] = thresh

    outliers = n05_with_lu[n05_with_lu["SPOT_DUR"] > n05_with_lu["THRESHOLD"]].copy()

    case_counter = 0
    for _, row in outliers.iterrows():
        case_counter += 1
        hu = row["EQUIP_IDENT"]
        lu = row.get("LU") or ""
        n05_start = row["START_TIMESTAMP"]
        n05_end = row["END_TIMESTAMP"]
        n05_dur = row["SPOT_DUR"]
        threshold = row["THRESHOLD"]
        feats = _compute_features(
            hu, lu, n05_start, n05_end, n05_dur,
            eq_status, haul_cycle, eq_coord, location_desc, equip_health,
            anomaly_type="OUTLIER",
        )
        c = Case(
            case_id=f"CASE_{case_counter:04d}",
            hu=str(hu),
            lu=str(lu),
            anomaly_type="OUTLIER",
            n05_start=n05_start,
            n05_end=n05_end,
            n05_dur=n05_dur,
            threshold=threshold,
            delta=n05_dur - threshold,
            features=feats,
        )
        cases.append(c)

    return cases


def _compute_features(
    hu, lu, n05_start, n05_end, n05_dur,
    eq_status, haul_cycle, eq_coord, location_desc, equip_health,
    anomaly_type="OUTLIER",
) -> dict:
    feats = {}
    feats["N05_DUR"] = n05_dur
    feats["N05_MISSING"] = anomaly_type == "MISSING"

    hu_coords = eq_coord[
        (eq_coord["EQUIP_IDENT"] == hu)
        & (eq_coord["TIMESTAMP"] >= n05_start)
        & (eq_coord["TIMESTAMP"] <= n05_end)
    ].sort_values("TIMESTAMP")

    lu_coords = pd.DataFrame()
    if lu:
        lu_coords = eq_coord[
            (eq_coord["EQUIP_IDENT"] == lu)
            & (eq_coord["TIMESTAMP"] >= n05_start)
            & (eq_coord["TIMESTAMP"] <= n05_end)
        ].sort_values("TIMESTAMP")

    feats["MAX_LAT_HU"] = float(hu_coords["PREVIOUS_LATENCY"].max()) if not hu_coords.empty else None
    entry_window = hu_coords[
        (hu_coords["TIMESTAMP"] >= n05_start - pd.Timedelta("30s"))
        & (hu_coords["TIMESTAMP"] <= n05_start + pd.Timedelta("30s"))
    ]
    feats["MAX_LAT_HU_AT_ENTRY"] = float(entry_window["PREVIOUS_LATENCY"].max()) if not entry_window.empty else None

    if lu_coords.empty or lu_coords["PREVIOUS_LATENCY"].isna().all():
        feats["LU_ALL_NAN"] = True
        feats["LU_OFFLINE_PCT"] = 100.0
        feats["LU_MEDIAN_LAT"] = None
    else:
        feats["LU_ALL_NAN"] = False
        total = len(lu_coords)
        offline = (lu_coords["PREVIOUS_LATENCY"] > 5000).sum()
        feats["LU_OFFLINE_PCT"] = 100.0 * offline / total if total > 0 else 0.0
        feats["LU_MEDIAN_LAT"] = float(lu_coords["PREVIOUS_LATENCY"].median())

    n06_rows = eq_status[
        (eq_status["EQUIP_IDENT"] == hu)
        & (eq_status["STATUS_CODE"] == "N06")
        & (eq_status["END_TIMESTAMP"] <= n05_start)
        & (eq_status["END_TIMESTAMP"] >= n05_start - pd.Timedelta("600s"))
    ]
    feats["HAS_N06"] = not n06_rows.empty
    if not n06_rows.empty:
        n06 = n06_rows.sort_values("END_TIMESTAMP").iloc[-1]
        feats["N06_DUR"] = (n06["END_TIMESTAMP"] - n06["START_TIMESTAMP"]).total_seconds()
        n06_hu_coords = eq_coord[
            (eq_coord["EQUIP_IDENT"] == hu)
            & (eq_coord["TIMESTAMP"] >= n06["START_TIMESTAMP"])
            & (eq_coord["TIMESTAMP"] <= n06["END_TIMESTAMP"])
        ]
        feats["HU_MOVES_IN_N06"] = bool((n06_hu_coords["SPEED"] > 3).any()) if not n06_hu_coords.empty else False
    else:
        feats["N06_DUR"] = None
        feats["HU_MOVES_IN_N06"] = False

    if lu:
        n11_rows = eq_status[
            (eq_status["EQUIP_IDENT"] == lu)
            & (eq_status["STATUS_CODE"] == "N11")
        ]
        feats["N11_LU_HOUR"] = len(n11_rows)
        feats["N11_OVERLAP_PCT"] = float(
            n11_rows.apply(
                lambda r: _ts_overlap_pct(n05_start, n05_end, r["START_TIMESTAMP"], r["END_TIMESTAMP"]),
                axis=1,
            ).max()
        ) if not n11_rows.empty else 0.0
    else:
        feats["N11_LU_HOUR"] = 0
        feats["N11_OVERLAP_PCT"] = 0.0

    has_vims = False
    for sc in ["N03", "N01"]:
        rows = eq_status[
            (eq_status["EQUIP_IDENT"] == hu)
            & (eq_status["STATUS_CODE"] == sc)
            & (eq_status["START_TIMESTAMP"] >= n05_start)
            & (eq_status["START_TIMESTAMP"] <= n05_end + pd.Timedelta("120s"))
        ]
        if not rows.empty and (rows["SOURCE"].str.contains("VIMS", na=False)).any():
            has_vims = True
    feats["HAS_VIMS"] = has_vims

    if not hu_coords.empty and len(hu_coords) >= 3:
        speeds = hu_coords["SPEED"].fillna(0).values
        moving = speeds > 3
        last_move_idx = None
        for i in range(len(moving) - 1, -1, -1):
            if moving[i]:
                last_move_idx = i
                break
        if last_move_idx is not None:
            feats["N05_REAL"] = (n05_end - hu_coords.iloc[last_move_idx]["TIMESTAMP"]).total_seconds()
        else:
            feats["N05_REAL"] = 0.0
        n2_dur = 0.0
        max_pause_dur = 0.0
        max_pause_disp = 0.0
        for i in range(1, len(hu_coords)):
            r0 = hu_coords.iloc[i - 1]
            r1 = hu_coords.iloc[i]
            if pd.isna(r0["NORTHING"]) or pd.isna(r1["NORTHING"]):
                continue
            disp = _haversine_m(r0["NORTHING"], r0["EASTING"], r1["NORTHING"], r1["EASTING"])
            seg_dur = (r1["TIMESTAMP"] - r0["TIMESTAMP"]).total_seconds()
            if disp < 6:
                n2_dur += seg_dur
                if seg_dur > max_pause_dur:
                    max_pause_dur = seg_dur
                    max_pause_disp = disp
        feats["N02_DUR"] = n2_dur
        feats["N02_DISPLACEMENT"] = max_pause_disp
    else:
        feats["N05_REAL"] = n05_dur
        feats["N02_DUR"] = 0.0
        feats["N02_DISPLACEMENT"] = 0.0

    lu_gps_ok = (
        not feats["LU_ALL_NAN"]
        and feats.get("LU_MEDIAN_LAT") is not None
        and feats["LU_MEDIAN_LAT"] < 1000
    )

    if lu_gps_ok and not lu_coords.empty and not hu_coords.empty:
        lu_pos = lu_coords.iloc[0]
        hu_start_pos = hu_coords.iloc[0]
        feats["DIST_HU_LU_START"] = _haversine_m(
            hu_start_pos["NORTHING"], hu_start_pos["EASTING"],
            lu_pos["NORTHING"], lu_pos["EASTING"],
        ) if not any(pd.isna(v) for v in [hu_start_pos["NORTHING"], hu_start_pos["EASTING"], lu_pos["NORTHING"], lu_pos["EASTING"]]) else None
        dists = []
        for _, hrow in hu_coords.iterrows():
            closest_lu = lu_coords.iloc[(lu_coords["TIMESTAMP"] - hrow["TIMESTAMP"]).abs().argsort()[:1]]
            if not closest_lu.empty:
                lr = closest_lu.iloc[0]
                if not any(pd.isna(v) for v in [hrow["NORTHING"], hrow["EASTING"], lr["NORTHING"], lr["EASTING"]]):
                    dists.append(_haversine_m(hrow["NORTHING"], hrow["EASTING"], lr["NORTHING"], lr["EASTING"]))
        if dists:
            feats["DIST_HU_LU_MIN"] = min(dists)
            feats["HU_APPROACHED_LU"] = dists[0] > dists[-1] if len(dists) > 1 else False
        else:
            feats["DIST_HU_LU_MIN"] = None
            feats["HU_APPROACHED_LU"] = False
    else:
        feats["DIST_HU_LU_START"] = None
        feats["DIST_HU_LU_MIN"] = None
        feats["HU_APPROACHED_LU"] = False

    if not hu_coords.empty:
        end_coords = hu_coords[hu_coords["TIMESTAMP"] >= n05_end - pd.Timedelta("30s")]
        feats["HU_SPEED_AT_N05_END"] = float(end_coords["SPEED"].mean()) if not end_coords.empty else 0.0
    else:
        feats["HU_SPEED_AT_N05_END"] = 0.0

    n01_rows = eq_status[
        (eq_status["EQUIP_IDENT"] == hu)
        & (eq_status["STATUS_CODE"] == "N01")
        & (eq_status["START_TIMESTAMP"] >= n05_end - pd.Timedelta("10s"))
        & (eq_status["START_TIMESTAMP"] <= n05_end + pd.Timedelta("120s"))
    ].sort_values("START_TIMESTAMP")

    if not n01_rows.empty:
        n01 = n01_rows.iloc[0]
        feats["N01_DUR"] = (n01["END_TIMESTAMP"] - n01["START_TIMESTAMP"]).total_seconds()
        feats["N01_SOURCE"] = str(n01.get("SOURCE", ""))
        n01_start_coords = hu_coords[hu_coords["TIMESTAMP"] >= n01["START_TIMESTAMP"] - pd.Timedelta("30s")]
        feats["HU_SPEED_AT_N01_START"] = float(n01_start_coords.iloc[0]["SPEED"]) if not n01_start_coords.empty else 0.0
        n04_after = eq_status[
            (eq_status["EQUIP_IDENT"] == hu)
            & (eq_status["STATUS_CODE"] == "N04")
            & (eq_status["START_TIMESTAMP"] >= n01["END_TIMESTAMP"])
            & (eq_status["START_TIMESTAMP"] <= n01["END_TIMESTAMP"] + pd.Timedelta("120s"))
        ]
        feats["SEQ_N01_N04"] = not n04_after.empty
        feats["SEQ_N05_N01_GAP"] = (n01["START_TIMESTAMP"] - n05_end).total_seconds() <= 10
    else:
        feats["N01_DUR"] = None
        feats["N01_SOURCE"] = None
        feats["HU_SPEED_AT_N01_START"] = None
        feats["SEQ_N01_N04"] = False
        feats["SEQ_N05_N01_GAP"] = False

    feats["LU_SPEED_DURING_STOP"] = float(lu_coords["SPEED"].max()) if not lu_coords.empty and "SPEED" in lu_coords.columns else None

    hc_rows = haul_cycle[
        (haul_cycle["HAULING_UNIT_IDENT"] == hu)
        & (haul_cycle["LOAD_START_TIMESTAMP"] >= n05_start - pd.Timedelta("300s"))
        & (haul_cycle["LOAD_START_TIMESTAMP"] <= n05_end + pd.Timedelta("300s"))
    ].sort_values("LOAD_START_TIMESTAMP")

    feats["QTY_SOURCE"] = None
    feats["LOAD_CONFIRMED"] = False
    feats["DIST_HU_LU_AT_LOAD"] = None
    feats["DUMP_END_TS"] = None
    feats["LOAD_START_TS"] = None

    if not hc_rows.empty:
        hc = hc_rows.iloc[0]
        feats["QTY_SOURCE"] = str(hc.get("QUANTITY_SOURCE", ""))
        qty_src = feats["QTY_SOURCE"]
        feats["LOAD_CONFIRMED"] = qty_src in ["H", "MDT", "VIMS", "DIO"] or qty_src.startswith("WCS::")
        feats["DUMP_END_TS"] = hc.get("DUMP_END_TIMESTAMP")
        feats["LOAD_START_TS"] = hc.get("LOAD_START_TIMESTAMP")
        if lu_gps_ok and not lu_coords.empty and pd.notna(hc.get("LOAD_START_TIMESTAMP")):
            load_ts = hc["LOAD_START_TIMESTAMP"]
            hu_at_load = hu_coords.iloc[(hu_coords["TIMESTAMP"] - load_ts).abs().argsort()[:1]]
            lu_at_load = lu_coords.iloc[(lu_coords["TIMESTAMP"] - load_ts).abs().argsort()[:1]]
            if not hu_at_load.empty and not lu_at_load.empty:
                h = hu_at_load.iloc[0]
                l = lu_at_load.iloc[0]
                if not any(pd.isna(v) for v in [h["NORTHING"], h["EASTING"], l["NORTHING"], l["EASTING"]]):
                    feats["DIST_HU_LU_AT_LOAD"] = _haversine_m(h["NORTHING"], h["EASTING"], l["NORTHING"], l["EASTING"])

    if lu:
        n13_rows = eq_status[
            (eq_status["EQUIP_IDENT"] == lu)
            & (eq_status["STATUS_CODE"] == "N13")
        ]
        if not n13_rows.empty:
            n13_durs = (n13_rows["END_TIMESTAMP"] - n13_rows["START_TIMESTAMP"]).dt.total_seconds()
            feats["TYPICAL_N13"] = float(n13_durs.median())
            n13_at_cycle = n13_rows[
                (n13_rows["START_TIMESTAMP"] >= n05_start - pd.Timedelta("300s"))
                & (n13_rows["END_TIMESTAMP"] <= n05_end + pd.Timedelta("300s"))
            ]
            feats["N13_DUR"] = float(
                (n13_at_cycle["END_TIMESTAMP"] - n13_at_cycle["START_TIMESTAMP"]).dt.total_seconds().sum()
            ) if not n13_at_cycle.empty else None
        else:
            feats["TYPICAL_N13"] = None
            feats["N13_DUR"] = None
    else:
        feats["TYPICAL_N13"] = None
        feats["N13_DUR"] = None

    if feats.get("N01_DUR") and feats.get("TYPICAL_N13"):
        feats["N01_DUR_RATIO"] = feats["N01_DUR"] / feats["TYPICAL_N13"]
        feats["N01_N13_DELTA"] = abs(feats["N01_DUR"] - (feats.get("N13_DUR") or feats["TYPICAL_N13"]))
    else:
        feats["N01_DUR_RATIO"] = None
        feats["N01_N13_DELTA"] = None

    feats["MATCH_N05_N11"] = feats["N11_OVERLAP_PCT"] > 50
    feats["MATCH_N01_N13"] = feats.get("N01_N13_DELTA") is not None and feats["N01_N13_DELTA"] < 10
    n01_ratio = feats.get("N01_DUR_RATIO")
    feats["LOAD_DUR_OK"] = bool(n01_ratio and 0.5 <= n01_ratio <= 1.5)
    feats["MOTION_STOP_NEAR"] = feats.get("DIST_HU_LU_MIN") is not None and feats["DIST_HU_LU_MIN"] < 20
    feats["MOTION_N02_STILL"] = feats["N02_DISPLACEMENT"] < 6
    feats["MOTION_LU_GPS_OK"] = lu_gps_ok
    feats["_hu_track"] = hu_coords[["TIMESTAMP", "NORTHING", "EASTING", "SPEED"]].to_dict("records") if not hu_coords.empty else []
    feats["_lu_track"] = lu_coords[["TIMESTAMP", "NORTHING", "EASTING", "SPEED"]].to_dict("records") if not lu_coords.empty else []
    return feats


def validate_dataframes(dfs: dict) -> list:
    errors = []
    required = {
        "equipment_status_trans": ["EQUIP_IDENT", "STATUS_CODE", "START_TIMESTAMP", "END_TIMESTAMP"],
        "haul_cycle_trans": ["HAUL_CYCLE_REC_IDENT", "HAULING_UNIT_IDENT", "LOADING_UNIT_IDENT",
                             "START_TIMESTAMP", "LOAD_START_TIMESTAMP", "DUMP_END_TIMESTAMP"],
        "equip_coord_trans": ["EQUIP_IDENT", "TIMESTAMP", "NORTHING", "EASTING"],
        "location_desc_current": ["LOCATION_SNAME", "NORTHING", "EASTING"],
        "EQUIP_HEALTH_INFO_TRANS": ["EQUIP_IDENT", "TIMESTAMP"],
    }
    for name, cols in required.items():
        if name not in dfs:
            errors.append(f"Файл '{name}' не загружен")
            continue
        df = dfs[name]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            errors.append(f"'{name}': отсутствуют колонки {missing}")
    return errors
