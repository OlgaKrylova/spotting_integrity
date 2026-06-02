"""
Spot Integrity Analyzer — MVP
Streamlit application for N05 spotting anomaly analysis and labeling.
"""

import io
import math
import yaml
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from calculator import compute_cases, validate_dataframes, Case
from classifier import classify, CLASS_LABELS, CLASSES, retrain, load_config
from exporter import export_xlsx

st.set_page_config(
    page_title="Spot Integrity Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_state():
    defaults = {
        "page": "upload",
        "dfs": {},
        "cases": [],
        "active_idx": 0,
        "filter_lu_type": "Все",
        "filter_class": "Все",
        "filter_status": "Все",
        "search_query": "",
        "sort_col": "N05_DUR",
        "sort_asc": False,
        "calculated": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()
ss = st.session_state

FILE_KEYS = [
    "equipment_status_trans",
    "haul_cycle_trans",
    "equip_coord_trans",
    "location_desc_current",
    "EQUIP_HEALTH_INFO_TRANS",
]

CLASS_LABEL_INV = {v: k for k, v in CLASS_LABELS.items()}


def _read_file(uploaded, sep=";") -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    data = uploaded.read()
    for enc in ["utf-8", "cp1251", "latin-1"]:
        try:
            return pd.read_csv(io.BytesIO(data), sep=sep, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(data), sep=sep, encoding="utf-8", errors="replace", low_memory=False)


def _run_classification():
    for c in ss.cases:
        probs = classify(c.features)
        c.auto_probs = probs
        c.auto_class = max(probs, key=probs.get)
        if not c.labeled:
            c.label_source = "auto"


def _labeled_count():
    return sum(1 for c in ss.cases if c.labeled)


def _status_badge(c) -> str:
    if not c.labeled:
        return "⬜ Не размечен"
    if c.label_source == "confirmed":
        return "✅ Подтверждён"
    return "🟡 Переопределён"


def _fmt_dur(sec) -> str:
    if sec is None or (isinstance(sec, float) and math.isnan(sec)):
        return "—"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}ч {m}м {s}с"
    if m:
        return f"{m}м {s}с"
    return f"{s}с"


# ── Sidebar ──
with st.sidebar:
    st.title("🔍 Spot Integrity")
    st.markdown("---")
    if st.button("📁 Загрузка данных", use_container_width=True):
        ss.page = "upload"
    if st.button("📋 Список аномалий", use_container_width=True, disabled=not ss.calculated):
        ss.page = "list"
    if ss.cases:
        st.markdown("---")
        labeled = _labeled_count()
        st.metric("Аномалий", len(ss.cases))
        st.metric("Размечено", labeled)
        st.progress(labeled / len(ss.cases))
        if labeled >= 30:
            if st.button("🔄 Переобучить модель", use_container_width=True):
                labeled_cases = [
                    {"features": c.features, "manual_class": c.manual_class or c.auto_class, "auto_class": c.auto_class}
                    for c in ss.cases if c.labeled
                ]
                new_probs = retrain(labeled_cases)
                cfg = load_config()
                cfg["feature_probs"] = new_probs
                with open(Path(__file__).parent / "config.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, allow_unicode=True)
                old_classes = {c.case_id: c.auto_class for c in ss.cases}
                _run_classification()
                changed = sum(1 for c in ss.cases if old_classes[c.case_id] != c.auto_class)
                st.success(f"Переобучено. Изменилось {changed} классов.")
        st.markdown("---")
        xlsx_bytes = export_xlsx(ss.cases)
        st.download_button(
            "⬇️ Экспорт XLSX",
            data=xlsx_bytes,
            file_name="spot_integrity_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


# ── Screen 1: Upload ──
if ss.page == "upload":
    st.header("📁 Загрузка данных")
    st.info("Загрузите 5 CSV-файлов (разделитель `;`) или один XLSX со всеми таблицами листами.")

    col_a, col_b = st.columns(2)
    uploads = {}

    with col_a:
        st.subheader("Отдельные CSV-файлы")
        for key in FILE_KEYS:
            f = st.file_uploader(f"`{key}`", key=f"up_{key}", type=["csv", "xlsx", "xls"])
            if f:
                uploads[key] = f

    with col_b:
        st.subheader("Или один XLSX")
        xlsx_file = st.file_uploader("XLSX (все таблицы листами)", key="up_xlsx", type=["xlsx", "xls"])
        if xlsx_file:
            xls = pd.ExcelFile(xlsx_file)
            for sheet in xls.sheet_names:
                for key in FILE_KEYS:
                    if key.lower() in sheet.lower() or sheet.lower() in key.lower():
                        uploads[key] = xls.parse(sheet)
                        break

    if uploads:
        dfs = {}
        for key, val in uploads.items():
            if isinstance(val, pd.DataFrame):
                dfs[key] = val
            else:
                try:
                    dfs[key] = _read_file(val)
                except Exception as e:
                    st.error(f"Ошибка чтения `{key}`: {e}")

        st.subheader("Валидация")
        errors = validate_dataframes(dfs)
        all_ok = True
        for key in FILE_KEYS:
            file_errors = [e for e in errors if key in e]
            if key in dfs:
                if file_errors:
                    st.error(f"✗ `{key}` — {'; '.join(file_errors)}")
                    all_ok = False
                else:
                    st.success(f"✓ `{key}` — {len(dfs[key]):,} строк")
            else:
                st.warning(f"— `{key}` не загружен")
                all_ok = False

        if "equipment_status_trans" in dfs:
            eq = dfs["equipment_status_trans"]
            st.subheader("Сводка данных")
            c1, c2, c3, c4 = st.columns(4)
            ts_min = eq["START_TIMESTAMP"].min() if "START_TIMESTAMP" in eq.columns else None
            ts_max = eq["END_TIMESTAMP"].max() if "END_TIMESTAMP" in eq.columns else None
            hus = dfs.get("haul_cycle_trans", pd.DataFrame())
            c1.metric("Временной диапазон", f"{ts_min} → {ts_max}" if ts_min else "—")
            c2.metric("Самосвалов (HU)", hus["HAULING_UNIT_IDENT"].nunique() if "HAULING_UNIT_IDENT" in hus.columns else 0)
            c3.metric("Экскаваторов (LU)", hus["LOADING_UNIT_IDENT"].nunique() if "LOADING_UNIT_IDENT" in hus.columns else 0)
            c4.metric("Завершённых циклов", len(hus))

        if all_ok:
            st.markdown("---")
            if st.button("🚀 Рассчитать аномалии", type="primary", use_container_width=True):
                with st.spinner("Расчёт признаков и классификация..."):
                    try:
                        cfg = load_config()
                        cases = compute_cases(
                            dfs["equipment_status_trans"],
                            dfs["haul_cycle_trans"],
                            dfs["equip_coord_trans"],
                            dfs["location_desc_current"],
                            dfs["EQUIP_HEALTH_INFO_TRANS"],
                            cfg,
                        )
                        ss.dfs = dfs
                        ss.cases = cases
                        _run_classification()
                        ss.calculated = True
                        ss.page = "list"
                        st.success(f"Найдено {len(cases)} аномалий")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка расчёта: {e}")
                        st.exception(e)


# ── Screen 2: List ──
elif ss.page == "list":
    st.header("📋 Список аномалий")

    if not ss.cases:
        st.warning("Нет данных. Сначала загрузите файлы.")
        ss.page = "upload"
        st.rerun()

    with st.expander("🔽 Фильтры и поиск", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        lu_types = ["Все", "EX", "WH"]
        all_classes = ["Все"] + [CLASS_LABELS[c] for c in CLASSES]
        all_statuses = ["Все", "Не размечен", "Подтверждён", "Переопределён"]
        ss.filter_lu_type = fc1.selectbox("Тип LU", lu_types, index=lu_types.index(ss.filter_lu_type))
        ss.filter_class = fc2.selectbox("Класс", all_classes, index=all_classes.index(ss.filter_class) if ss.filter_class in all_classes else 0)
        ss.filter_status = fc3.selectbox("Статус", all_statuses, index=all_statuses.index(ss.filter_status))
        ss.search_query = fc4.text_input("Поиск по HU/LU", ss.search_query)

    visible = list(ss.cases)
    if ss.filter_lu_type != "Все":
        visible = [c for c in visible if ss.filter_lu_type.upper() in c.lu.upper()]
    if ss.filter_class != "Все":
        cls_key = CLASS_LABEL_INV.get(ss.filter_class, "")
        visible = [c for c in visible if c.auto_class == cls_key or c.manual_class == cls_key]
    if ss.filter_status == "Не размечен":
        visible = [c for c in visible if not c.labeled]
    elif ss.filter_status == "Подтверждён":
        visible = [c for c in visible if c.labeled and c.label_source == "confirmed"]
    elif ss.filter_status == "Переопределён":
        visible = [c for c in visible if c.labeled and c.label_source == "manual"]
    if ss.search_query:
        q = ss.search_query.upper()
        visible = [c for c in visible if q in c.hu.upper() or q in c.lu.upper()]

    st.caption(f"Показано {len(visible)} из {len(ss.cases)} аномалий")

    if visible:
        hcols = st.columns([1, 1, 1, 1, 1, 1, 2, 1, 2, 2])
        for h, col in zip(["HU", "LU", "Тип", "N05_DUR", "Порог", "+Δ", "Авто-класс", "P(%)", "Метка", "Статус"], hcols):
            col.markdown(f"**{h}**")

        for c in visible:
            label_display = CLASS_LABELS.get(c.manual_class or c.auto_class, "—")
            auto_p = round(c.auto_probs.get(c.auto_class, 0) * 100, 1)
            bg = "🟢" if (c.labeled and c.label_source == "confirmed") else ("🟡" if c.labeled else "")
            cols = st.columns([1, 1, 1, 1, 1, 1, 2, 1, 2, 2])
            cols[0].write(c.hu)
            cols[1].write(c.lu or "—")
            cols[2].write(c.anomaly_type)
            cols[3].write(_fmt_dur(c.n05_dur))
            cols[4].write(_fmt_dur(c.threshold))
            cols[5].write(_fmt_dur(c.delta))
            cols[6].write(CLASS_LABELS.get(c.auto_class, "—"))
            cols[7].write(f"{auto_p}%")
            cols[8].write(f"{bg} {label_display}")
            cols[9].write(_status_badge(c))
            orig_idx = ss.cases.index(c)
            if cols[0].button("→", key=f"open_{orig_idx}"):
                ss.active_idx = orig_idx
                ss.page = "card"
                st.rerun()


# ── Screen 3: Card ──
elif ss.page == "card":
    if not ss.cases:
        ss.page = "upload"
        st.rerun()

    idx = ss.active_idx
    c = ss.cases[idx]

    nav = st.columns([1, 6, 1])
    with nav[0]:
        if st.button("◀ Пред.", disabled=idx == 0):
            ss.active_idx = idx - 1
            st.rerun()
    with nav[1]:
        st.markdown(f"### Кейс {idx + 1} / {len(ss.cases)}  —  {c.case_id}")
    with nav[2]:
        if st.button("След. ▶", disabled=idx == len(ss.cases) - 1):
            ss.active_idx = idx + 1
            st.rerun()
    if st.button("← К списку"):
        ss.page = "list"
        st.rerun()

    st.markdown("---")

    # A — Identification
    st.subheader("A — Идентификация")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("HU", c.hu)
    a2.metric("LU", c.lu or "—")
    a3.metric("Тип", c.anomaly_type)
    a4.metric("N05_DUR", _fmt_dur(c.n05_dur))
    a5.metric("Превышение", f"+{_fmt_dur(c.delta)}")
    st.caption(f"Период: {c.n05_start} → {c.n05_end}")

    # B — Indicators
    st.subheader("B — Индикаторы")
    b1, b2, b3, b4 = st.columns(4)

    def _ind(val, label):
        return f"{'✓' if val else '✗'} {label}"

    b1.info(_ind(c.features.get("HAS_N06"), "Зона ожидания N06"))
    b2.info(_ind(not c.features.get("LU_ALL_NAN") and c.features.get("LU_OFFLINE_PCT", 100) < 50, "Связь LU"))
    b3.info(_ind(c.features.get("MOTION_LU_GPS_OK"), "GPS LU"))
    b4.info(_ind(c.features.get("N02_DUR", 0) > 30, "Простой в N05"))

    # C — Checks
    st.subheader("C — Базовые проверки")
    tabs = st.tabs(["Последовательность", "Мэтч HU↔LU", "Загрузка", "Движение"])

    def _chk(val, name, val_str=""):
        return f"{'✓' if val else '✗'} **{name}**" + (f" — {val_str}" if val_str else "")

    with tabs[0]:
        st.markdown(_chk(c.features.get("SEQ_N05_N01_GAP"), "N05→N01 ≤ 10с"))
        st.markdown(_chk(c.features.get("SEQ_N01_N04"), "N01→N04 следует"))
    with tabs[1]:
        st.markdown(_chk(c.features.get("MATCH_N05_N11"), "N05↔N11 > 50%", f"{c.features.get('N11_OVERLAP_PCT', 0):.1f}%"))
        st.markdown(_chk(c.features.get("MATCH_N01_N13"), "N01≈N13 < 10с", f"Δ={c.features.get('N01_N13_DELTA', '—')}с"))
    with tabs[2]:
        st.markdown(_chk(c.features.get("LOAD_DUR_OK"), "0.5 ≤ N01/N13 ≤ 1.5", f"ratio={c.features.get('N01_DUR_RATIO', '—')}"))
        st.markdown(_chk(c.features.get("LOAD_CONFIRMED"), "LOAD_CONFIRMED", str(c.features.get("QTY_SOURCE", "—"))))
    with tabs[3]:
        st.markdown(_chk(c.features.get("HU_APPROACHED_LU"), "HU приближался к LU"))
        st.markdown(_chk(c.features.get("MOTION_STOP_NEAR"), "DIST_HU_LU_MIN < 20м", f"{c.features.get('DIST_HU_LU_MIN', '—')}м"))
        st.markdown(_chk(c.features.get("MOTION_N02_STILL"), "N02_DISPLACEMENT < 6м", f"{c.features.get('N02_DISPLACEMENT', '—')}м"))
        st.markdown(_chk(c.features.get("MOTION_LU_GPS_OK"), "LU GPS надёжен"))

    # D — Feature table
    st.subheader("D — Признаки классификатора")
    from classifier import binarize_features
    binary = binarize_features(c.features)
    feat_display = []
    for fk, fv in c.features.items():
        if fk.startswith("_"):
            continue
        bin_key = fk + ("_5000" if "LAT_HU_AT_ENTRY" in fk else "_15" if "OFFLINE_PCT" in fk else "_30" if fk == "N02_DUR" else "")
        triggered = binary.get(bin_key, binary.get(fk, None))
        feat_display.append({"Признак": fk, "Значение": fv, "Бинаризован": "🔴" if triggered is True else ("⚪" if triggered is False else "")})
    st.dataframe(pd.DataFrame(feat_display), use_container_width=True, hide_index=True)

    # E — GPS track
    st.subheader("E — GPS-трек")
    hu_track = c.features.get("_hu_track", [])
    lu_track = c.features.get("_lu_track", [])
    if len(hu_track) >= 3:
        import altair as alt
        hu_df = pd.DataFrame(hu_track)
        chart_data = [{"EASTING": r["EASTING"], "NORTHING": r["NORTHING"],
                       "phase": "подъезд" if (r.get("SPEED") or 0) > 3 else "пауза (N02)", "entity": "HU"}
                      for r in hu_track]
        if lu_track and not c.features.get("LU_ALL_NAN"):
            lu_df = pd.DataFrame(lu_track)
            chart_data.append({"EASTING": lu_df["EASTING"].mean(), "NORTHING": lu_df["NORTHING"].mean(),
                               "phase": "LU позиция", "entity": "LU"})
        plot_df = pd.DataFrame(chart_data)
        color_scale = alt.Scale(domain=["подъезд", "пауза (N02)", "LU позиция"], range=["#2196F3", "#FF9800", "#E91E63"])
        line = alt.Chart(plot_df[plot_df["entity"] == "HU"]).mark_line(opacity=0.5).encode(x="EASTING:Q", y="NORTHING:Q")
        pts = alt.Chart(plot_df).mark_point(size=60).encode(
            x=alt.X("EASTING:Q", title="Easting (м)"), y=alt.Y("NORTHING:Q", title="Northing (м)"),
            color=alt.Color("phase:N", scale=color_scale, title="Фаза"),
            shape=alt.Shape("entity:N"), tooltip=["entity", "phase", "EASTING", "NORTHING"])
        st.altair_chart((line + pts).interactive(), use_container_width=True)
    else:
        st.info("Недостаточно GPS-точек HU (нужно ≥ 3).")

    # F — Gantt
    st.subheader("F — Временная диаграмма (Гант)")
    import altair as alt
    n05_dur = c.n05_dur
    n05_real = c.features.get("N05_REAL", n05_dur) or n05_dur
    n02_dur = c.features.get("N02_DUR", 0) or 0
    n01_dur = c.features.get("N01_DUR") or 0
    gantt_rows = [{"Ряд": "Wenco", "Статус": "N05", "Начало": 0, "Конец": n05_dur}]
    if n01_dur:
        gantt_rows.append({"Ряд": "Wenco", "Статус": "N01", "Начало": n05_dur, "Конец": n05_dur + n01_dur})
    gantt_rows.append({"Ряд": "Исправлено", "Статус": "N05_REAL", "Начало": n05_dur - n05_real, "Конец": n05_dur})
    if n02_dur:
        gantt_rows.append({"Ряд": "Исправлено", "Статус": "N02 (пауза)", "Начало": n05_dur - n05_real - n02_dur, "Конец": n05_dur - n05_real})
    gantt_df = pd.DataFrame(gantt_rows)
    st.altair_chart(alt.Chart(gantt_df).mark_bar(height=20).encode(
        x=alt.X("Начало:Q", title="Секунды от начала N05"), x2="Конец:Q",
        y=alt.Y("Ряд:N", title=""), color=alt.Color("Статус:N"),
        tooltip=["Статус", "Начало", "Конец"]).properties(height=100), use_container_width=True)
    st.caption(f"N05_DUR={_fmt_dur(n05_dur)}, N05_REAL={_fmt_dur(n05_real)}, N02_DUR={_fmt_dur(n02_dur)}")

    # G — Classification
    st.subheader("G — Классификация")
    st.markdown("**Авто-предложение модели:**")
    for cls_key, prob in sorted(c.auto_probs.items(), key=lambda x: -x[1])[:2]:
        st.markdown(f"**{CLASS_LABELS.get(cls_key, cls_key)}** — {prob*100:.1f}%")
        st.progress(prob)

    st.markdown("**Выберите класс:**")
    class_selection = st.radio(
        "Класс", options=list(CLASS_LABELS.keys()), format_func=lambda k: CLASS_LABELS[k],
        index=list(CLASS_LABELS.keys()).index(c.manual_class or c.auto_class) if (c.manual_class or c.auto_class) in CLASS_LABELS else 0,
        horizontal=True, key=f"cls_radio_{idx}",
    )
    conf_col, _ = st.columns([2, 4])
    confidence = conf_col.radio("Уверенность", ["УВЕРЕН", "ПРЕДПОЛОЖИТЕЛЬНО"],
                                index=0 if c.manual_conf in ["sure", ""] else 1,
                                horizontal=True, key=f"conf_{idx}")
    note = st.text_area("Заметка (опционально)", value=c.note or "", max_chars=500, key=f"note_{idx}")

    ac = st.columns([2, 2, 4])
    if ac[0].button("✅ Подтвердить", type="primary", key=f"confirm_{idx}"):
        c.manual_class = class_selection
        c.manual_conf = "sure" if confidence == "УВЕРЕН" else "guess"
        c.note = note
        c.labeled = True
        c.label_source = "manual" if class_selection != c.auto_class else "confirmed"
        ss.cases[idx] = c
        if idx + 1 < len(ss.cases):
            ss.active_idx = idx + 1
        st.rerun()
    if ac[1].button("⏭ Пропустить", key=f"skip_{idx}"):
        if idx + 1 < len(ss.cases):
            ss.active_idx = idx + 1
            st.rerun()

    st.caption("Горячие клавиши: 1–6 — выбор класса, Enter — подтвердить, Esc — пропустить, ←/→ — навигация")
