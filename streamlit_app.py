"""Streamlit GUI for Crisis Radar ML Pipeline."""

import pickle
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from crisis_radar.config import CrisisRadarConfig
from crisis_radar.features import build_features, get_feature_names
from crisis_radar.live import get_live_risk_score
from crisis_radar.pipeline import run_pipeline
from crisis_radar.utils import load_json

from streamlit_i18n import sample_csv_bytes, t

GSPC_ALIASES = frozenset(
    {"gspc_close", "^gspc", "gspc", "spx", "s&p_500", "sandp500", "^spx", "spy"}
)
VIX_ALIASES = frozenset({"vix_close", "^vix", "vix"})


def _col_key(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    min_ok = max(3, len(df) // 4)
    for c in df.columns:
        if _col_key(c) in ("date", "datetime", "time"):
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().sum() > min_ok:
                sub = df.drop(columns=[c]).copy()
                sub.index = dt
                return sub.loc[sub.index.notna()]
    if len(df.columns) >= 2:
        dt = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        if dt.notna().sum() > min_ok:
            body = df.iloc[:, 1:].copy()
            body.index = dt
            return body.loc[body.index.notna()]
    idx = pd.to_datetime(df.index, errors="coerce")
    if idx.notna().sum() > min_ok:
        out = df.copy()
        out.index = idx
        return out.loc[out.index.notna()]
    return df


def _normalize_uploaded_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_datetime_index(df)
    gspc_col = None
    vix_col = None
    for c in df.columns:
        k = _col_key(c)
        if k in GSPC_ALIASES and gspc_col is None:
            gspc_col = c
        elif k in VIX_ALIASES and vix_col is None:
            vix_col = c
    if gspc_col is None or vix_col is None:
        return df
    out = pd.DataFrame(
        {
            "GSPC_Close": pd.to_numeric(df[gspc_col], errors="coerce"),
            "VIX_Close": pd.to_numeric(df[vix_col], errors="coerce"),
        },
        index=df.index,
    )
    return out.sort_index().dropna(subset=["GSPC_Close", "VIX_Close"], how="all")


st.set_page_config(
    page_title="Crisis Radar — ML Pipeline",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.35rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        text-align: center;
        color: #444;
        font-size: 1.05rem;
        margin-bottom: 1.25rem;
    }
    .risk-high { color: #c0392b; font-weight: 700; font-size: 1.45rem; }
    .risk-medium { color: #d68910; font-weight: 700; font-size: 1.45rem; }
    .risk-low { color: #1e8449; font-weight: 700; font-size: 1.45rem; }
    </style>
""",
    unsafe_allow_html=True,
)

if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

PAGE_KEYS = ["home", "train", "live", "test", "viz", "about"]

with st.sidebar:
    lang = st.selectbox(
        "Language / Γλώσσα",
        options=["el", "en"],
        format_func=lambda c: "Ελληνικά" if c == "el" else "English",
        index=0,
        help="Επιλέξτε γλώσσα διεπαφής / UI language",
    )
    st.header(t(lang, "nav_header"))
    page_labels = [t(lang, f"page_{k}") for k in PAGE_KEYS]
    page_idx = st.radio(
        "page_nav",
        range(len(PAGE_KEYS)),
        format_func=lambda i: page_labels[i],
        label_visibility="collapsed",
    )
    page = PAGE_KEYS[page_idx]
    st.markdown("---")

    model_path = Path("artifacts/models/calibrated_model.pkl")
    if model_path.exists():
        st.success(t(lang, "sidebar_model_ready"))
        st.session_state.model_trained = True
    else:
        st.warning(t(lang, "sidebar_no_model"))
        st.session_state.model_trained = False

    st.markdown("---")
    st.markdown(t(lang, "sidebar_credits"))
    st.markdown(t(lang, "sidebar_version"))

st.markdown('<div class="main-header">Crisis Radar</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{t(lang, "home_tagline")}</div>', unsafe_allow_html=True)
st.markdown("---")

if page == "home":
    st.header(t(lang, "home_title"))
    st.markdown(t(lang, "home_intro"))
    st.markdown(f"### {t(lang, 'home_features_title')}")
    st.markdown(t(lang, "home_features"))
    st.markdown(f"### {t(lang, 'home_qs_title')}")
    st.markdown(t(lang, "home_qs"))
    st.markdown(f"### {t(lang, 'home_decision_title')}")
    st.markdown(t(lang, "home_decision"))
    with st.expander(t(lang, "decision_cheatsheet"), expanded=False):
        st.markdown(t(lang, "decision_cheatsheet_body"))
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.model_trained:
            st.success(t(lang, "home_status_model_ok"))
        else:
            st.error(t(lang, "home_status_model_no"))
    with col2:
        if Path("artifacts/metrics/test_metrics.json").exists():
            metrics = load_json("artifacts/metrics/test_metrics.json")
            st.metric(t(lang, "home_status_prauc"), f"{metrics.get('pr_auc', 0):.4f}")
        else:
            st.metric(t(lang, "home_status_prauc"), "N/A")
    with col3:
        if Path("data/raw/market_data.csv").exists():
            st.success(t(lang, "home_status_data_ok"))
        else:
            st.info(t(lang, "home_status_data_no"))

elif page == "train":
    st.header(t(lang, "train_title"))
    with st.expander(t(lang, "train_expander"), expanded=False):
        st.markdown(t(lang, "train_expander_body"))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t(lang, "train_sub_model"))
        model_type = st.selectbox(t(lang, "train_model_type"), ["rf", "gb", "logreg"], index=0)
        calibration_method = st.selectbox(
            t(lang, "train_calibration"), ["isotonic", "sigmoid"], index=0
        )
        horizon_days = st.number_input(
            t(lang, "train_horizon"), min_value=5, max_value=60, value=20
        )
        drawdown_threshold = (
            st.slider(t(lang, "train_dd"), min_value=1.0, max_value=20.0, value=8.0) / 100
        )

    with col2:
        st.subheader(t(lang, "train_sub_data"))
        train_start = st.date_input(t(lang, "train_start"), value=datetime(2005, 1, 1).date())
        train_end = st.date_input(t(lang, "train_end"), value=datetime(2015, 12, 31).date())
        test_start = st.date_input(t(lang, "test_start"), value=datetime(2016, 1, 1).date())
        test_end = st.date_input(t(lang, "test_end"), value=date.today())
        data_start = st.date_input(t(lang, "data_start"), value=datetime(2000, 1, 1).date())
        random_seed = st.number_input(t(lang, "train_seed"), min_value=0, max_value=9999, value=42)

    with st.expander(t(lang, "train_advanced")):
        n_estimators = st.number_input(t(lang, "train_n_est"), min_value=10, max_value=1000, value=100)
        unlimited_depth = st.checkbox(
            t(lang, "train_max_depth_unlimited"),
            value=True,
            help=t(lang, "train_max_depth_help"),
        )
        if unlimited_depth:
            max_depth_val = None
        else:
            max_depth_val = st.number_input(
                t(lang, "train_max_depth"), min_value=1, max_value=30, value=8
            )

    if st.button(t(lang, "train_button"), type="primary"):
        with st.spinner(t(lang, "train_spinner")):
            try:
                config = CrisisRadarConfig(
                    start_date=data_start.strftime("%Y-%m-%d"),
                    end_date=test_end.strftime("%Y-%m-%d"),
                    horizon_days=int(horizon_days),
                    drawdown_threshold=float(drawdown_threshold),
                    train_start=train_start.strftime("%Y-%m-%d"),
                    train_end=train_end.strftime("%Y-%m-%d"),
                    test_start=test_start.strftime("%Y-%m-%d"),
                    test_end=test_end.strftime("%Y-%m-%d"),
                    model_type=model_type,
                    calibration_method=calibration_method,
                    n_estimators=int(n_estimators),
                    max_depth=max_depth_val,
                    random_seed=int(random_seed),
                )
                artifacts = run_pipeline(config)
                st.session_state.pipeline_run = True
                st.session_state.model_trained = True
                st.success(t(lang, "train_success"))
                st.balloons()
                if Path(artifacts["metrics"]).exists():
                    metrics = load_json(artifacts["metrics"])
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
                    with c2:
                        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                    with c3:
                        st.metric("Brier", f"{metrics['brier_score']:.4f}")
                    with c4:
                        st.metric("F1", f"{metrics['f1']:.4f}")
            except Exception as e:
                st.error(f"{t(lang, 'train_fail')}: {e}")
                st.exception(e)

elif page == "live":
    st.header(t(lang, "live_title"))
    if not st.session_state.model_trained:
        st.warning(t(lang, "live_warn"))
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(t(lang, "live_sub_get"))
            force_refresh = st.checkbox(
                t(lang, "live_refresh"),
                value=False,
                help=t(lang, "live_refresh_help"),
            )
        with col2:
            st.subheader(t(lang, "live_sub_info"))
            cfg_path = Path("artifacts/configs/pipeline_config.json")
            if cfg_path.exists():
                config = load_json(str(cfg_path))
                st.info(
                    f"{t(lang, 'live_horizon')}: {config['horizon_days']} · "
                    f"{t(lang, 'live_threshold')}: −{config['drawdown_threshold']:.1%}"
                )
            else:
                st.caption(t(lang, "live_no_config"))

        if st.button(t(lang, "live_button"), type="primary"):
            with st.spinner(t(lang, "live_spinner")):
                try:
                    result = get_live_risk_score(force_refresh=force_refresh)
                    st.markdown("---")
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c2:
                        st.markdown(f"### {t(lang, 'live_date')}: {result['date']}")
                        st.caption(
                            f"{t(lang, 'live_horizon')}: {result['horizon_days']} · "
                            f"{t(lang, 'live_threshold')}: −{result['drawdown_threshold']:.1%}"
                        )
                        st.markdown("---")
                        risk_class = f"risk-{result['risk_level'].lower()}"
                        st.markdown(
                            f'<div class="{risk_class}">{result["risk_level"]}</div>',
                            unsafe_allow_html=True,
                        )
                        prob = result["probability"]
                        st.metric(t(lang, "live_prob"), f"{prob:.2%}")
                        st.progress(min(max(float(prob), 0.0), 1.0))
                        st.markdown("---")
                        st.markdown(f"**{t(lang, 'live_interp_title')}**")
                        st.info(
                            t(
                                lang,
                                "live_interp_body",
                                prob=float(prob),
                                dd=float(result["drawdown_threshold"]),
                                h=int(result["horizon_days"]),
                            )
                        )
                        with st.expander(t(lang, "decision_cheatsheet"), expanded=False):
                            st.markdown(t(lang, "decision_cheatsheet_body"))
                except Exception as e:
                    st.error(f"{t(lang, 'live_fail')}: {e}")
                    st.exception(e)

elif page == "test":
    st.header(t(lang, "test_title"))
    if not st.session_state.model_trained:
        st.warning(t(lang, "test_warn"))
    else:
        st.markdown(t(lang, "test_intro"))
        st.download_button(
            label=t(lang, "test_sample_btn"),
            data=sample_csv_bytes(),
            file_name="crisis_radar_sample_template.csv",
            mime="text/csv",
        )
        uploaded_file = st.file_uploader(t(lang, "test_uploader"), type="csv")
        if uploaded_file is not None:
            try:
                raw = pd.read_csv(uploaded_file)
                df_uploaded = _normalize_uploaded_df(raw)
                st.success(f"{len(df_uploaded)} {t(lang, 'common_rows')}")
                with st.expander(t(lang, "test_preview")):
                    st.dataframe(df_uploaded.head(10))
                    st.write(f"{t(lang, 'test_shape')}: {df_uploaded.shape}")
                    st.write(f"{t(lang, 'test_cols')}: {list(df_uploaded.columns)}")
                required_cols = ["GSPC_Close", "VIX_Close"]
                missing_cols = [c for c in required_cols if c not in df_uploaded.columns]
                if missing_cols:
                    st.error(f"{t(lang, 'test_missing')}: {missing_cols}")
                elif st.button(t(lang, "test_button"), type="primary"):
                    with st.spinner(t(lang, "test_spinner")):
                        try:
                            df_spx = df_uploaded["GSPC_Close"]
                            df_vix = df_uploaded["VIX_Close"]
                            df_features = build_features(df_spx, df_vix)
                            with open("artifacts/models/calibrated_model.pkl", "rb") as f:
                                model = pickle.load(f)
                            feature_names = get_feature_names()
                            X_test = df_features[feature_names].values
                            if pd.isna(X_test).any():
                                st.warning(t(lang, "test_nan_warn"))
                                X_test = pd.DataFrame(X_test, columns=feature_names).ffill().values
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                            results_df = pd.DataFrame(
                                {"date": df_features.index, "predicted_probability": y_pred_proba}
                            )
                            st.success(t(lang, "test_success_pred"))
                            st.subheader(t(lang, "test_results"))
                            st.dataframe(results_df)
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label=t(lang, "test_download"),
                                data=csv,
                                file_name=f"predictions_{date.today().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                            )
                            m1, m2, m3 = st.columns(3)
                            with m1:
                                st.metric(t(lang, "test_mean"), f"{y_pred_proba.mean():.2%}")
                            with m2:
                                st.metric(t(lang, "test_max"), f"{y_pred_proba.max():.2%}")
                            with m3:
                                high_risk = (y_pred_proba >= 0.3).sum()
                                st.metric(
                                    t(lang, "test_high_days"),
                                    f"{high_risk} ({high_risk / max(len(y_pred_proba), 1):.1%})",
                                )
                        except Exception as e:
                            st.error(f"{t(lang, 'test_fail')}: {e}")
                            st.exception(e)
            except Exception as e:
                st.error(f"{t(lang, 'test_read_fail')}: {e}")
                st.exception(e)

elif page == "viz":
    st.header(t(lang, "viz_title"))
    if not st.session_state.model_trained:
        st.warning(t(lang, "viz_warn"))
    else:
        figures_dir = Path("reports/figures")
        if not figures_dir.exists() or len(list(figures_dir.glob("*.png"))) == 0:
            st.info(t(lang, "viz_empty"))
        else:
            metrics_path = Path("artifacts/metrics/test_metrics.json")
            if metrics_path.exists():
                metrics = load_json(str(metrics_path))
                a1, a2, a3, a4 = st.columns(4)
                with a1:
                    st.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
                with a2:
                    st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                with a3:
                    st.metric("Brier", f"{metrics['brier_score']:.4f}")
                with a4:
                    st.metric("F1", f"{metrics['f1']:.4f}")
                st.markdown("---")
            fig_files = {
                "Probability Timeline": "probability_timeline",
                "Precision-Recall Curve": "precision_recall_curve",
                "Calibration Curve": "calibration_curve",
                "Confusion Matrix": "confusion_matrix",
                "Feature Importance": "feature_importance",
            }
            for fig_name, fig_base in fig_files.items():
                pickle_path = figures_dir / f"{fig_base}.pkl"
                html_path = figures_dir / f"{fig_base}.html"
                png_path = figures_dir / f"{fig_base}.png"
                if pickle_path.exists():
                    st.subheader(fig_name)
                    try:
                        with open(pickle_path, "rb") as f:
                            fig = pickle.load(f)
                        st.plotly_chart(fig, use_container_width=True, height=600)
                    except Exception as e:
                        st.warning(f"{e}")
                        if png_path.exists():
                            st.image(str(png_path))
                elif html_path.exists():
                    st.subheader(fig_name)
                    st.caption("Re-train to regenerate interactive pickle charts.")
                    try:
                        with open(html_path, encoding="utf-8") as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=600, scrolling=False)
                    except Exception:
                        if png_path.exists():
                            st.image(str(png_path))
                    try:
                        with open(html_path, "rb") as f:
                            st.download_button(
                                label=f"Download {fig_name} (HTML)",
                                data=f.read(),
                                file_name=f"{fig_base}.html",
                                mime="text/html",
                                key=f"dl_{fig_base}",
                            )
                    except OSError:
                        pass
                    st.markdown("---")
                elif png_path.exists():
                    st.subheader(fig_name)
                    st.image(str(png_path))
                    st.markdown("---")

elif page == "about":
    st.header(t(lang, "about_title"))
    st.markdown(t(lang, "about_body"))
