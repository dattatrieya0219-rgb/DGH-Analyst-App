"""
analyst_pro_app.py
===================
Auto Data Analyst PRO — a single-file Streamlit application covering the
practical core of a data-analyst workflow:

    1. Upload   - CSV, Excel, JSON, Parquet, TSV/TXT (multi-format ingestion)
    2. Clean    - missing values, duplicates, outliers, type conversion,
                  encoding, scaling — with a running cleaning log + undo
    3. Explore  - automated EDA: profiling, univariate, correlation,
                  scatter matrix, time-series trend (if a date column exists)
    4. Stats    - descriptive stats, hypothesis tests (t-tests, chi-square,
                  ANOVA, Mann-Whitney, Wilcoxon, KS normality), and a
                  statsmodels-backed regression module with p-values & CIs
    5. Model    - automated ML: problem-type detection, multiple algorithms,
                  train/test split, metrics, confusion matrix, ROC-AUC,
                  feature importance, optional Random Forest tuning
    6. Report   - compiles everything done in the session into a
                  downloadable Markdown + HTML report

Run with:
    streamlit run analyst_pro_app.py

Install once:
    pip install -r requirements.txt
"""

import io
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sps

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Auto Data Analyst PRO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_HISTORY = 15  # cap undo stack so memory doesn't grow unbounded


# --------------------------------------------------------------------------- #
# SESSION STATE
# --------------------------------------------------------------------------- #
def init_state() -> None:
    """Initialize all session_state keys used across the app."""
    defaults = {
        "raw_df": None,          # original, untouched upload
        "df": None,              # working (possibly cleaned) dataframe
        "history": [],           # stack of previous df snapshots (for undo)
        "cleaning_log": [],      # list of human-readable log strings
        "test_results": [],      # hypothesis test results, for the report
        "regression_results": None,  # last statsmodels regression summary
        "model_comparison": None,    # ML model comparison dataframe
        "model_details": {},         # per-model fitted objects & extras
        "model_meta": None,          # problem type / target / features / class names
        "file_name": None,
        "report_md": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def push_history() -> None:
    """Save a snapshot of the current df before a mutating operation."""
    st.session_state.history.append(st.session_state.df.copy())
    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history.pop(0)


def log_action(message: str) -> None:
    """Append a timestamped entry to the cleaning log."""
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.cleaning_log.append(f"`{ts}` — {message}")


def undo_last() -> bool:
    """Revert to the previous df snapshot, if one exists."""
    if st.session_state.history:
        st.session_state.df = st.session_state.history.pop()
        log_action("↩️ Undo — reverted the last cleaning operation.")
        return True
    return False


# --------------------------------------------------------------------------- #
# 1. DATA LOADING (multi-format)
# --------------------------------------------------------------------------- #
def load_data(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """
    Parse an uploaded file into a DataFrame. Supports CSV (auto delimiter
    detection), Excel (.xlsx/.xls), JSON, Parquet, and TSV/TXT.
    """
    name = file_name.lower()
    buffer = io.BytesIO(file_bytes)

    if name.endswith(".csv"):
        # sep=None + engine='python' auto-detects comma / semicolon / tab
        # (low_memory is not supported by the python engine, so it's omitted here)
        df = pd.read_csv(buffer, sep=None, engine="python")
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(buffer)
    elif name.endswith(".json"):
        try:
            df = pd.read_json(buffer)
        except ValueError:
            buffer.seek(0)
            df = pd.read_json(buffer, lines=True)
    elif name.endswith(".parquet"):
        df = pd.read_parquet(buffer)
    elif name.endswith((".tsv", ".txt")):
        df = pd.read_csv(buffer, sep=None, engine="python")
    else:
        raise ValueError(
            "Unsupported file type. Please upload CSV, Excel, JSON, Parquet, or TSV/TXT."
        )

    df.columns = [str(c).strip() for c in df.columns]

    # Automatic date detection: convert object columns that look like dates,
    # keeping the conversion only if >= 80% of non-null values parse cleanly.
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty:
            continue
        looks_datey = sample.str.contains(
            r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", regex=True
        ).mean() > 0.6
        if looks_datey:
            converted = pd.to_datetime(df[col], errors="coerce")
            success_rate = converted.notna().sum() / max(df[col].notna().sum(), 1)
            if success_rate >= 0.8:
                df[col] = converted

    return df


def split_columns(df: pd.DataFrame):
    """Return (numeric_cols, categorical_cols, datetime_cols)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]
    return numeric_cols, categorical_cols, datetime_cols


# --------------------------------------------------------------------------- #
# 2. DATA CLEANING — pure helper functions (no st.* calls, easy to test)
# --------------------------------------------------------------------------- #
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table of columns with missing values and their %."""
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        return pd.DataFrame(columns=["Column", "Missing Count", "Missing %"])
    return pd.DataFrame(
        {
            "Column": miss.index,
            "Missing Count": miss.values,
            "Missing %": (miss.values / len(df) * 100).round(2),
        }
    ).sort_values("Missing %", ascending=False).reset_index(drop=True)


def impute_column(df: pd.DataFrame, col: str, method: str, custom_value=None) -> pd.DataFrame:
    """Impute missing values in a single column using the chosen method."""
    out = df.copy()
    if method == "Mean":
        out[col] = out[col].fillna(out[col].mean())
    elif method == "Median":
        out[col] = out[col].fillna(out[col].median())
    elif method == "Mode":
        mode_val = out[col].mode(dropna=True)
        out[col] = out[col].fillna(mode_val.iloc[0] if not mode_val.empty else np.nan)
    elif method == "Forward fill":
        out[col] = out[col].ffill()
    elif method == "Backward fill":
        out[col] = out[col].bfill()
    elif method == "Custom value":
        out[col] = out[col].fillna(custom_value)
    elif method == "Drop rows":
        out = out[out[col].notna()].reset_index(drop=True)
    return out


def remove_duplicates(df: pd.DataFrame) -> tuple:
    """Drop fully duplicated rows. Returns (new_df, n_removed)."""
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    return out, before - len(out)


def detect_outliers_iqr(series: pd.Series):
    """Return (mask_of_outliers, lower_bound, upper_bound) via 1.5*IQR rule."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return mask, lower, upper


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0):
    """Return (mask_of_outliers, threshold) via Z-score method."""
    mean, std = series.mean(), series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(False, index=series.index), threshold
    z = (series - mean) / std
    mask = z.abs() > threshold
    return mask, threshold


def treat_outliers(df: pd.DataFrame, col: str, method: str, action: str, threshold: float = 3.0) -> tuple:
    """
    Treat outliers in a numeric column.

    method: 'IQR' or 'Z-score'
    action: 'Cap (winsorize)', 'Remove rows', or 'Flag only (new column)'
    Returns (new_df, n_affected).
    """
    out = df.copy()
    series = out[col]

    if method == "IQR":
        mask, lower, upper = detect_outliers_iqr(series)
    else:
        mask, thr = detect_outliers_zscore(series, threshold)
        lower, upper = series.mean() - thr * series.std(), series.mean() + thr * series.std()

    n_affected = int(mask.sum())

    if action == "Cap (winsorize)":
        out[col] = series.clip(lower=lower, upper=upper)
    elif action == "Remove rows":
        out = out[~mask].reset_index(drop=True)
    elif action == "Flag only (new column)":
        out[f"{col}_is_outlier"] = mask

    return out, n_affected


def convert_dtype(df: pd.DataFrame, col: str, target_type: str) -> pd.DataFrame:
    """Convert a column to a target dtype: Text, Numeric, Date, or Categorical."""
    out = df.copy()
    if target_type == "Text":
        out[col] = out[col].astype(str)
    elif target_type == "Numeric":
        out[col] = pd.to_numeric(out[col], errors="coerce")
    elif target_type == "Date":
        out[col] = pd.to_datetime(out[col], errors="coerce")
    elif target_type == "Categorical":
        out[col] = out[col].astype("category")
    return out


def encode_columns(df: pd.DataFrame, cols: list, method: str) -> pd.DataFrame:
    """One-hot or label encode the given categorical columns."""
    out = df.copy()
    if method == "One-hot encoding":
        out = pd.get_dummies(out, columns=cols, prefix=cols)
    elif method == "Label encoding":
        for col in cols:
            out[col] = out[col].astype("category").cat.codes
    return out


def scale_columns(df: pd.DataFrame, cols: list, method: str) -> pd.DataFrame:
    """Min-max scale, Z-score standardize, or log-transform numeric columns."""
    out = df.copy()
    for col in cols:
        series = out[col]
        if method == "Min-max scaling":
            rng = series.max() - series.min()
            out[col] = (series - series.min()) / rng if rng != 0 else 0.0
        elif method == "Z-score standardization":
            std = series.std()
            out[col] = (series - series.mean()) / std if std != 0 else 0.0
        elif method == "Log transform":
            # shift so the minimum is > 0 before taking the log
            shift = abs(series.min()) + 1 if series.min() <= 0 else 0
            out[col] = np.log(series + shift)
    return out


# --------------------------------------------------------------------------- #
# TAB 1 — UPLOAD & PREVIEW
# --------------------------------------------------------------------------- #
def render_upload_tab() -> None:
    df = st.session_state.df
    st.subheader("Dataset Preview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    c3.metric("Memory Usage", f"{mem_mb:.2f} MB")

    st.write("**First 10 rows**")
    st.dataframe(df.head(10), use_container_width=True)
    st.write("**Last 10 rows**")
    st.dataframe(df.tail(10), use_container_width=True)

    st.write("**Column Types**")
    numeric_cols, categorical_cols, datetime_cols = split_columns(df)
    type_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Detected Type": [
                "Numeric" if c in numeric_cols else "Date/Time" if c in datetime_cols else "Categorical/Text"
                for c in df.columns
            ],
            "Pandas dtype": [str(df[c].dtype) for c in df.columns],
        }
    )
    st.dataframe(type_df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# TAB 2 — CLEAN
# --------------------------------------------------------------------------- #
def render_clean_tab() -> None:
    df = st.session_state.df
    numeric_cols, categorical_cols, datetime_cols = split_columns(df)

    top = st.columns([1, 1, 4])
    if top[0].button("↩️ Undo Last Step", disabled=not st.session_state.history):
        if undo_last():
            st.rerun()
    if top[1].button("🔄 Reset to Original"):
        push_history()
        st.session_state.df = st.session_state.raw_df.copy()
        log_action("🔄 Reset — reverted all cleaning back to the original upload.")
        st.rerun()

    section = st.selectbox(
        "Choose a cleaning operation:",
        [
            "Missing values",
            "Duplicates",
            "Outliers",
            "Data type conversion",
            "Rename / drop columns",
            "Encode categorical variables",
            "Scale / transform numeric columns",
        ],
    )
    st.markdown("---")

    # ---- Missing values ------------------------------------------------------
    if section == "Missing values":
        summary = missing_summary(df)
        if summary.empty:
            st.success("✅ No missing values in the current dataset.")
        else:
            st.dataframe(summary, use_container_width=True, hide_index=True)
            fig = px.imshow(
                df[summary["Column"]].isna().T,
                labels=dict(x="Row index", y="Column", color="Missing"),
                aspect="auto",
                title="Missing Data Pattern (yellow = missing)",
            )
            st.plotly_chart(fig, use_container_width=True)

            col = st.selectbox("Column to fix:", summary["Column"].tolist())
            is_numeric = col in numeric_cols
            options = ["Mean", "Median", "Mode", "Forward fill", "Backward fill", "Custom value", "Drop rows"]
            if not is_numeric:
                options = [o for o in options if o not in ("Mean", "Median")]
            method = st.selectbox("Imputation method:", options)
            custom_value = None
            if method == "Custom value":
                custom_value = st.text_input("Value to fill with:")
                if is_numeric and custom_value:
                    try:
                        custom_value = float(custom_value)
                    except ValueError:
                        st.warning("Enter a numeric value for a numeric column.")

            if st.button("Apply imputation"):
                push_history()
                before_rows = len(df)
                st.session_state.df = impute_column(df, col, method, custom_value)
                after_rows = len(st.session_state.df)
                extra = f" ({before_rows - after_rows} rows dropped)" if method == "Drop rows" else ""
                log_action(f"Filled missing values in **{col}** using **{method}**{extra}.")
                st.rerun()

    # ---- Duplicates -----------------------------------------------------------
    elif section == "Duplicates":
        dup_count = int(df.duplicated().sum())
        if dup_count == 0:
            st.success("✅ No duplicate rows found.")
        else:
            st.warning(f"⚠ Found **{dup_count:,}** duplicate rows.")
            st.dataframe(df[df.duplicated(keep=False)].sort_values(list(df.columns)).head(50), use_container_width=True)
            if st.button("Remove duplicate rows"):
                push_history()
                st.session_state.df, n_removed = remove_duplicates(df)
                log_action(f"Removed **{n_removed}** duplicate rows.")
                st.rerun()

    # ---- Outliers ---------------------------------------------------------------
    elif section == "Outliers":
        if not numeric_cols:
            st.info("No numeric columns available for outlier detection.")
        else:
            col = st.selectbox("Column:", numeric_cols)
            method = st.radio("Detection method:", ["IQR", "Z-score"], horizontal=True)
            threshold = 3.0
            if method == "Z-score":
                threshold = st.slider("Z-score threshold:", 2.0, 5.0, 3.0, 0.5)
                mask, _ = detect_outliers_zscore(df[col], threshold)
            else:
                mask, lower, upper = detect_outliers_iqr(df[col])
                st.caption(f"Bounds: [{lower:.3f}, {upper:.3f}]")

            st.write(f"**{int(mask.sum())}** potential outliers detected ({mask.mean()*100:.1f}% of non-null rows).")
            fig = px.box(df, y=col, points="outliers", title=f"Boxplot of {col}")
            st.plotly_chart(fig, use_container_width=True)

            action = st.selectbox(
                "Action:", ["Cap (winsorize)", "Remove rows", "Flag only (new column)"]
            )
            if st.button("Apply outlier treatment"):
                push_history()
                st.session_state.df, n_affected = treat_outliers(df, col, method, action, threshold)
                log_action(f"Treated **{n_affected}** outliers in **{col}** via {method} → {action}.")
                st.rerun()

    # ---- Data type conversion -----------------------------------------------------
    elif section == "Data type conversion":
        col = st.selectbox("Column:", df.columns.tolist())
        target = st.selectbox("Convert to:", ["Text", "Numeric", "Date", "Categorical"])
        if st.button("Apply conversion"):
            push_history()
            before_dtype = str(df[col].dtype)
            st.session_state.df = convert_dtype(df, col, target)
            after_dtype = str(st.session_state.df[col].dtype)
            log_action(f"Converted **{col}** from `{before_dtype}` to `{after_dtype}` ({target}).")
            st.rerun()

    # ---- Rename / drop columns --------------------------------------------------
    elif section == "Rename / drop columns":
        r1, r2 = st.columns(2)
        with r1:
            st.write("**Rename a column**")
            old_name = st.selectbox("Column to rename:", df.columns.tolist(), key="rename_old")
            new_name = st.text_input("New name:", value=old_name)
            if st.button("Rename"):
                if new_name and new_name != old_name:
                    push_history()
                    st.session_state.df = df.rename(columns={old_name: new_name})
                    log_action(f"Renamed column **{old_name}** → **{new_name}**.")
                    st.rerun()
        with r2:
            st.write("**Drop columns**")
            drop_cols = st.multiselect("Columns to drop:", df.columns.tolist())
            if st.button("Drop selected columns", disabled=not drop_cols):
                push_history()
                st.session_state.df = df.drop(columns=drop_cols)
                log_action(f"Dropped columns: {', '.join(drop_cols)}.")
                st.rerun()

    # ---- Encoding --------------------------------------------------------------
    elif section == "Encode categorical variables":
        if not categorical_cols:
            st.info("No categorical columns available to encode.")
        else:
            cols = st.multiselect("Columns to encode:", categorical_cols)
            method = st.selectbox("Method:", ["One-hot encoding", "Label encoding"])
            if st.button("Apply encoding", disabled=not cols):
                push_history()
                before_shape = df.shape
                st.session_state.df = encode_columns(df, cols, method)
                after_shape = st.session_state.df.shape
                log_action(
                    f"Applied **{method}** to {', '.join(cols)} "
                    f"(columns: {before_shape[1]} → {after_shape[1]})."
                )
                st.rerun()

    # ---- Scaling ----------------------------------------------------------------
    elif section == "Scale / transform numeric columns":
        if not numeric_cols:
            st.info("No numeric columns available to scale.")
        else:
            cols = st.multiselect("Columns to transform:", numeric_cols)
            method = st.selectbox(
                "Method:", ["Min-max scaling", "Z-score standardization", "Log transform"]
            )
            if st.button("Apply transform", disabled=not cols):
                push_history()
                st.session_state.df = scale_columns(df, cols, method)
                log_action(f"Applied **{method}** to: {', '.join(cols)}.")
                st.rerun()

    # ---- Cleaning log -------------------------------------------------------------
    st.markdown("---")
    st.subheader("📝 Cleaning Log")
    if st.session_state.cleaning_log:
        for entry in reversed(st.session_state.cleaning_log):
            st.markdown(f"- {entry}")
    else:
        st.caption("No cleaning steps applied yet.")


# --------------------------------------------------------------------------- #
# TAB 3 — EXPLORE (EDA)
# --------------------------------------------------------------------------- #
def build_profile_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column profile: type, nulls, uniques, most frequent value."""
    rows = []
    n = len(df)
    for col in df.columns:
        series = df[col]
        non_null = series.notna().sum()
        try:
            mode_val = series.mode(dropna=True)
            most_freq = str(mode_val.iloc[0]) if not mode_val.empty else "—"
        except TypeError:
            most_freq = "—"
        rows.append(
            {
                "Column": col,
                "Data Type": str(series.dtype),
                "Non-Null Count": int(non_null),
                "Null %": round(100 * (n - non_null) / n, 2) if n else 0.0,
                "Unique Values": int(series.nunique(dropna=True)),
                "Most Frequent Value": most_freq[:50],
            }
        )
    return pd.DataFrame(rows)


def normality_test(series: pd.Series) -> dict:
    """Run Shapiro-Wilk (sampled if large) and KS-test-vs-normal on a column."""
    clean = series.dropna()
    if len(clean) < 3:
        return {}
    sample = clean.sample(min(len(clean), 5000), random_state=42) if len(clean) > 5000 else clean
    shapiro_stat, shapiro_p = sps.shapiro(sample)
    standardized = (clean - clean.mean()) / clean.std() if clean.std() != 0 else clean * 0
    ks_stat, ks_p = sps.kstest(standardized, "norm")
    return {
        "Shapiro-Wilk stat": round(shapiro_stat, 4),
        "Shapiro-Wilk p-value": round(shapiro_p, 4),
        "KS stat": round(ks_stat, 4),
        "KS p-value": round(ks_p, 4),
        "Likely Normal?": "Yes" if shapiro_p > 0.05 and ks_p > 0.05 else "No",
    }


def render_explore_tab() -> None:
    df = st.session_state.df
    numeric_cols, categorical_cols, datetime_cols = split_columns(df)

    st.subheader("Data Profile")
    st.dataframe(build_profile_table(df), use_container_width=True, hide_index=True)

    st.subheader("Univariate Analysis")
    all_cols = numeric_cols + categorical_cols
    if all_cols:
        selected = st.selectbox("Column to visualize:", all_cols, key="uni_col")
        if selected in numeric_cols:
            clean = df[selected].dropna()
            c1, c2 = st.columns(2)
            with c1:
                if clean.nunique() > 1:
                    try:
                        fig = ff.create_distplot([clean.tolist()], [selected], show_rug=False)
                        fig.update_layout(title=f"Distribution of {selected}")
                    except Exception:
                        # figure_factory's KDE path can raise version-mismatch
                        # errors on some hosting environments — fall back to a
                        # plain histogram with a smoothed overlay instead.
                        fig = px.histogram(df, x=selected, title=f"Distribution of {selected}",
                                            marginal="box")
                else:
                    fig = px.histogram(df, x=selected, title=f"Distribution of {selected}")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.box(df, y=selected, points="outliers", title=f"Boxplot of {selected}")
                st.plotly_chart(fig2, use_container_width=True)

            with st.expander("🔬 Normality test for this column"):
                result = normality_test(df[selected])
                if result:
                    st.table(pd.DataFrame([result]).T.rename(columns={0: "Value"}))
                    st.caption(
                        "H0: the data comes from a normal distribution. "
                        "p > 0.05 on both tests means we fail to reject H0 (looks normal)."
                    )
        else:
            top_counts = df[selected].astype(str).value_counts().head(20)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(x=top_counts.index, y=top_counts.values,
                             labels={"x": selected, "y": "Count"}, title=f"Top categories in {selected}")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.pie(values=top_counts.values, names=top_counts.index,
                              title=f"Share of {selected}")
                st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation Analysis (Bivariate)")
    if len(numeric_cols) >= 2:
        method = st.radio("Correlation method:", ["Pearson", "Spearman"], horizontal=True, key="corr_method")
        corr = df[numeric_cols].corr(method=method.lower()).round(3)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title=f"{method} Correlation Heatmap", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        x_col = c1.selectbox("X (scatter):", numeric_cols, key="scatter_x")
        y_col = c2.selectbox("Y (scatter):", numeric_cols, index=min(1, len(numeric_cols)-1), key="scatter_y")
        color_opts = ["(none)"] + [c for c in categorical_cols if df[c].nunique() <= 10]
        color_col = st.selectbox("Color by (optional):", color_opts, key="scatter_color")
        color_arg = None if color_col == "(none)" else color_col
        try:
            # Trendline requires statsmodels under the hood; on some hosted
            # environments a plotly/narwhals version mismatch makes this
            # raise a DuplicateError. Fall back to a plain scatter if so.
            fig2 = px.scatter(df, x=x_col, y=y_col, color=color_arg,
                               trendline="ols", title=f"{x_col} vs {y_col}")
        except Exception:
            fig2 = px.scatter(df, x=x_col, y=y_col, color=color_arg,
                               title=f"{x_col} vs {y_col}")
            st.caption("ℹ️ Trend line unavailable in this environment — showing scatter without it.")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Need at least two numeric columns for correlation analysis.")

    st.subheader("Scatter Matrix (Pair Plot)")
    if 2 <= len(numeric_cols) <= 10:
        color_opts = ["(none)"] + [c for c in categorical_cols if df[c].nunique() <= 10]
        color_col = st.selectbox("Optional color grouping:", color_opts, key="matrix_color")
        fig = px.scatter_matrix(df, dimensions=numeric_cols,
                                 color=None if color_col == "(none)" else color_col,
                                 height=max(600, 160 * len(numeric_cols)))
        fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=3))
        st.plotly_chart(fig, use_container_width=True)
    elif len(numeric_cols) > 10:
        st.info(f"Scatter matrix skipped: {len(numeric_cols)} numeric columns (limit is 10).")

    if datetime_cols and numeric_cols:
        st.subheader("Time Series Trend")
        date_col = st.selectbox("Date column:", datetime_cols, key="ts_date")
        metric_col = st.selectbox("Metric:", numeric_cols, key="ts_metric")
        ts = df[[date_col, metric_col]].dropna().sort_values(date_col)
        if len(ts) > 1:
            window = st.slider("Rolling average window (periods):", 1, max(2, min(30, len(ts)//2)), min(7, max(1, len(ts)//10)))
            ts["Rolling Avg"] = ts[metric_col].rolling(window, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts[date_col], y=ts[metric_col], mode="lines", name=metric_col, opacity=0.4))
            fig.add_trace(go.Scatter(x=ts[date_col], y=ts["Rolling Avg"], mode="lines", name=f"{window}-period rolling avg"))
            fig.update_layout(title=f"{metric_col} over time", xaxis_title=date_col, yaxis_title=metric_col)
            st.plotly_chart(fig, use_container_width=True)

            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                freq_guess = min(12, max(2, len(ts) // 4))
                indexed = ts.set_index(date_col)[metric_col].asfreq(
                    pd.infer_freq(ts[date_col]) or "D"
                ).interpolate()
                if len(indexed) >= 2 * freq_guess:
                    decomp = seasonal_decompose(indexed, period=freq_guess, model="additive", extrapolate_trend="freq")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=indexed.index, y=decomp.trend, name="Trend"))
                    fig2.add_trace(go.Scatter(x=indexed.index, y=decomp.seasonal, name="Seasonal"))
                    fig2.add_trace(go.Scatter(x=indexed.index, y=decomp.resid, name="Residual"))
                    fig2.update_layout(title="Seasonal Decomposition (additive)")
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                st.caption("Seasonal decomposition skipped — not enough regularly-spaced data points.")
        else:
            st.info("Not enough non-missing data points to plot a trend.")


# --------------------------------------------------------------------------- #
# TAB 4 — STATISTICS (descriptive stats, hypothesis tests, regression)
# --------------------------------------------------------------------------- #
def interpret_p(p_value: float, alpha: float = 0.05) -> str:
    """Plain-English interpretation of a p-value."""
    if p_value < alpha:
        return (
            f"p = {p_value:.4f} < {alpha} → **statistically significant**. "
            "We reject the null hypothesis; the observed effect is unlikely to be due to chance alone."
        )
    return (
        f"p = {p_value:.4f} ≥ {alpha} → **not statistically significant**. "
        "We fail to reject the null hypothesis; there isn't enough evidence of a real effect."
    )


def render_stats_tab() -> None:
    df = st.session_state.df
    numeric_cols, categorical_cols, _ = split_columns(df)

    st.subheader("Descriptive Statistics")
    if numeric_cols:
        desc = df[numeric_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
        desc["skew"] = df[numeric_cols].skew()
        desc["kurtosis"] = df[numeric_cols].kurtosis()
        st.dataframe(desc.round(3), use_container_width=True)
    else:
        st.info("No numeric columns for descriptive statistics.")

    st.markdown("---")
    st.subheader("Hypothesis Testing")
    alpha = st.slider("Significance level (α):", 0.01, 0.10, 0.05, 0.01)
    test_type = st.selectbox(
        "Choose a test:",
        [
            "One-sample t-test",
            "Two-sample t-test (independent)",
            "Paired t-test",
            "Chi-square test of independence",
            "One-way ANOVA",
            "Mann-Whitney U test",
            "Wilcoxon signed-rank test",
        ],
    )

    result_row = None

    if test_type == "One-sample t-test":
        if numeric_cols:
            col = st.selectbox("Numeric column:", numeric_cols, key="t1_col")
            mu = st.number_input("Hypothesized mean (H0):", value=float(df[col].mean()))
            if st.button("Run test"):
                stat, p = sps.ttest_1samp(df[col].dropna(), mu)
                st.write(f"**t-statistic:** {stat:.4f}")
                st.info(interpret_p(p, alpha))
                result_row = {"Test": "One-sample t-test", "Detail": f"{col} vs μ={mu}", "Statistic": round(stat, 4), "p-value": round(p, 4)}
        else:
            st.info("No numeric columns available.")

    elif test_type == "Two-sample t-test (independent)":
        if numeric_cols and categorical_cols:
            c1, c2 = st.columns(2)
            num_col = c1.selectbox("Numeric column:", numeric_cols, key="t2_num")
            grp_col = c2.selectbox("Grouping column:", categorical_cols, key="t2_grp")
            groups = df[grp_col].dropna().unique().tolist()
            chosen = st.multiselect("Pick exactly 2 groups:", groups, default=groups[:2])
            if st.button("Run test", key="t2_run"):
                if len(chosen) != 2:
                    st.error("Please select exactly 2 groups.")
                else:
                    g1 = df[df[grp_col] == chosen[0]][num_col].dropna()
                    g2 = df[df[grp_col] == chosen[1]][num_col].dropna()
                    stat, p = sps.ttest_ind(g1, g2, equal_var=False)
                    st.write(f"**t-statistic:** {stat:.4f}  |  {chosen[0]} mean={g1.mean():.3f}, {chosen[1]} mean={g2.mean():.3f}")
                    st.info(interpret_p(p, alpha))
                    result_row = {"Test": "Two-sample t-test", "Detail": f"{num_col} by {grp_col} ({chosen[0]} vs {chosen[1]})", "Statistic": round(stat, 4), "p-value": round(p, 4)}
        else:
            st.info("Need at least one numeric and one categorical column.")

    elif test_type == "Paired t-test":
        if len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            col_a = c1.selectbox("Column A:", numeric_cols, key="pt_a")
            col_b = c2.selectbox("Column B:", [c for c in numeric_cols if c != col_a], key="pt_b")
            if st.button("Run test", key="pt_run"):
                paired = df[[col_a, col_b]].dropna()
                stat, p = sps.ttest_rel(paired[col_a], paired[col_b])
                st.write(f"**t-statistic:** {stat:.4f}  |  mean diff = {(paired[col_a]-paired[col_b]).mean():.3f}")
                st.info(interpret_p(p, alpha))
                result_row = {"Test": "Paired t-test", "Detail": f"{col_a} vs {col_b}", "Statistic": round(stat, 4), "p-value": round(p, 4)}
        else:
            st.info("Need at least two numeric columns.")

    elif test_type == "Chi-square test of independence":
        if len(categorical_cols) >= 2:
            c1, c2 = st.columns(2)
            col_a = c1.selectbox("Column A:", categorical_cols, key="chi_a")
            col_b = c2.selectbox("Column B:", [c for c in categorical_cols if c != col_a], key="chi_b")
            if st.button("Run test", key="chi_run"):
                table = pd.crosstab(df[col_a], df[col_b])
                stat, p, dof, _ = sps.chi2_contingency(table)
                st.write(f"**Chi² statistic:** {stat:.4f}  |  df = {dof}")
                st.dataframe(table, use_container_width=True)
                st.info(interpret_p(p, alpha))
                result_row = {"Test": "Chi-square", "Detail": f"{col_a} × {col_b}", "Statistic": round(stat, 4), "p-value": round(p, 4)}
        else:
            st.info("Need at least two categorical columns.")

    elif test_type == "One-way ANOVA":
        if numeric_cols and categorical_cols:
            c1, c2 = st.columns(2)
            num_col = c1.selectbox("Numeric column:", numeric_cols, key="an_num")
            grp_col = c2.selectbox("Grouping column:", categorical_cols, key="an_grp")
            if st.button("Run test", key="an_run"):
                groups = [g[num_col].dropna().values for _, g in df.groupby(grp_col) if len(g) > 0]
                if len(groups) < 2:
                    st.error("Need at least 2 groups with data.")
                else:
                    stat, p = sps.f_oneway(*groups)
                    st.write(f"**F-statistic:** {stat:.4f}")
                    st.info(interpret_p(p, alpha))
                    result_row = {"Test": "One-way ANOVA", "Detail": f"{num_col} by {grp_col}", "Statistic": round(stat, 4), "p-value": round(p, 4)}
        else:
            st.info("Need at least one numeric and one categorical column.")

    elif test_type == "Mann-Whitney U test":
        if numeric_cols and categorical_cols:
            c1, c2 = st.columns(2)
            num_col = c1.selectbox("Numeric column:", numeric_cols, key="mw_num")
            grp_col = c2.selectbox("Grouping column:", categorical_cols, key="mw_grp")
            groups = df[grp_col].dropna().unique().tolist()
            chosen = st.multiselect("Pick exactly 2 groups:", groups, default=groups[:2], key="mw_groups")
            if st.button("Run test", key="mw_run"):
                if len(chosen) != 2:
                    st.error("Please select exactly 2 groups.")
                else:
                    g1 = df[df[grp_col] == chosen[0]][num_col].dropna()
                    g2 = df[df[grp_col] == chosen[1]][num_col].dropna()
                    stat, p = sps.mannwhitneyu(g1, g2, alternative="two-sided")
                    st.write(f"**U-statistic:** {stat:.4f}")
                    st.info(interpret_p(p, alpha))
                    result_row = {"Test": "Mann-Whitney U", "Detail": f"{num_col} by {grp_col}", "Statistic": round(stat, 4), "p-value": round(p, 4)}
        else:
            st.info("Need at least one numeric and one categorical column.")

    elif test_type == "Wilcoxon signed-rank test":
        if len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            col_a = c1.selectbox("Column A:", numeric_cols, key="wc_a")
            col_b = c2.selectbox("Column B:", [c for c in numeric_cols if c != col_a], key="wc_b")
            if st.button("Run test", key="wc_run"):
                paired = df[[col_a, col_b]].dropna()
                stat, p = sps.wilcoxon(paired[col_a], paired[col_b])
                st.write(f"**W-statistic:** {stat:.4f}")
                st.info(interpret_p(p, alpha))
                result_row = {"Test": "Wilcoxon signed-rank", "Detail": f"{col_a} vs {col_b}", "Statistic": round(stat, 4), "p-value": round(p, 4)}
        else:
            st.info("Need at least two numeric columns.")

    if result_row is not None:
        st.session_state.test_results.append(result_row)

    # -------------------- Regression (statsmodels: coefficients, CI, p-values) ----
    st.markdown("---")
    st.subheader("Regression Analysis (with confidence intervals)")
    reg_type = st.radio("Type:", ["Linear / Multiple Regression", "Logistic Regression"], horizontal=True)

    if reg_type == "Linear / Multiple Regression":
        if numeric_cols:
            target = st.selectbox("Target (Y):", numeric_cols, key="lr_target")
            features = st.multiselect("Predictors (X):", [c for c in numeric_cols if c != target], key="lr_features")
            if st.button("Fit model", key="lr_fit") and features:
                import statsmodels.api as sm
                data = df[[target] + features].dropna()
                X = sm.add_constant(data[features])
                model = sm.OLS(data[target], X).fit()
                summary_df = pd.DataFrame({
                    "Coefficient": model.params.round(4),
                    "Std Error": model.bse.round(4),
                    "t-value": model.tvalues.round(4),
                    "p-value": model.pvalues.round(4),
                    "CI 2.5%": model.conf_int()[0].round(4),
                    "CI 97.5%": model.conf_int()[1].round(4),
                })
                st.dataframe(summary_df, use_container_width=True)
                st.write(f"**R² = {model.rsquared:.4f}** | Adjusted R² = {model.rsquared_adj:.4f}")
                st.session_state.regression_results = {
                    "type": "Linear Regression", "target": target, "features": features,
                    "r_squared": round(model.rsquared, 4), "summary": summary_df,
                }
        else:
            st.info("No numeric columns available.")

    else:  # Logistic Regression
        binary_targets = [c for c in df.columns if df[c].nunique() == 2]
        if binary_targets and numeric_cols:
            target = st.selectbox("Target (binary):", binary_targets, key="log_target")
            features = st.multiselect("Predictors (X):", [c for c in numeric_cols if c != target], key="log_features")
            if st.button("Fit model", key="log_fit") and features:
                import statsmodels.api as sm
                data = df[[target] + features].dropna()
                y = pd.Categorical(data[target]).codes
                X = sm.add_constant(data[features])
                model = sm.Logit(y, X).fit(disp=0)
                summary_df = pd.DataFrame({
                    "Coefficient": model.params.round(4),
                    "Std Error": model.bse.round(4),
                    "z-value": model.tvalues.round(4),
                    "p-value": model.pvalues.round(4),
                    "CI 2.5%": model.conf_int()[0].round(4),
                    "CI 97.5%": model.conf_int()[1].round(4),
                })
                st.dataframe(summary_df, use_container_width=True)
                st.write(f"**Pseudo R² = {model.prsquared:.4f}**")
                st.session_state.regression_results = {
                    "type": "Logistic Regression", "target": target, "features": features,
                    "pseudo_r_squared": round(model.prsquared, 4), "summary": summary_df,
                }
        else:
            st.info("Need a binary (2-class) target column and numeric predictors.")


# --------------------------------------------------------------------------- #
# TAB 5 — MODEL (automated ML)
# --------------------------------------------------------------------------- #
def guess_target_and_type(df: pd.DataFrame):
    """Heuristically suggest a target column and problem type."""
    numeric_cols, categorical_cols, _ = split_columns(df)
    keywords = ["target", "label", "churn", "success", "outcome", "class", "fraud",
                "default", "converted", "survived", "sales", "price", "revenue"]
    for kw in keywords:
        for col in df.columns:
            if kw in col.lower():
                if df[col].nunique() <= 10:
                    return col, "Classification"
                if col in numeric_cols:
                    return col, "Regression"
    for col in numeric_cols + categorical_cols:
        if 2 <= df[col].nunique() <= 10:
            return col, "Classification"
    return (numeric_cols[0], "Regression") if numeric_cols else (df.columns[0], "Classification")


def build_preprocessor(numeric_features, categorical_features, scale: bool):
    """Build a sklearn ColumnTransformer: impute (+ optional scale) + one-hot encode."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipe, categorical_features))

    return ColumnTransformer(transformers)


def get_regression_models(tune_rf: bool):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.tree import DecisionTreeRegressor

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200, max_depth=12),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    return models


def get_classification_models(tune_rf: bool):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200, max_depth=12),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
    }
    return models


def render_model_tab() -> None:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                  mean_absolute_error, mean_absolute_percentage_error,
                                  precision_score, r2_score, recall_score, roc_auc_score,
                                  roc_curve, root_mean_squared_error)
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder

    df = st.session_state.df
    numeric_cols, categorical_cols, datetime_cols = split_columns(df)

    if df.shape[0] < 20:
        st.warning("Dataset is quite small (<20 rows) — model results may not be reliable.")

    suggested_target, suggested_type = guess_target_and_type(df)
    st.subheader("1. Choose your target")
    target = st.selectbox(
        "Target column (what you want to predict):",
        df.columns.tolist(),
        index=df.columns.tolist().index(suggested_target),
    )
    problem_type = st.radio(
        "Problem type:", ["Classification", "Regression"],
        index=0 if suggested_type == "Classification" else 1,
        horizontal=True,
    )
    st.caption(f"💡 Auto-suggested: **{suggested_target}** as **{suggested_type}** target.")

    st.subheader("2. Choose your features")
    default_features = [c for c in (numeric_cols + categorical_cols) if c != target and c not in datetime_cols]
    features = st.multiselect("Predictor columns:", default_features, default=default_features)

    if not features:
        st.info("Select at least one feature to continue.")
        return

    feat_numeric = [c for c in features if c in numeric_cols]
    feat_categorical = [c for c in features if c in categorical_cols]

    st.subheader("3. Training options")
    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
    scale = c2.checkbox("Standard-scale numeric features", value=(problem_type == "Regression"))
    stratify_opt = c3.checkbox("Stratified split", value=(problem_type == "Classification"), disabled=(problem_type == "Regression"))

    if problem_type == "Classification":
        algo_options = list(get_classification_models(False).keys())
    else:
        algo_options = list(get_regression_models(False).keys())
    algos = st.multiselect("Algorithms to compare:", algo_options, default=algo_options)
    tune_rf = st.checkbox("Also tune Random Forest with grid search (slower)")

    if st.button("🚀 Train & Compare Models", disabled=not algos):
        data = df[[target] + features].dropna()
        X = data[features]
        y_raw = data[target]

        label_encoder = None
        if problem_type == "Classification":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_raw.astype(str))
            class_names = label_encoder.classes_
        else:
            y = y_raw.values
            class_names = None

        strat = y if (stratify_opt and problem_type == "Classification") else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=strat
        )

        preprocessor = build_preprocessor(feat_numeric, feat_categorical, scale)
        models = get_classification_models(tune_rf) if problem_type == "Classification" else get_regression_models(tune_rf)

        rows = []
        details = {}
        progress = st.progress(0.0, text="Training models...")

        for i, name in enumerate(algos):
            model = models[name]
            pipe = Pipeline([("prep", preprocessor), ("model", model)])

            if tune_rf and name == "Random Forest":
                if problem_type == "Classification":
                    grid = {"model__n_estimators": [100, 200], "model__max_depth": [8, 12, None]}
                else:
                    grid = {"model__n_estimators": [100, 200], "model__max_depth": [8, 12, None]}
                search = GridSearchCV(pipe, grid, cv=3, n_jobs=-1)
                search.fit(X_train, y_train)
                pipe = search.best_estimator_
            else:
                pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)

            if problem_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                auc = np.nan
                try:
                    if hasattr(pipe, "predict_proba"):
                        proba = pipe.predict_proba(X_test)
                        if len(class_names) == 2:
                            auc = roc_auc_score(y_test, proba[:, 1])
                        else:
                            auc = roc_auc_score(y_test, proba, multi_class="ovr")
                except Exception:
                    pass
                rows.append({"Model": name, "Accuracy": round(acc, 4), "Precision": round(prec, 4),
                             "Recall": round(rec, 4), "F1-Score": round(f1, 4),
                             "ROC-AUC": round(auc, 4) if not np.isnan(auc) else "—"})
            else:
                r2 = r2_score(y_test, y_pred)
                rmse = root_mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                try:
                    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                except Exception:
                    mape = np.nan
                rows.append({"Model": name, "R²": round(r2, 4), "RMSE": round(rmse, 4),
                             "MAE": round(mae, 4), "MAPE %": round(mape, 2) if not np.isnan(mape) else "—"})

            details[name] = {"pipeline": pipe, "X_test": X_test, "y_test": y_test, "y_pred": y_pred}
            progress.progress((i + 1) / len(algos), text=f"Trained {name}")

        progress.empty()
        comparison = pd.DataFrame(rows)
        sort_col = "F1-Score" if problem_type == "Classification" else "R²"
        comparison = comparison.sort_values(sort_col, ascending=False).reset_index(drop=True)

        st.session_state.model_comparison = comparison
        st.session_state.model_details = details
        st.session_state.model_meta = {
            "problem_type": problem_type, "target": target, "features": features,
            "class_names": list(class_names) if class_names is not None else None,
        }
        st.rerun()

    # -------------------- Show results if available -----------------------------
    if st.session_state.model_comparison is not None:
        st.markdown("---")
        st.subheader("Model Comparison")
        st.dataframe(st.session_state.model_comparison, use_container_width=True, hide_index=True)
        best_model_name = st.session_state.model_comparison.iloc[0]["Model"]
        st.success(f"🏆 Best model: **{best_model_name}**")

        meta = st.session_state.model_meta
        inspect_model = st.selectbox("Inspect a model:", st.session_state.model_comparison["Model"].tolist())
        detail = st.session_state.model_details[inspect_model]
        pipe, X_test, y_test, y_pred = detail["pipeline"], detail["X_test"], detail["y_test"], detail["y_pred"]

        if meta["problem_type"] == "Classification":
            cm = confusion_matrix(y_test, y_pred)
            labels = meta["class_names"]
            fig = px.imshow(cm, text_auto=True, x=labels, y=labels,
                             labels=dict(x="Predicted", y="Actual", color="Count"),
                             title=f"Confusion Matrix — {inspect_model}")
            st.plotly_chart(fig, use_container_width=True)

            if len(labels) == 2 and hasattr(pipe, "predict_proba"):
                try:
                    proba_test = pipe.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, proba_test)
                    auc_val = roc_auc_score(y_test, proba_test)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_val:.3f})"))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
                    fig_roc.update_layout(title=f"ROC Curve — {inspect_model}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                    st.plotly_chart(fig_roc, use_container_width=True)
                except Exception:
                    pass

        else:
            fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"},
                              title=f"Actual vs Predicted — {inspect_model}")
            min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(dash="dash", color="gray"))
            st.plotly_chart(fig, use_container_width=True)

        model_obj = pipe.named_steps.get("model")
        if hasattr(model_obj, "feature_importances_"):
            try:
                feature_names = pipe.named_steps["prep"].get_feature_names_out()
                importances = pd.Series(model_obj.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
                fig = px.bar(x=importances.values, y=importances.index, orientation="h",
                             title=f"Feature Importance — {inspect_model}", labels={"x": "Importance", "y": "Feature"})
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# TAB 6 — REPORT
# --------------------------------------------------------------------------- #
def df_to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    """
    Convert a DataFrame to a Markdown table without depending on the
    optional `tabulate` package (which may not be installed in all
    hosting environments).
    """
    work = df.reset_index() if index else df.copy()
    headers = [str(c) for c in work.columns]
    rows = work.astype(str).values.tolist()
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + body_lines)


def build_markdown_report() -> str:
    """Compile everything done in this session into a Markdown report."""
    df = st.session_state.df
    numeric_cols, categorical_cols, datetime_cols = split_columns(df)
    lines = []
    lines.append(f"# Data Analysis Report")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · Source file: "
                  f"{st.session_state.file_name or 'uploaded dataset'}*\n")

    lines.append("## 1. Dataset Overview")
    lines.append(f"- **Rows:** {df.shape[0]:,}  \n- **Columns:** {df.shape[1]:,}")
    lines.append(f"- **Numeric columns:** {', '.join(numeric_cols) or 'none'}")
    lines.append(f"- **Categorical columns:** {', '.join(categorical_cols) or 'none'}")
    lines.append(f"- **Date/time columns:** {', '.join(datetime_cols) or 'none'}\n")

    lines.append("## 2. Cleaning Steps Applied")
    if st.session_state.cleaning_log:
        for entry in st.session_state.cleaning_log:
            lines.append(f"- {entry}")
    else:
        lines.append("*No cleaning steps were applied in this session.*")
    lines.append("")

    if numeric_cols:
        lines.append("## 3. Descriptive Statistics (numeric columns)")
        desc = df[numeric_cols].describe().T.round(3)
        lines.append(df_to_markdown(desc, index=True))
        lines.append("")

    lines.append("## 4. Hypothesis Test Results")
    if st.session_state.test_results:
        test_df = pd.DataFrame(st.session_state.test_results)
        lines.append(df_to_markdown(test_df, index=False))
    else:
        lines.append("*No hypothesis tests were run in this session.*")
    lines.append("")

    lines.append("## 5. Regression Analysis")
    reg = st.session_state.regression_results
    if reg:
        lines.append(f"**{reg['type']}** — Target: `{reg['target']}` | Predictors: {', '.join(reg['features'])}")
        fit_stat = reg.get("r_squared", reg.get("pseudo_r_squared"))
        lines.append(f"Model fit: **{fit_stat}**\n")
        lines.append(df_to_markdown(reg["summary"], index=True))
    else:
        lines.append("*No regression model was fit in this session.*")
    lines.append("")

    lines.append("## 6. Machine Learning Model Comparison")
    if st.session_state.model_comparison is not None:
        meta = st.session_state.model_meta
        lines.append(f"**Problem type:** {meta['problem_type']} | **Target:** `{meta['target']}` | "
                      f"**Features:** {', '.join(meta['features'])}\n")
        lines.append(df_to_markdown(st.session_state.model_comparison, index=False))
        best = st.session_state.model_comparison.iloc[0]["Model"]
        lines.append(f"\n**Best performing model: {best}**")
    else:
        lines.append("*No models were trained in this session.*")
    lines.append("")

    lines.append("## 7. Recommended Next Steps")
    lines.append(
        "- Validate the best model with cross-validation before deploying it.\n"
        "- Re-check any columns flagged for high missingness or skew during cleaning.\n"
        "- Consider engineering additional features from date columns (day/month/weekday) if not already done.\n"
        "- Share this report alongside the underlying dataset for reproducibility."
    )

    return "\n".join(lines)


def render_report_tab() -> None:
    st.subheader("📄 Session Report")
    st.caption(
        "This report compiles the cleaning steps, statistics, tests, and models "
        "you've run in this session. Generate it, then download as Markdown or HTML."
    )

    if st.button("🔄 Generate / Refresh Report"):
        st.session_state["report_md"] = build_markdown_report()

    report_md = st.session_state.get("report_md")
    if report_md:
        st.markdown("---")
        st.markdown(report_md)
        st.markdown("---")

        c1, c2 = st.columns(2)
        c1.download_button(
            "⬇️ Download as Markdown (.md)",
            data=report_md,
            file_name="analysis_report.md",
            mime="text/markdown",
        )
        try:
            import markdown as md_lib
            html_body = md_lib.markdown(report_md, extensions=["tables"])
            html_full = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Data Analysis Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background: #f5f5f5; }}
h1, h2 {{ border-bottom: 2px solid #eee; padding-bottom: 6px; }}
</style></head><body>{html_body}</body></html>"""
            c2.download_button(
                "⬇️ Download as HTML (.html)",
                data=html_full,
                file_name="analysis_report.html",
                mime="text/html",
            )
        except ImportError:
            c2.caption("Install the `markdown` package to enable HTML export.")
    else:
        st.info("Click **Generate / Refresh Report** to build the report from your session.")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main() -> None:
    init_state()

    st.title("📊 Auto Data Analyst PRO")
    st.caption(
        "Upload → Clean → Explore → Test → Model → Report. "
        "A full analyst workflow in one app."
    )

    with st.sidebar:
        st.header("⚙️ Dataset")
        uploaded = st.file_uploader(
            "Upload your data",
            type=["csv", "xlsx", "xls", "json", "parquet", "tsv", "txt"],
            help="Supported: CSV, Excel, JSON, Parquet, TSV/TXT",
        )
        if uploaded is not None and uploaded.name != st.session_state.file_name:
            try:
                parsed = load_data(uploaded.getvalue(), uploaded.name)
                if parsed.empty:
                    st.error("The file parsed but contains no rows.")
                else:
                    st.session_state.raw_df = parsed
                    st.session_state.df = parsed.copy()
                    st.session_state.history = []
                    st.session_state.cleaning_log = []
                    st.session_state.test_results = []
                    st.session_state.regression_results = None
                    st.session_state.model_comparison = None
                    st.session_state.model_details = {}
                    st.session_state.model_meta = None
                    st.session_state.report_md = None
                    st.session_state.file_name = uploaded.name
                    st.success(f"Loaded **{uploaded.name}** — {parsed.shape[0]:,} rows × {parsed.shape[1]:,} cols")
            except Exception as exc:
                st.error(f"❌ Could not read **{uploaded.name}**.\n\nDetails: `{exc}`")

        if st.session_state.df is not None:
            st.markdown("---")
            st.caption(f"Current file: **{st.session_state.file_name}**")
            st.caption(f"Shape: {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]:,} cols")
            if st.session_state.history:
                st.caption(f"↩️ {len(st.session_state.history)} undo step(s) available")

    if st.session_state.df is None:
        st.info("👈 Upload a CSV, Excel, JSON, Parquet, or TSV/TXT file from the sidebar to begin.")
        return

    tabs = st.tabs([
        "📋 Upload & Preview", "🧹 Clean", "🔎 Explore", "📈 Statistics", "🤖 Model", "📄 Report",
    ])
    with tabs[0]:
        render_upload_tab()
    with tabs[1]:
        render_clean_tab()
    with tabs[2]:
        render_explore_tab()
    with tabs[3]:
        render_stats_tab()
    with tabs[4]:
        render_model_tab()
    with tabs[5]:
        render_report_tab()


if __name__ == "__main__":
    main()
