from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_FILE = "student_dataset_10,000_rows.csv"
TARGET_COLUMN = "placement_status"
FEATURE_COLUMNS = [
    "study_hours",
    "attendance",
    "sleep_hours",
    "internet_usage",
    "assignments_completed",
    "previous_score",
    "exam_score",
]

PLOT_LAYOUT = {
    "template": "plotly_white",
    "paper_bgcolor": "#ffffff",
    "plot_bgcolor": "#ffffff",
    "font": {"color": "#111827", "size": 13},
    "title_font": {"color": "#111827", "size": 20},
    "xaxis": {"gridcolor": "#e5e7eb", "linecolor": "#9ca3af", "zerolinecolor": "#d1d5db"},
    "yaxis": {"gridcolor": "#e5e7eb", "linecolor": "#9ca3af", "zerolinecolor": "#d1d5db"},
}


st.set_page_config(
    page_title="Student Placement Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            color-scheme: light !important;
        }
        :root {
            --app-bg: #f4f7fb;
            --panel-bg: #ffffff;
            --text-main: #111827;
            --text-muted: #4b5563;
            --border: #d7dee8;
            --accent: #2563eb;
            --accent-soft: #eff6ff;
            --success: #166534;
            --danger: #b91c1c;
        }
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stHeader"] {
            background: var(--app-bg);
            color: var(--text-main);
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--border);
        }
        h1, h2, h3, h4, h5, h6, p, label, span,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: var(--text-main);
        }
        [data-testid="stWidgetLabel"] p,
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"],
        div[data-testid="stMetric"] * {
            color: var(--text-main) !important;
            opacity: 1 !important;
        }
        div[data-testid="stMetric"] {
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
        }
        .main-title {
            font-size: 2.25rem;
            font-weight: 800;
            color: var(--text-main);
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: var(--text-muted);
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }
        .dataset-banner {
            background: #e8f5ee;
            border: 1px solid #b8e2c8;
            color: #14532d;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 1rem 0;
            font-weight: 600;
        }
        .result-box {
            color: #ffffff !important;
            border-radius: 8px;
            padding: 1.25rem;
            margin-top: 0.75rem;
        }
        .result-box,
        .result-box *,
        .result-box div,
        .result-box h3 {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            opacity: 1 !important;
        }
        .result-box h3 {
            margin: 0.25rem 0 0;
            font-size: 1.8rem;
        }
        .placed-box {
            background: #14532d;
        }
        .not-placed-box {
            background: #7f1d1d;
        }
        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] div,
        div[data-testid="stFileUploaderDropzone"],
        div[data-testid="stFileUploader"] section,
        div[data-testid="stFileUploader"] div,
        div[data-testid="stNumberInput"] div[data-baseweb="input"] {
            background: #ffffff !important;
            background-color: #ffffff !important;
            border-color: var(--border) !important;
            color: var(--text-main) !important;
        }
        div[data-testid="stFileUploader"] *,
        div[data-testid="stFileUploaderDropzone"] * {
            color: var(--text-main) !important;
            -webkit-text-fill-color: var(--text-main) !important;
            opacity: 1 !important;
        }
        div[data-testid="stFileUploader"] button,
        div[data-testid="stFileUploaderDropzone"] button {
            background: #eff6ff !important;
            border: 1px solid #bfdbfe !important;
            color: #1d4ed8 !important;
            -webkit-text-fill-color: #1d4ed8 !important;
        }
        div[data-baseweb="input"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] svg,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stNumberInput"] div,
        div[data-testid="stNumberInput"] span {
            background-color: #ffffff !important;
            color: var(--text-main) !important;
            -webkit-text-fill-color: var(--text-main) !important;
            fill: var(--text-main) !important;
            opacity: 1 !important;
        }
        div[data-testid="stNumberInput"] button {
            background: #ffffff !important;
            border-color: var(--border) !important;
            color: var(--text-main) !important;
        }
        button[data-baseweb="tab"] p {
            color: var(--text-muted) !important;
            font-weight: 600;
        }
        button[data-baseweb="tab"][aria-selected="true"] p {
            color: var(--accent) !important;
        }
        .table-wrap {
            max-height: 460px;
            overflow: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }
        .light-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
        }
        .light-table th {
            position: sticky;
            top: 0;
            background: #f1f5f9;
            color: #111827;
            text-align: left;
            padding: 0.7rem;
            border-bottom: 1px solid var(--border);
            font-weight: 700;
            z-index: 1;
        }
        .light-table td {
            color: #111827;
            padding: 0.65rem 0.7rem;
            border-bottom: 1px solid #eef2f7;
            background: #ffffff;
        }
        .light-table tr:nth-child(even) td {
            background: #f8fafc;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_column_name(column_name: str) -> str:
    return column_name.strip().lower().replace(" ", "_")


@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def placement_to_number(value) -> int:
    text = str(value).strip().lower().replace(" ", "").replace("_", "")
    if text in {"1", "yes", "true", "placed", "selected"}:
        return 1
    if "not" in text or text in {"0", "no", "false", "unplaced", "rejected"}:
        return 0
    return 1 if text == "placed" else 0


def prepare_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = raw_df.copy()
    df.columns = [clean_column_name(col) for col in df.columns]
    df = df.drop_duplicates()

    missing_columns = [col for col in [*FEATURE_COLUMNS, TARGET_COLUMN] if col not in df.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(placement_to_number)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return df, X, y


@st.cache_resource(show_spinner=False)
def train_model(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    return model, scaler, X_test, y_test, y_pred, y_proba, metrics


def render_light_table(df: pd.DataFrame, max_rows: int | None = None) -> None:
    table_df = df.copy()
    if max_rows is not None:
        table_df = table_df.head(max_rows)

    html = table_df.to_html(index=False, classes="light-table", border=0, escape=True)
    st.markdown(f'<div class="table-wrap">{html}</div>', unsafe_allow_html=True)


def metric_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def status_label(value: int) -> str:
    return "Placed" if value == 1 else "Not Placed"


def build_student_input(df: pd.DataFrame) -> pd.DataFrame:
    inputs = {}

    ranges = {
        "study_hours": (1, 12, 1, "Study Hours"),
        "attendance": (0, 100, 1, "Attendance (%)"),
        "sleep_hours": (1, 12, 1, "Sleep Hours"),
        "internet_usage": (0, 12, 1, "Internet Usage (hours)"),
        "assignments_completed": (0, 25, 1, "Assignments Completed"),
        "previous_score": (0, 100, 1, "Previous Score"),
        "exam_score": (0.0, 100.0, 0.1, "Exam Score"),
    }

    for col in FEATURE_COLUMNS:
        min_default, max_default, step, label = ranges[col]
        min_value = max(min_default, float(df[col].min()))
        max_value = min(max_default, float(df[col].max()))
        value = float(df[col].median())

        if min_value >= max_value:
            min_value, max_value = min_default, max_default

        if isinstance(step, int):
            inputs[col] = st.number_input(
                label,
                min_value=int(min_value),
                max_value=int(max_value),
                value=int(round(value)),
                step=step,
                format="%d",
            )
        else:
            inputs[col] = st.number_input(
                label,
                min_value=float(min_value),
                max_value=float(max_value),
                value=float(round(value, 2)),
                step=step,
            )

    return pd.DataFrame([inputs], columns=FEATURE_COLUMNS)


def confusion_matrix_chart(y_test: pd.Series, y_pred: np.ndarray):
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Students"),
        x=["Not Placed", "Placed"],
        y=["Not Placed", "Placed"],
        title="Confusion Matrix",
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20), **PLOT_LAYOUT)
    return fig


def placement_distribution_chart(df: pd.DataFrame):
    counts = df[TARGET_COLUMN].map(status_label).value_counts().reset_index()
    counts.columns = ["Placement Status", "Students"]
    fig = px.bar(
        counts,
        x="Placement Status",
        y="Students",
        color="Placement Status",
        text="Students",
        title="Placement Status Distribution",
        color_discrete_map={"Placed": "#16a34a", "Not Placed": "#dc2626"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=430, margin=dict(l=20, r=20, t=60, b=20), **PLOT_LAYOUT)
    return fig


def coefficient_chart(model: LogisticRegression):
    coef_df = pd.DataFrame(
        {
            "Feature": [col.replace("_", " ").title() for col in FEATURE_COLUMNS],
            "Impact": model.coef_[0],
        }
    ).sort_values("Impact")
    fig = px.bar(
        coef_df,
        x="Impact",
        y="Feature",
        orientation="h",
        color="Impact",
        color_continuous_scale="RdYlGn",
        title="Feature Impact on Placement Prediction",
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=20), **PLOT_LAYOUT)
    return fig


st.markdown('<div class="main-title">Student Placement Prediction App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Train a Logistic Regression model, check performance, explore graphs, and predict whether a student may get placed.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dataset")
    default_data_path = Path(DATA_FILE)
    if default_data_path.exists():
        selected_local_file = DATA_FILE
        st.caption(f"Default dataset loaded from `{DATA_FILE}`.")
    else:
        selected_local_file = None
        st.caption(f"Place `{DATA_FILE}` in the same folder as this app.")

    uploaded_file = st.file_uploader("Upload another CSV", type=["csv"])

    st.header("Training")
    test_size = st.slider("Test data size", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)


raw_df = None
dataset_source = None

if uploaded_file is not None:
    raw_df = load_csv_from_bytes(uploaded_file.getvalue())
    dataset_source = uploaded_file.name
elif selected_local_file:
    raw_df = load_csv_from_path(selected_local_file)
    dataset_source = selected_local_file

if raw_df is None:
    st.info(f"Add your dataset CSV in this folder as `{DATA_FILE}`, or upload it from the sidebar.")
    st.stop()

st.markdown(
    f'<div class="dataset-banner">Dataset loaded: {dataset_source}</div>',
    unsafe_allow_html=True,
)

try:
    df, X, y = prepare_data(raw_df)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if len(df) < 10:
    st.error("Dataset needs at least 10 valid rows for training and testing.")
    st.stop()

if y.nunique() < 2:
    st.error("Placement status must contain both Placed and Not Placed values.")
    st.stop()

with st.spinner("Training model and preparing dashboard..."):
    model, scaler, X_test, y_test, y_pred, y_proba, metrics = train_model(
        X, y, test_size, int(random_state)
    )

metric_cols = st.columns(4)
metric_cols[0].metric("Accuracy", metric_percent(metrics["accuracy"]))
metric_cols[1].metric("Precision", metric_percent(metrics["precision"]))
metric_cols[2].metric("Recall", metric_percent(metrics["recall"]))
metric_cols[3].metric("F1 Score", metric_percent(metrics["f1"]))

tabs = st.tabs(["Predict", "Model Results", "Graphs", "Dataset"])

with tabs[0]:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Enter Student Details")
        user_input = build_student_input(df)

    with right:
        st.subheader("Placement Result")
        scaled_input = scaler.transform(user_input[FEATURE_COLUMNS])
        prediction = int(model.predict(scaled_input)[0])
        placement_probability = float(model.predict_proba(scaled_input)[0][1])
        box_class = "placed-box" if prediction == 1 else "not-placed-box"

        st.markdown(
            f"""
            <div class="result-box {box_class}">
                <div>Prediction Based on Input</div>
                <h3>{status_label(prediction)}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        col_a, col_b = st.columns(2)
        col_a.metric("Placed Probability", metric_percent(placement_probability))
        col_b.metric("Not Placed Probability", metric_percent(1 - placement_probability))

with tabs[1]:
    st.subheader("Model Performance")
    performance_df = pd.DataFrame(
        [
            {"Metric": "Accuracy", "Score": metric_percent(metrics["accuracy"])},
            {"Metric": "Precision", "Score": metric_percent(metrics["precision"])},
            {"Metric": "Recall", "Score": metric_percent(metrics["recall"])},
            {"Metric": "F1 Score", "Score": metric_percent(metrics["f1"])},
        ]
    )
    render_light_table(performance_df)

    st.plotly_chart(confusion_matrix_chart(y_test, y_pred), use_container_width=True)

with tabs[2]:
    graph_choice = st.selectbox(
        "Select Graph",
        ["Placement Distribution", "Feature Impact", "Score vs Attendance"],
    )

    if graph_choice == "Placement Distribution":
        st.plotly_chart(placement_distribution_chart(df), use_container_width=True)
    elif graph_choice == "Feature Impact":
        st.plotly_chart(coefficient_chart(model), use_container_width=True)
    else:
        scatter_df = df.copy()
        scatter_df["placement_label"] = scatter_df[TARGET_COLUMN].map(status_label)
        fig = px.scatter(
            scatter_df,
            x="attendance",
            y="exam_score",
            color="placement_label",
            color_discrete_map={"Placed": "#16a34a", "Not Placed": "#dc2626"},
            title="Exam Score vs Attendance",
            labels={
                "attendance": "Attendance (%)",
                "exam_score": "Exam Score",
                "placement_label": "Placement Status",
            },
            opacity=0.65,
        )
        fig.update_layout(height=470, margin=dict(l=20, r=20, t=60, b=20), **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("Dataset Preview")
    stat_cols = st.columns(4)
    stat_cols[0].metric("Rows", f"{len(df):,}")
    stat_cols[1].metric("Columns", f"{len(df.columns):,}")
    stat_cols[2].metric("Placed", f"{int(y.sum()):,}")
    stat_cols[3].metric("Not Placed", f"{int((y == 0).sum()):,}")

    preview_df = df.copy()
    preview_df[TARGET_COLUMN] = preview_df[TARGET_COLUMN].map(status_label)
    render_light_table(preview_df, max_rows=100)

    with st.expander("Summary statistics"):
        summary_df = df[FEATURE_COLUMNS].describe().T.reset_index().rename(columns={"index": "feature"})
        render_light_table(summary_df)
