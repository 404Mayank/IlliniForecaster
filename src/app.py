import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


DATA_URL = "https://waf.cs.illinois.edu/discovery/gpa.csv"
TERM_OPTIONS = ["Fall", "Spring", "Summer", "Winter"]
GRADE_WEIGHTS = {
    "A+": 4.0,
    "A": 4.0,
    "A-": 3.67,
    "B+": 3.33,
    "B": 3.0,
    "B-": 2.67,
    "C+": 2.33,
    "C": 2.0,
    "C-": 1.67,
    "D+": 1.33,
    "D": 1.0,
    "D-": 0.67,
    "F": 0.0,
}


st.set_page_config(page_title="Course GPA Predictor", layout="wide")

# Suppress the known LightGBM/sklearn warning globally for this app run.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def compute_target_average_gpa(df: pd.DataFrame) -> pd.Series:
    weighted_points = pd.Series(0.0, index=df.index)
    graded_students = pd.Series(0.0, index=df.index)
    for grade_col, weight in GRADE_WEIGHTS.items():
        weighted_points += df[grade_col] * weight
        graded_students += df[grade_col]
    return _safe_divide(weighted_points, graded_students)


@st.cache_resource
def load_pipeline():
    return joblib.load("artifacts/best_model.joblib")


@st.cache_data(show_spinner=False)
def build_lookup_tables(data_url: str):
    raw = pd.read_csv(data_url)
    raw["avg_gpa"] = compute_target_average_gpa(raw)
    raw = raw.dropna(subset=["avg_gpa", "Subject", "Primary Instructor"]).copy()

    subject_lookup = (
        raw.groupby("Subject", as_index=False)["avg_gpa"]
        .mean()
        .sort_values("Subject")
    )

    instructor_lookup = (
        raw.groupby("Primary Instructor", as_index=False)
        .agg(instructor_hist_gpa=("avg_gpa", "mean"), instructor_sections=("avg_gpa", "count"))
        .sort_values(["instructor_sections", "Primary Instructor"], ascending=[False, True])
    )

    subject_instructor_lookup = (
        raw.groupby(["Subject", "Primary Instructor"], as_index=False)
        .agg(subject_sections=("avg_gpa", "count"))
        .sort_values(["Subject", "subject_sections", "Primary Instructor"], ascending=[True, False, True])
    )

    subject_map = dict(zip(subject_lookup["Subject"], subject_lookup["avg_gpa"]))
    instructor_map = dict(
        zip(instructor_lookup["Primary Instructor"], instructor_lookup["instructor_hist_gpa"])
    )
    section_count_map = dict(
        zip(instructor_lookup["Primary Instructor"], instructor_lookup["instructor_sections"])
    )
    return subject_map, instructor_lookup, instructor_map, section_count_map, subject_instructor_lookup


def infer_chart_mode(mode_choice: str) -> str:
    if mode_choice in {"Light", "Dark"}:
        return mode_choice.lower()
    
    # "Auto" mode: since st.get_option("theme.base") doesn't always reflect 
    # the user's system dark mode, we default to "dark" for better contrast
    # in most Streamlit environments, unless explicitly set to 'light' in config.
    configured_theme = st.get_option("theme.base")
    if configured_theme == "light":
        return "light"
    return "dark"


def chart_palette(mode: str) -> dict[str, str]:
    if mode == "dark":
        return {
            "text": "#ffffff",  # Brighter white for better visibility
            "grid": "#707070",
            "pos": "#7fb09b",
            "neg": "#c08a75",
        }
    return {
        "text": "#000000",    # Solid black in light mode
        "grid": "#d0d5da",
        "pos": "#4f8f77",
        "neg": "#b16e59",
    }


def prettify_feature_name(raw_name: str) -> str:
    cleaned = raw_name.replace("num__", "").replace("cat__", "")
    if cleaned.startswith("Term_"):
        return "Term: " + cleaned.replace("Term_", "")
    rename_map = {
        "course_level": "Course Level",
        "class_size": "Class Size",
        "subject_hist_gpa": "Subject Historical GPA",
        "instructor_hist_gpa": "Instructor Historical GPA",
    }
    return rename_map.get(cleaned, cleaned)


def transformed_frame(preprocessor, input_row: pd.DataFrame) -> pd.DataFrame:
    transformed = preprocessor.transform(input_row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]
    return pd.DataFrame(transformed, columns=feature_names, index=input_row.index)


try:
    pipeline = load_pipeline()
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
except Exception:
    st.error("Could not load artifacts/best_model.joblib. Run the training script first.")
    st.stop()

subject_map, instructor_lookup_df, instructor_map, section_count_map, subject_instructor_df = build_lookup_tables(DATA_URL)

st.title("Course GPA Predictor")
st.caption("Compact prediction UI with instructor lookup and SHAP explainability.")

# Inject CSS to change primary colors from Streamlit pink to a nice blue
st.markdown(
    """
    <style>
    /* Targeted accent updates without forcing a full custom Streamlit theme */
    div.stSlider > div[data-baseweb="slider"] div[role="slider"] {
        background-color: #3b82f6 !important;
        border-color: #3b82f6 !important;
        outline-color: #3b82f6 !important;
    }

    /* Recolor the generated slider fill gradient from Streamlit red to blue. */
    div.stSlider > div[data-baseweb="slider"] div[style*="height: 0.25rem"] {
        filter: hue-rotate(165deg) saturate(150%);
    }

    /* Remove residual red on the thumb value badge and hover text. */
    div.stSlider div[data-testid="stSliderThumbValue"],
    div.stSlider div[data-testid="stSliderThumbValue"] *,
    div.stSlider div[data-testid="stMarkdownContainer"] p {
        color: #3b82f6 !important;
        border-color: #3b82f6 !important;
        outline-color: #3b82f6 !important;
    }

    div.stSlider > div[data-baseweb="slider"] div[data-testid="stTickBar"] > div {
        background-color: #3b82f6 !important;
    }

    .stCheckbox > label > div[data-baseweb="checkbox"] > div {
        background-color: #3b82f6 !important;
        border-color: #3b82f6 !important;
    }
    /* Minimal hack to reduce monotony without removing native theme switcher */
    </style>
    """,
    unsafe_allow_html=True
)

left, right = st.columns([1.0, 1.35], gap="large")

with left:
    st.subheader("Inputs")

    subject_options = sorted(subject_map.keys())
    default_subject = "CS" if "CS" in subject_options else subject_options[0]
    subject = st.selectbox(
        "Subject",
        options=subject_options,
        index=subject_options.index(default_subject),
        help="Department code. This is used to derive the engineered feature 'Subject Hist GPA' (historical average of the department).",
    )

    instructor_candidates = subject_instructor_df.loc[
        subject_instructor_df["Subject"] == subject, "Primary Instructor"
    ].tolist()
    if not instructor_candidates:
        instructor_candidates = instructor_lookup_df["Primary Instructor"].head(300).tolist()

    instructor = st.selectbox(
        "Instructor Name",
        options=instructor_candidates,
        help="Selecting an instructor automatically derives the engineered feature 'Instructor Hist GPA' based on their past grading history.",
    )

    term = st.selectbox(
        "Term",
        options=TERM_OPTIONS,
        index=0,
        help="Seasonal term indicator used by the model.",
    )

    course_level = st.select_slider(
        "Course Level",
        options=[100, 200, 300, 400, 500],
        value=400,
        help="Higher levels are usually more advanced.",
    )

    class_size = st.slider(
        "Class Size",
        min_value=10,
        max_value=500,
        value=150,
        step=10,
        help="Total students in the section.",
    )

    theme_choice = st.selectbox(
        "Chart Theme",
        options=["Dark", "Light", "Auto"],
        index=0,
        help="Select Dark or Light to force the chart text color.",
    )

    subject_hist_gpa = float(subject_map.get(subject, np.nan))
    instructor_hist_gpa = float(instructor_map.get(instructor, np.nan))
    instructor_sections = int(section_count_map.get(instructor, 0))

    info_col1, info_col2 = st.columns(2)
    info_col1.metric("Subject Hist GPA", f"{subject_hist_gpa:.3f}")
    info_col2.metric("Instructor Hist GPA", f"{instructor_hist_gpa:.3f}")
    st.caption(f"Instructor history based on {instructor_sections} prior sections.")

input_df = pd.DataFrame(
    [
        {
            "course_level": course_level,
            "class_size": class_size,
            "subject_hist_gpa": subject_hist_gpa,
            "instructor_hist_gpa": instructor_hist_gpa,
            "Term": term,
        }
    ]
)

transformed_df = transformed_frame(preprocessor, input_df)
feature_names = transformed_df.columns.tolist()

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    prediction = float(model.predict(transformed_df)[0])

baseline_value = np.nan
plot_df = pd.DataFrame(columns=["feature", "shap_value"])
shap_error = None

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        explainer = shap.TreeExplainer(model)
        shap_raw = explainer.shap_values(transformed_df)

    expected_value = explainer.expected_value
    expected_arr = np.asarray(expected_value).reshape(-1)
    if expected_arr.size > 0:
        baseline_value = float(expected_arr[0])

    if isinstance(shap_raw, list):
        shap_raw = shap_raw[0]
    shap_values = np.asarray(shap_raw)[0]

    plot_df = pd.DataFrame(
        {
            "feature": [prettify_feature_name(name) for name in feature_names],
            "shap_value": shap_values,
        }
    )

    plot_df = plot_df.loc[plot_df["shap_value"].abs() > 1e-10].copy()
    plot_df = plot_df.reindex(plot_df["shap_value"].abs().sort_values(ascending=False).index)
    plot_df = plot_df.head(10).iloc[::-1]
except Exception as exc:
    shap_error = str(exc)

with right:
    st.subheader("Prediction")
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Predicted Avg GPA", f"{prediction:.3f}")
    if np.isnan(baseline_value):
        metric_col2.metric("Model Baseline", "N/A")
    else:
        metric_col2.metric("Model Baseline", f"{baseline_value:.3f}")

    st.subheader("SHAP Explainability")
    st.caption("Bars show how each feature pushes the prediction up or down from the baseline.")

    if shap_error:
        st.warning(f"Could not generate SHAP chart for this model: {shap_error}")
    elif plot_df.empty:
        st.info("No non-zero SHAP contributions were returned for this input.")
    else:
        chart_mode = infer_chart_mode(theme_choice)
        palette = chart_palette(chart_mode)

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

        colors = [palette["pos"] if val >= 0 else palette["neg"] for val in plot_df["shap_value"]]
        ax.barh(plot_df["feature"], plot_df["shap_value"], color=colors, height=0.62)

        ax.axvline(0, linewidth=1.0, color=palette["grid"])
        ax.grid(axis="x", linestyle="--", linewidth=0.7, color=palette["grid"], alpha=0.6)

        ax.set_xlabel("Contribution to predicted GPA", color=palette["text"])
        ax.tick_params(axis="x", colors=palette["text"])
        ax.tick_params(axis="y", colors=palette["text"])
        for spine in ax.spines.values():
            spine.set_visible(False)

        for y_pos, value in enumerate(plot_df["shap_value"]):
            text = f"{value:+.3f}"
            x_pos = value + (0.006 if value >= 0 else -0.006)
            ax.text(
                x_pos,
                y_pos,
                text,
                va="center",
                ha="left" if value >= 0 else "right",
                color=palette["text"],
                fontsize=9,
            )

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True, transparent=True)