# IlliniForecaster: UIUC GPA Prediction

🚀 **Live App:** [https://illiniforecaster.streamlit.app/](https://illiniforecaster.streamlit.app/)

A machine learning project and web app to predict average course GPAs at the University of Illinois Urbana-Champaign (UIUC). Created for MUJ's ML Subject Project.

## Dataset Credit
This project uses the excellent [UIUC GPA Dataset](https://discovery.cs.illinois.edu/dataset/gpa/) provided by Wade Fagen-Ulmschneider and the UIUC Discovery team. All data comes directly from their repository.

## Overview
The goal is to predict the `avg_gpa` of a course section using past data. GPAs are calculated from standard grade points (`A+` = 4.0 ... `F` = 0.0), ignoring students who withdrew (`W`).

We trained three models:
1. **LightGBM** (Best performer: fast and accurate)
2. **XGBoost** (Strong baseline)
3. **Random Forest** (Stable, reliable baseline)

## How It Works

### The Data Split (80/20)
To make real-world predictions, we split our data by time. The oldest 80% of semesters are used for training, and the newest 20% are kept hidden for testing. This prevents the model from "cheating" by looking into the future.

### Features Used
The models make predictions based on:
1. `course_level`: Extracted from the course number (e.g., 100, 200).
2. `class_size`: Total students in the section.
3. `Term`: Fall, Spring, Summer, or Winter.
4. `subject_hist_gpa`: The subject's historical average GPA.
5. `instructor_hist_gpa`: The instructor's historical average GPA.

*Note: For historical GPAs, the model only uses data from previous semesters to avoid data leakage.*

## Web App (Streamlit)
You can test the models interactively on the web app (`src/app.py`):
- **Inputs**: Easy-to-use dropdowns and sliders for Subject, Instructor, Term, Course Level, and Class Size.
- **Explainability**: The app shows a dynamic SHAP chart, breaking down exactly how each input affected the final predicted GPA.
- **Theming**: Dark and light modes are fully supported.

## Setup and Execution

1. **Clone the Repo**:
    ```bash
    git clone https://github.com/404Mayank/IlliniForecaster.git
    cd IlliniForecaster
    ```

2. **Set up Environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Train the Model**:
    ```bash
    python src/model_comparison.py
    ```

5. **Run the App**:
    ```bash
    streamlit run src/app.py
    ```

## License
This project is licensed under the [MIT License](LICENSE).
