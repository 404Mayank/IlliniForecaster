# IlliniForecaster: UIUC GPA Prediction Pipeline

This repository hosts a machine learning pipeline and interactive Streamlit web application designed to forecast section-level average GPAs for courses at the University of Illinois Urbana-Champaign (UIUC). It serves as the primary artifact for MUJ's ML Subject Project.

## Overview

The core objective is to accurately predict the `avg_gpa` of a given course section based on historical grade data. The target variable is derived directly from raw grade counts (`A+` through `F`, excluding `W`), where `avg_gpa = weighted_grade_points / graded_students`. 

The web application (`src/app.py`) provides an interactive interface to manipulate input parameters and view real-time predictions alongside SHAP (SHapley Additive exPlanations) values to interpret model feature importance.

## Technical Architecture & Modeling

The predictive modeling pipeline is housed in `src/model_comparison.py`, evaluating three regressors against the target:

1.  **LightGBM (`lightgbm.LGBMRegressor`)**: A gradient boosting framework that utilizes tree-based learning algorithms. It is configured to grow trees leaf-wise rather than depth-wise, optimizing for lower loss. Key parameters include `learning_rate=0.05`, `num_leaves=31`, `subsample=0.9`, and `colsample_bytree=0.9`. In baseline tests, this model achieved the lowest RMSE.
2.  **XGBoost (`xgboost.XGBRegressor`)**: A robust, distributed gradient boosting library. It builds trees depth-wise. Key parameters mirror the LightGBM configuration for a controlled baseline: `max_depth=6`, `learning_rate=0.05`, `subsample=0.9`, and `colsample_bytree=0.9`.
3.  **Random Forest (`sklearn.ensemble.RandomForestRegressor`)**: An ensemble learning method constructing multiple decision trees via bootstrap aggregation (bagging). It serves as a strong, variance-reducing baseline with `n_estimators=300` and `min_samples_leaf=2`.

### Temporal Validation & Split

To evaluate real-world predictive performance and prevent look-ahead bias, a strict chronological train/test split is enforced.

-   **Temporal Indexing**: Every semester is assigned a sequential index based on chronological order (e.g., Spring 2020 < Fall 2020 ...).
-   **Split Methodology**: An 80/20 chronological partition is used. The earliest 80% of terms (approx. 61,000 rows) are isolated for training, while the most recent 20% of terms (approx. 15,700 rows) are held out purely for testing.
-   **Purpose**: This guarantees the models are evaluated identically to a production deployment scenario—training on past data to predict future, unseen term results.

### Leakage-Safe Feature Engineering

The pipeline engineers historical features that summarize past performance. Crucially, these calculations rigorously obey the temporal index, ensuring that for any given term $T$, the historical aggregations only incorporate data from terms $T-1, T-2, \ldots, T_0$. 

The engineering process yields the following features delivered to the models:

1.  `course_level`: An integer representation derived from the course number (e.g., CS 225 -> 200).
2.  `class_size`: The total number of graded students in the section.
3.  `Term`: Categorical representation (`Fall`, `Spring`, `Summer`, `Winter`), one-hot encoded by the pipeline.
4.  `subject_hist_gpa`: The historically accumulated average GPA for the selected subject (e.g., `CS`, `STAT`).
5.  `instructor_hist_gpa`: The historically accumulated average GPA for the selected primary instructor.

If an instructor or subject is unobserved in the historical window preceding a given prediction row, the value falls back to the global historical mean up to that point.

## Application Interface (Streamlit)

The web UI provides an intuitive, preset-driven control panel mapped to the engineered features. The UI logic abstracts the historical target encoding away from the user:

-   **Global Data Alignment**: The Streamlit app loads the full historical dataset to construct definitive lookup tables.
-   **User Inputs**:
    -   `Subject` (Dropdown)
    -   `Instructor Name` (Dropdown, dynamically filtered by `Subject`)
    -   `Term` (Dropdown)
    -   `Course Level` (Select Slider: 100-700)
    -   `Class Size` (Integer Slider)
-   **Tooltip Explanations**: A prominent "?" tooltip provides insight into how the predictive engine operates. It details that `Subject` and `Instructor` inputs are implicitly mapped to `subject_hist_gpa` and `instructor_hist_gpa` using the full historical dataset average computed prior to inference. It specifically notes that if an instructor is selected who has never taught prior to the prediction context, the model defaults to the global university average.
-   **Visualization**: SHAP values are extracted dynamically using `shap.TreeExplainer` on the best-performing model artifact and rendered via Matplotlib. The GUI theme is customized (CSS injection) from default pink to a professional blue scheme, ensuring high-contrast visibility for SHAP plots in both Light and Dark modes.

## Setup and Execution

1.  **Clone and Enter Directory**:
    ```bash
    git clone https://github.com/404Mayank/IlliniForecaster.git
    cd IlliniForecaster
    ```

2.  **Environment Initialization**:
    It is recommended to use a virtual environment.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute Model Pipeline (Training)**:
    This command downloads the latest data, runs the chronologically split training pipeline, evaluates the models, and persists the `best_model.joblib` to the `artifacts/` directory.
    ```bash
    python src/model_comparison.py
    ```

5.  **Launch Web Application**:
    ```bash
    streamlit run src/app.py
    ```
