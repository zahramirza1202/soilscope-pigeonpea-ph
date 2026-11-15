import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------
# Pigeonpea reference values for screening
# ---------------------------------------------------------
PIGEONPEA_PREFERRED_PH = (5.0, 7.0)
PIGEONPEA_TOLERATED_PH = (4.5, 8.4)
PIGEONPEA_TYPICAL_TEMP_C = (18.0, 30.0)
PIGEONPEA_ANNUAL_RAIN_MM = (600.0, 1000.0)

PH_REFERENCE_URL = (
    "https://www.ctahr.hawaii.edu/oc/freepubs/pdf/GreenManureCrops/pigeonpea.pdf"
)
AGRONOMY_REFERENCE_URL = (
    "https://oar.icrisat.org/2422/1/Pigeonpea-Botany-Production-Practices.pdf"
)


# ---------------------------------------------------------
# Page setup and styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="SoilScope | Pigeonpea Suitability",
    page_icon="🌱",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f1f7f1 0%, #fbfcfa 450px);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 {
            color: #193b2b;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e1e9e2;
            padding: 16px;
            border-radius: 12px;
        }
        div[data-testid="stAlert"] {
            border-radius: 10px;
        }
        div[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid #e1e9e2;
            border-radius: 12px;
        }
        .eyebrow {
            color: #52745e;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        div.stFormSubmitButton > button {
            background: #24583b;
            border: 0;
            border-radius: 9px;
            color: white;
            font-weight: 650;
            min-height: 2.7rem;
        }
        div.stFormSubmitButton > button:hover {
            background: #193b2b;
            color: white;
            border: 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Introduction
# ---------------------------------------------------------
st.markdown(
    '<div class="eyebrow">Agriculture research prototype</div>',
    unsafe_allow_html=True,
)
st.title("🌱 SoilScope")
st.subheader("Soil pH prediction and pigeonpea screening")

st.write(
    "Estimate soil pH from the uploaded dataset, compare prediction "
    "models, and review how the available conditions align with pigeonpea "
    "growing references."
)

st.info(
    "The suitability result is an indicative screening, not a guarantee "
    "of crop success. Use measured soil results and local agricultural "
    "guidance before making planting decisions."
)


# ---------------------------------------------------------
# Dataset upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your crop recommendation CSV",
    type=["csv"],
)

if uploaded_file is None:
    st.markdown("### Expected CSV columns")

    st.dataframe(
        pd.DataFrame(
            {
                "Column": [
                    "N",
                    "P",
                    "K",
                    "temperature",
                    "humidity",
                    "ph",
                    "rainfall",
                    "label (optional)",
                ],
                "Description": [
                    "Nitrogen measurement",
                    "Phosphorus measurement",
                    "Potassium measurement",
                    "Temperature, expected in °C",
                    "Relative humidity",
                    "Measured soil pH; model prediction target",
                    "Rainfall value; check its unit and time period",
                    "Crop category",
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.stop()


try:
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV file is empty.")
    st.stop()


# ---------------------------------------------------------
# Standardize column names
# ---------------------------------------------------------
df.columns = [
    str(column).strip().lower().replace(" ", "_")
    for column in df.columns
]

aliases = {
    "nitrogen": "n",
    "phosphorus": "p",
    "potassium": "k",
    "soil_ph": "ph",
    "p_h": "ph",
    "temp": "temperature",
    "relative_humidity": "humidity",
    "crop": "label",
    "crop_name": "label",
}
df = df.rename(columns=aliases)

if df.columns.duplicated().any():
    st.error(
        "Duplicate column names were found after standardizing the CSV. "
        "Please correct the headers and upload it again."
    )
    st.stop()


required_columns = [
    "n",
    "p",
    "k",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

feature_columns = [
    "n",
    "p",
    "k",
    "temperature",
    "humidity",
    "rainfall",
]

feature_labels = {
    "n": "Nitrogen (N)",
    "p": "Phosphorus (P)",
    "k": "Potassium (K)",
    "temperature": "Temperature (°C)",
    "humidity": "Humidity",
    "rainfall": "Rainfall",
}

feature_symbols = {
    "n": "N",
    "p": "P",
    "k": "K",
    "temperature": "T",
    "humidity": "H",
    "rainfall": "R",
}

missing_columns = [
    column for column in required_columns if column not in df.columns
]

if missing_columns:
    st.error("Missing required columns: " + ", ".join(missing_columns))
    st.write("Columns found:", df.columns.tolist())
    st.stop()


# ---------------------------------------------------------
# Clean data
# ---------------------------------------------------------
original_rows = len(df)

for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df[required_columns] = df[required_columns].replace(
    [np.inf, -np.inf],
    np.nan,
)

# A row without a valid pH target cannot train the pH models.
df = df.dropna(subset=["ph"]).copy()
df = df[df["ph"] > 0].copy()

if len(df) < 10:
    st.error(
        "At least 10 records with valid, positive pH values are required."
    )
    st.stop()

cleaned_data = df.copy()
cleaned_rows = len(cleaned_data)


# ---------------------------------------------------------
# Select the training population
# ---------------------------------------------------------
st.subheader("Choose the model population")

if "label" in cleaned_data.columns:
    crop_options = ["All crop categories"] + sorted(
        cleaned_data["label"].dropna().astype(str).unique().tolist()
    )

    default_crop = (
        "pigeonpeas"
        if "pigeonpeas" in crop_options
        else "All crop categories"
    )

    selected_crop = st.selectbox(
        "Train and evaluate using",
        crop_options,
        index=crop_options.index(default_crop),
        help=(
            "Selecting pigeonpeas trains the pH models on pigeonpea-labeled "
            "rows only."
        ),
    )

    if selected_crop == "All crop categories":
        df = cleaned_data.copy()
        training_scope = "All crop categories"
    else:
        df = cleaned_data[
            cleaned_data["label"].astype(str) == selected_crop
        ].copy()
        training_scope = selected_crop
else:
    df = cleaned_data.copy()
    training_scope = "All uploaded records"

if len(df) < 10:
    st.error(
        f"Only {len(df)} usable records are available in this selection. "
        "Choose a broader group or upload more records."
    )
    st.stop()

X = df[feature_columns].copy()
y = df["ph"].copy()


# ---------------------------------------------------------
# Dataset description
# ---------------------------------------------------------
st.header("1. Dataset description")

st.write(
    "Each row contains soil and climate measurements. The model uses "
    "nitrogen, phosphorus, potassium, temperature, humidity, and rainfall "
    "to estimate the `ph` value. The optional `label` column identifies "
    "the crop category."
)

st.caption(
    f"File: {uploaded_file.name} · Training scope: {training_scope}. "
    "Check the original dataset documentation for measurement units and "
    "collection methods."
)

with st.expander("Column guide"):
    st.dataframe(
        pd.DataFrame(
            {
                "Column": [
                    "n",
                    "p",
                    "k",
                    "temperature",
                    "humidity",
                    "rainfall",
                    "ph",
                    "label",
                ],
                "Meaning": [
                    "Nitrogen input feature",
                    "Phosphorus input feature",
                    "Potassium input feature",
                    "Temperature input feature",
                    "Humidity input feature",
                    "Rainfall input feature",
                    "Measured soil pH target",
                    "Optional crop category",
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )


# ---------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------
st.header("2. Dataset overview")

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Uploaded records", f"{original_rows:,}")
metric2.metric("Valid pH records", f"{cleaned_rows:,}")
metric3.metric("Model records", f"{len(df):,}")
metric4.metric("Model input features", len(feature_columns))

st.write(
    "Missing input values in the selected data:",
    int(X.isna().sum().sum()),
)

if "label" in cleaned_data.columns:
    st.write(
        f"Crop categories in uploaded data: "
        f"{cleaned_data['label'].nunique()}"
    )

st.markdown("#### Measured pH profile in the model data")

ph_col1, ph_col2, ph_col3, ph_col4 = st.columns(4)
ph_col1.metric("Minimum pH", f"{df['ph'].min():.2f}")
ph_col2.metric("Median pH", f"{df['ph'].median():.2f}")
ph_col3.metric("Mean pH", f"{df['ph'].mean():.2f}")
ph_col4.metric("Maximum pH", f"{df['ph'].max():.2f}")

# Round 0.5-wide pH bins so the histogram labels are easy to read.
bin_start = np.floor(df["ph"].min() * 2) / 2
bin_end = np.ceil(df["ph"].max() * 2) / 2
hist_edges = np.arange(bin_start, bin_end + 0.25, 0.5)
if len(hist_edges) < 2:
    hist_edges = np.array([bin_start, bin_start + 0.5])

hist_counts, hist_edges = np.histogram(df["ph"], bins=hist_edges)
histogram_df = pd.DataFrame(
    {
        "pH interval": [
            f"{hist_edges[i]:.1f}–{hist_edges[i + 1]:.1f}"
            for i in range(len(hist_edges) - 1)
        ],
        "Number of records": hist_counts,
    }
).set_index("pH interval")

st.bar_chart(
    histogram_df,
    x_label="pH interval",
    y_label="Number of records",
    width="stretch",
)

with st.expander("Preview records"):
    st.dataframe(df.head(10).reset_index(drop=True), width="stretch")

with st.expander("Descriptive statistics"):
    st.dataframe(
        df[required_columns].describe().round(3),
        width="stretch",
    )

with st.expander("Missing values by column"):
    missing_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing values": df.isna().sum().values,
        }
    )
    st.dataframe(missing_df, hide_index=True, width="stretch")


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )
except ValueError as error:
    st.error(f"Unable to split the dataset: {error}")
    st.stop()


# ---------------------------------------------------------
# Define regression models
# ---------------------------------------------------------
models = {
    "Mean Baseline": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="mean")),
        ]
    ),
    "Linear Regression": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}


# ---------------------------------------------------------
# Train and evaluate models
# ---------------------------------------------------------
results = {}
trained_models = {}
prediction_results = {}

cv_splitter = KFold(
    n_splits=min(5, len(X)),
    shuffle=True,
    random_state=42,
)

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, predictions) if len(y_test) >= 2 else np.nan

        # Cross-validation on all model records gives a steadier estimate
        # than the small single test split.
        cv_predictions = cross_val_predict(model, X, y, cv=cv_splitter)
        cv_rmse = float(np.sqrt(mean_squared_error(y, cv_predictions)))
        cv_r2 = float(r2_score(y, cv_predictions))

        results[name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2,
            "CV RMSE": cv_rmse,
            "CV R²": cv_r2,
        }
        trained_models[name] = model
        prediction_results[name] = predictions

    except Exception as error:
        st.error(f"Could not train {name}: {error}")

if not trained_models:
    st.error("No models could be trained. Check the uploaded dataset.")
    st.stop()


# ---------------------------------------------------------
# Compare models
# ---------------------------------------------------------
st.header("3. Model comparison")

results_df = pd.DataFrame(results).T
results_df.index.name = "Model"

st.write(
    "All models use the same train/test split. Lower MAE, MSE, and RMSE "
    "mean smaller pH prediction errors. R² compares performance with a "
    "constant-mean prediction; a value at or below 0 means no better than "
    "that. CV RMSE and CV R² come from 5-fold cross-validation on all model "
    "records and are less affected by one lucky or unlucky split."
)

st.caption(
    f"Train/test split: {len(y_train)} training records and "
    f"{len(y_test)} test records."
)

st.dataframe(results_df.round(4), width="stretch")

st.markdown("#### Test-set RMSE")
st.bar_chart(
    results_df[["RMSE"]].sort_values("RMSE"),
    x_label="Model",
    y_label="Test-set RMSE",
    width="stretch",
)

best_model_name = results_df["CV RMSE"].idxmin()
best_model = trained_models[best_model_name]
best_cv_r2 = float(results_df.loc[best_model_name, "CV R²"])

if best_model_name == "Mean Baseline":
    st.warning(
        "No model beat the mean baseline in cross-validation. The inputs "
        "explain very little of the pH variation in this data, so pH "
        "estimates here are not reliable."
    )
elif best_cv_r2 < 0.05:
    st.warning(
        f"Lowest cross-validated error: {best_model_name} "
        f"(CV R² = {best_cv_r2:.2f}). This is no meaningful improvement "
        "over the mean baseline, so the inputs do not predict pH in this "
        "data. Use a measured soil-test pH rather than the model estimate."
    )
elif best_cv_r2 < 0.3:
    st.warning(
        f"Lowest cross-validated error: {best_model_name} "
        f"(CV R² = {best_cv_r2:.2f}). This is only a small improvement over "
        "the mean baseline, so treat pH estimates as rough and prefer a "
        "measured soil-test pH."
    )
else:
    st.success(
        f"Best pH prediction model by cross-validation: {best_model_name} "
        f"(CV R² = {best_cv_r2:.2f})"
    )


# ---------------------------------------------------------
# Explain the formulas
# ---------------------------------------------------------
st.header("4. How the models calculate pH")

baseline_tab, linear_tab, forest_tab = st.tabs(
    ["Mean Baseline", "Linear Regression", "Random Forest"]
)

with baseline_tab:
    baseline_model = trained_models.get("Mean Baseline")

    st.write(
        "The baseline ignores the input measurements. It predicts the "
        "average pH from the training records for every case."
    )

    if baseline_model is not None:
        baseline_value = float(
            np.asarray(
                baseline_model.named_steps["model"].constant_
            ).reshape(-1)[0]
        )

        st.latex(
            r"\widehat{pH}_{baseline}"
            r" = \frac{1}{n}\sum_{i=1}^{n}pH_i"
        )

        baseline_col1, baseline_col2 = st.columns(2)
        baseline_col1.metric("Training records", f"{len(y_train):,}")
        baseline_col2.metric("Baseline pH estimate", f"{baseline_value:.4f}")

        st.caption(
            "The baseline estimate stays the same when you change the "
            "prediction form inputs."
        )


with linear_tab:
    st.write(
        "Linear Regression multiplies each input by a learned coefficient "
        "and adds the results to an intercept."
    )

    linear_pipeline = trained_models.get("Linear Regression")

    if linear_pipeline is not None:
        fitted_linear = linear_pipeline.named_steps["model"]

        equation_terms = [
            f"{coefficient:+.4f}x_{{{feature_symbols[feature]}}}"
            for feature, coefficient in zip(
                feature_columns,
                fitted_linear.coef_,
            )
        ]

        equation = (
            rf"\widehat{{pH}} = {fitted_linear.intercept_:.4f} "
            + " ".join(equation_terms)
        )
        st.latex(equation)

        coefficient_table = pd.DataFrame(
            {
                "Feature": [
                    feature_labels[feature] for feature in feature_columns
                ],
                "Coefficient": fitted_linear.coef_,
            }
        )

        st.write(f"**Intercept:** {fitted_linear.intercept_:.6f}")
        st.dataframe(
            coefficient_table.round(6),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Coefficients are learned from the training data. They describe "
            "model relationships and do not establish cause and effect."
        )


with forest_tab:
    forest_pipeline = trained_models.get("Random Forest")

    st.write(
        "Random Forest creates many decision trees and averages their "
        "individual pH estimates."
    )
    st.latex(
        r"\widehat{pH}_{RF}(x)"
        r" = \frac{1}{M}\sum_{m=1}^{M}T_m(x)"
    )

    if forest_pipeline is not None:
        forest_model = forest_pipeline.named_steps["model"]

        forest_col1, forest_col2 = st.columns(2)
        forest_col1.metric(
            "Trees averaged",
            f"{len(forest_model.estimators_)}",
        )
        forest_col2.metric("Training records", f"{len(y_train):,}")

        st.write(
            "Settings: 200 trees, minimum 2 training records per leaf, "
            "and random seed 42."
        )


test_predictions = pd.DataFrame(
    {
        "Actual pH": y_test.to_numpy(),
        "Predicted pH": prediction_results[best_model_name],
    }
)

with st.expander(f"Actual versus predicted test values ({best_model_name})"):
    axis_low = float(test_predictions.min().min()) - 0.1
    axis_high = float(test_predictions.max().max()) + 0.1
    axis_scale = alt.Scale(domain=[axis_low, axis_high])

    points = (
        alt.Chart(test_predictions)
        .mark_circle(size=80, color="#24583b", opacity=0.8)
        .encode(
            x=alt.X("Actual pH", scale=axis_scale, axis=alt.Axis(tickCount=6, format=".1f")),
            y=alt.Y("Predicted pH", scale=axis_scale),
            tooltip=["Actual pH", "Predicted pH"],
        )
    )

    ideal_line = (
        alt.Chart(
            pd.DataFrame(
                {"Actual pH": [axis_low, axis_high],
                 "Predicted pH": [axis_low, axis_high]}
            )
        )
        .mark_line(strokeDash=[5, 5], color="#999999")
        .encode(
            x=alt.X("Actual pH", scale=axis_scale, axis=alt.Axis(tickCount=6, format=".1f")),
            y=alt.Y("Predicted pH", scale=axis_scale),
        )
    )

    st.altair_chart((points + ideal_line).properties(height=380))
    st.caption(
        "The dashed line is a perfect prediction. Points that stay close "
        "to a horizontal band mean the model is mostly predicting the "
        "average pH."
    )

    st.dataframe(
        test_predictions.round(3),
        hide_index=True,
        width="stretch",
    )


# ---------------------------------------------------------
# Research reference
# ---------------------------------------------------------
st.header("5. Pigeonpea growing references")

st.write(
    "Pigeonpea is commonly described as preferring soil pH around "
    "**5.0–7.0**, with a wider reported tolerance of about **4.5–8.4**. "
    "Typical growing temperatures are often around **18–30°C**. "
    "A general annual rainfall reference is **600–1,000 mm**, but rainfall "
    "distribution and local conditions matter."
)

st.markdown(
    f"""
    - [University of Hawaiʻi CTAHR pigeonpea guide]({PH_REFERENCE_URL})
    - [ICRISAT pigeonpea production guide]({AGRONOMY_REFERENCE_URL})
    """
)

st.caption(
    "These ranges are general references. They are not universal pass/fail "
    "limits for every variety or location."
)


# ---------------------------------------------------------
# Prediction and suitability form
# ---------------------------------------------------------
st.header("6. Estimate pH and review pigeonpea suitability")

st.write(
    "Enter the soil and climate values. The model estimates pH. The "
    "screening below then checks pH and the additional site conditions "
    "you report."
)

st.caption(
    "Use temperature in °C. The CSV does not document the rainfall period "
    "or units, so the app will only compare rainfall with an annual range "
    "if you confirm that it is an annual total in millimetres."
)


def get_input_range(column):
    values = df[column].dropna()

    if values.empty:
        return 0.0, 100.0

    low = float(values.min())
    high = float(values.max())

    if low == high:
        padding = max(abs(low) * 0.1, 1.0)
        low -= padding
        high += padding

    return low, high


with st.form("pigeonpea_screening_form"):
    input_values = {}
    col1, col2, col3 = st.columns(3)

    input_layout = [
        ("n", "Nitrogen (N)", col1),
        ("p", "Phosphorus (P)", col2),
        ("k", "Potassium (K)", col3),
        ("temperature", "Temperature (°C)", col1),
        ("humidity", "Humidity", col2),
        ("rainfall", "Rainfall value from dataset", col3),
    ]

    for key, label, column in input_layout:
        low, high = get_input_range(key)
        median_value = df[key].median()

        if pd.isna(median_value):
            median_value = (low + high) / 2

        median_value = min(max(float(median_value), low), high)

        with column:
            input_values[key] = st.number_input(
                label,
                min_value=low,
                max_value=high,
                value=median_value,
                format="%.2f",
                key=f"input_{key}",
            )

    st.markdown("#### Additional field information")

    extra_col1, extra_col2 = st.columns(2)

    with extra_col1:
        rainfall_basis = st.selectbox(
            "Rainfall measurement",
            [
                "Unknown or not an annual total",
                "Annual total in millimetres",
            ],
        )

        drainage_status = st.selectbox(
            "Field drainage",
            [
                "Unknown",
                "Well drained",
                "Waterlogging is likely",
            ],
        )

    with extra_col2:
        salinity_status = st.selectbox(
            "Soil salinity",
            [
                "Unknown",
                "No known salinity concern",
                "Salinity concern from soil test",
            ],
        )

        sunlight_status = st.selectbox(
            "Sunlight during flowering and pod development",
            [
                "Unknown",
                "Full sunlight available",
                "Shade or low sunlight likely",
            ],
        )

    # Optional soil-test pH. Leave empty to use the model estimate.
    measured_ph = st.number_input(
        "Measured soil pH (optional)",
        min_value=0.0,
        max_value=14.0,
        value=None,
        step=0.1,
        format="%.2f",
        placeholder="Leave empty to use the model estimate",
        help=(
            "A soil-test pH is preferred for the crop screen. If left "
            "empty, the model-estimated pH is used instead."
        ),
    )

    submitted = st.form_submit_button("Estimate and assess")

has_measured_ph = measured_ph is not None


if submitted:
    input_df = pd.DataFrame(
        [input_values],
        columns=feature_columns,
    )

    try:
        # Calculate pH from all three models.
        predictions_by_model = {
            name: float(model.predict(input_df)[0])
            for name, model in trained_models.items()
        }

        selected_prediction = predictions_by_model[best_model_name]
        screening_ph = (
            float(measured_ph) if has_measured_ph else selected_prediction
        )
        ph_source_text = (
            "measured soil pH" if has_measured_ph else "model-estimated pH"
        )

        # -------------------------------------------------
        # Evaluate factors using a simple, explainable screen
        # -------------------------------------------------
        factor_rows = []
        major_constraints = []
        cautions = []
        missing_checks = []

        # pH assessment
        preferred_min, preferred_max = PIGEONPEA_PREFERRED_PH
        tolerated_min, tolerated_max = PIGEONPEA_TOLERATED_PH

        if preferred_min <= screening_ph <= preferred_max:
            factor_rows.append(
                {
                    "Factor": "Soil pH",
                    "Finding": "Within preferred reference range",
                    "Reason": (
                        f"{screening_ph:.2f} is within "
                        f"{preferred_min:.1f}–{preferred_max:.1f}."
                    ),
                }
            )
        elif tolerated_min <= screening_ph <= tolerated_max:
            factor_rows.append(
                {
                    "Factor": "Soil pH",
                    "Finding": "Within wider tolerance; outside preferred range",
                    "Reason": (
                        f"{screening_ph:.2f} is within the wider "
                        f"{tolerated_min:.1f}–{tolerated_max:.1f} range, "
                        f"but outside the preferred {preferred_min:.1f}–"
                        f"{preferred_max:.1f} range."
                    ),
                }
            )
            cautions.append("pH is outside the preferred range")
        else:
            factor_rows.append(
                {
                    "Factor": "Soil pH",
                    "Finding": "Potential constraint",
                    "Reason": (
                        f"{screening_ph:.2f} is outside the reported wider "
                        f"tolerance range of {tolerated_min:.1f}–"
                        f"{tolerated_max:.1f}."
                    ),
                }
            )
            major_constraints.append("pH is outside the reported tolerance range")

        # Temperature assessment
        temp_min, temp_max = PIGEONPEA_TYPICAL_TEMP_C
        temperature = float(input_values["temperature"])

        if temp_min <= temperature <= temp_max:
            factor_rows.append(
                {
                    "Factor": "Temperature",
                    "Finding": "Within a commonly reported range",
                    "Reason": (
                        f"{temperature:.2f}°C is within "
                        f"{temp_min:.0f}–{temp_max:.0f}°C."
                    ),
                }
            )
        else:
            factor_rows.append(
                {
                    "Factor": "Temperature",
                    "Finding": "Outside typical reference range",
                    "Reason": (
                        f"{temperature:.2f}°C is outside "
                        f"{temp_min:.0f}–{temp_max:.0f}°C; variety, season, "
                        "and moisture can affect tolerance."
                    ),
                }
            )
            cautions.append("temperature is outside the typical range")

        # Rainfall is only assessed if its period and units are known.
        rainfall_value = float(input_values["rainfall"])
        rain_min, rain_max = PIGEONPEA_ANNUAL_RAIN_MM

        if rainfall_basis == "Annual total in millimetres":
            if rain_min <= rainfall_value <= rain_max:
                factor_rows.append(
                    {
                        "Factor": "Annual rainfall",
                        "Finding": "Within a general reference range",
                        "Reason": (
                            f"{rainfall_value:.2f} mm is within the general "
                            f"{rain_min:.0f}–{rain_max:.0f} mm annual range."
                        ),
                    }
                )
            else:
                factor_rows.append(
                    {
                        "Factor": "Annual rainfall",
                        "Finding": "Outside the general reference range",
                        "Reason": (
                            f"{rainfall_value:.2f} mm is outside the general "
                            f"{rain_min:.0f}–{rain_max:.0f} mm annual range. "
                            "Rainfall timing and irrigation also matter."
                        ),
                    }
                )
                cautions.append("annual rainfall is outside the general range")
        else:
            factor_rows.append(
                {
                    "Factor": "Rainfall",
                    "Finding": "Not assessed",
                    "Reason": (
                        "The dataset's rainfall unit or time period has not "
                        "been confirmed as an annual total in millimetres."
                    ),
                }
            )
            missing_checks.append("confirm rainfall amount, unit, and period")

        # Drainage assessment
        if drainage_status == "Well drained":
            factor_rows.append(
                {
                    "Factor": "Drainage",
                    "Finding": "Favorable condition reported",
                    "Reason": "You reported that the field is well drained.",
                }
            )
        elif drainage_status == "Waterlogging is likely":
            factor_rows.append(
                {
                    "Factor": "Drainage",
                    "Finding": "Potential major constraint",
                    "Reason": (
                        "Pigeonpea is sensitive to waterlogging and needs "
                        "free-draining soil."
                    ),
                }
            )
            major_constraints.append("waterlogging risk was reported")
        else:
            factor_rows.append(
                {
                    "Factor": "Drainage",
                    "Finding": "Not assessed",
                    "Reason": "Field drainage information was not provided.",
                }
            )
            missing_checks.append("check field drainage and waterlogging risk")

        # Salinity assessment
        if salinity_status == "No known salinity concern":
            factor_rows.append(
                {
                    "Factor": "Salinity",
                    "Finding": "No concern reported",
                    "Reason": (
                        "You reported no known salinity issue. A soil test "
                        "is needed to confirm this."
                    ),
                }
            )
        elif salinity_status == "Salinity concern from soil test":
            factor_rows.append(
                {
                    "Factor": "Salinity",
                    "Finding": "Potential constraint",
                    "Reason": (
                        "Pigeonpea is sensitive to high salinity; review "
                        "the soil-test electrical conductivity result."
                    ),
                }
            )
            cautions.append("a salinity concern was reported")
        else:
            factor_rows.append(
                {
                    "Factor": "Salinity",
                    "Finding": "Not assessed",
                    "Reason": "No soil salinity information was provided.",
                }
            )
            missing_checks.append("check soil salinity or electrical conductivity")

        # Sunlight assessment
        if sunlight_status == "Full sunlight available":
            factor_rows.append(
                {
                    "Factor": "Sunlight",
                    "Finding": "Favorable condition reported",
                    "Reason": (
                        "You reported full sunlight during flowering and "
                        "pod development."
                    ),
                }
            )
        elif sunlight_status == "Shade or low sunlight likely":
            factor_rows.append(
                {
                    "Factor": "Sunlight",
                    "Finding": "Potential constraint",
                    "Reason": (
                        "Low sunlight during reproductive growth can limit "
                        "pigeonpea performance."
                    ),
                }
            )
            cautions.append("low sunlight was reported")
        else:
            factor_rows.append(
                {
                    "Factor": "Sunlight",
                    "Finding": "Not assessed",
                    "Reason": "Sunlight information was not provided.",
                }
            )
            missing_checks.append("check sunlight during flowering and pod development")

        # NPK and humidity are shown but not classified as good/bad because
        # their dataset units and measurement methods are undocumented.
        factor_rows.append(
            {
                "Factor": "N, P, K and humidity",
                "Finding": "Recorded, not classified",
                "Reason": (
                    "The CSV does not document units or soil-test methods "
                    "for NPK, and no suitability thresholds are applied "
                    "to these fields."
                ),
            }
        )

        # -------------------------------------------------
        # Result first: pH estimate and overall verdict
        # -------------------------------------------------
        st.subheader("Result")

        result_col1, result_col2 = st.columns(2)
        result_col1.metric(
            f"Model-estimated soil pH ({best_model_name})",
            f"{selected_prediction:.2f}",
        )
        result_col2.metric(
            f"pH used for suitability screen ({ph_source_text})",
            f"{screening_ph:.2f}",
        )

        if has_measured_ph:
            st.caption(
                "The suitability screen uses your measured soil-test pH. "
                f"The model estimate differs from it by "
                f"{abs(selected_prediction - float(measured_ph)):.2f} pH units."
            )
        else:
            st.caption(
                "The suitability screen uses the model-estimated pH. "
                "A measured soil-test pH is preferable."
            )
            if best_cv_r2 < 0.05:
                st.warning(
                    "This pH estimate is close to the dataset average and "
                    "does not reflect the values you entered. Enter a "
                    "measured soil pH for a reliable screen."
                )

        if major_constraints:
            st.warning(
                "Potentially unsuitable conditions were flagged. "
                "Review the listed constraint(s) before planting."
            )
            st.write("Flagged: " + "; ".join(major_constraints) + ".")
        elif missing_checks:
            st.info(
                "Promising on the available indicators, but the assessment "
                "is incomplete."
            )
            st.write(
                "To strengthen the screen, " + "; ".join(missing_checks) + "."
            )
        elif cautions:
            st.warning(
                "Some conditions are within range, with factors that need "
                "local review."
            )
            st.write("Review: " + "; ".join(cautions) + ".")
        else:
            st.success(
                "Likely suitable based on the entered screening factors."
            )
            st.write(
                "The checked pH, temperature, rainfall, drainage, salinity, "
                "and sunlight indicators are within the screening conditions "
                "entered here. This is not a guarantee of crop success."
            )

        st.caption(
            f"The screen used {ph_source_text}. It does not evaluate every "
            "field condition or replace local agronomic advice."
        )

        st.subheader("Pigeonpea suitability factors")
        st.dataframe(
            pd.DataFrame(factor_rows),
            hide_index=True,
            width="stretch",
        )

        # -------------------------------------------------
        # Supporting calculation details, collapsed by default
        # -------------------------------------------------
        with st.expander("Show how each model calculated this estimate"):
            input_table = pd.DataFrame(
                {
                    "Feature": [
                        feature_labels[feature] for feature in feature_columns
                    ],
                    "Value used by pH models": [
                        input_values[feature] for feature in feature_columns
                    ],
                }
            )

            st.markdown("#### Values used by the pH models")
            st.dataframe(
                input_table.round(4),
                hide_index=True,
                width="stretch",
            )

            prediction_table = pd.DataFrame(
                {
                    "Model": list(predictions_by_model.keys()),
                    "Estimated pH": list(predictions_by_model.values()),
                    "Test RMSE": [
                        results_df.loc[name, "RMSE"]
                        for name in predictions_by_model
                    ],
                    "CV RMSE": [
                        results_df.loc[name, "CV RMSE"]
                        for name in predictions_by_model
                    ],
                }
            ).sort_values("CV RMSE")

            st.markdown("#### pH estimate by model")
            st.dataframe(
                prediction_table.round(4),
                hide_index=True,
                width="stretch",
            )

            # Linear Regression calculation
            linear_pipeline = trained_models.get("Linear Regression")

            if linear_pipeline is not None:
                st.markdown("#### Linear Regression calculation")

                fitted_linear = linear_pipeline.named_steps["model"]
                entered_array = np.array(
                    [input_values[feature] for feature in feature_columns],
                    dtype=float,
                )
                contributions = fitted_linear.coef_ * entered_array

                contribution_table = pd.DataFrame(
                    {
                        "Feature": [
                            feature_labels[feature]
                            for feature in feature_columns
                        ],
                        "Value entered": entered_array,
                        "Learned coefficient": fitted_linear.coef_,
                        "Contribution": contributions,
                    }
                )

                st.dataframe(
                    contribution_table.round(5),
                    hide_index=True,
                    width="stretch",
                )

                st.write(
                    f"Intercept {fitted_linear.intercept_:.4f} + sum of "
                    f"feature contributions = "
                    f"**{predictions_by_model['Linear Regression']:.4f} pH**."
                )

            # Random Forest calculation
            forest_pipeline = trained_models.get("Random Forest")

            if forest_pipeline is not None:
                st.markdown("#### Random Forest calculation")

                forest_model = forest_pipeline.named_steps["model"]

                # Trees inside the pipeline were fitted on imputed data, so
                # pass the imputed input to each tree.
                imputed_input = forest_pipeline.named_steps[
                    "imputer"
                ].transform(input_df)

                tree_predictions = np.array(
                    [
                        tree.predict(imputed_input)[0]
                        for tree in forest_model.estimators_
                    ],
                    dtype=float,
                )

                rf_col1, rf_col2, rf_col3 = st.columns(3)
                rf_col1.metric("Trees averaged", len(tree_predictions))
                rf_col2.metric(
                    "Mean tree estimate",
                    f"{tree_predictions.mean():.4f}",
                )
                rf_col3.metric(
                    "Tree estimate range",
                    f"{tree_predictions.min():.3f}–"
                    f"{tree_predictions.max():.3f}",
                )

                st.caption(
                    "The tree estimate range is not a statistical "
                    "confidence interval."
                )

            # Mean Baseline calculation
            if "Mean Baseline" in predictions_by_model:
                st.markdown("#### Mean Baseline calculation")
                st.latex(
                    r"\widehat{pH}_{baseline}"
                    r" = \frac{1}{n}\sum_{i=1}^{n}pH_i"
                )
                st.write(
                    f"The training-set mean is "
                    f"**{predictions_by_model['Mean Baseline']:.4f} pH**. "
                    "It does not change when the input measurements change."
                )

    except Exception as error:
        st.error(f"Could not generate the assessment: {error}")


# ---------------------------------------------------------
# Limitations and sources
# ---------------------------------------------------------
st.header("7. How to interpret this screen")

st.markdown(
    """
    - “Likely suitable” means the entered screening indicators align with
      the reference checks shown above.
    - It is not a verified prediction of actual yield or crop survival.
    - The dataset's rainfall period and the units for NPK are undocumented;
      those factors are not scored unless their basis is confirmed.
    - The pH models have limited accuracy on this data (see the model
      comparison); a measured soil pH is more reliable than the estimate.
    - Soil texture, drainage, salinity, sunlight, cultivar, pests, and
      management can change the outcome.
    - Use a measured soil pH and a local soil test when available.
    """
)

st.markdown(
    f"""
    **References**

    - [University of Hawaiʻi CTAHR pigeonpea guide]({PH_REFERENCE_URL})
    - [ICRISAT pigeonpea production guide]({AGRONOMY_REFERENCE_URL})
    """
)

st.divider()
st.caption("SoilScope | Pigeonpea suitability screening prototype")