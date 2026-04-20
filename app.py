import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNetCV
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(page_title="Maternal Health Risk — Age Predictor", layout="wide")
st.title(" Maternal Health Risk — Age Prediction (OLS Regression)")

# ─────────────────────────────────────────
# 1. DATA UPLOAD
# ─────────────────────────────────────────
st.sidebar.header("📂 Upload Dataset")
uploaded = st.sidebar.file_uploader(
    "Upload 'Maternal Health Risk Data Set.csv'", type=["csv"]
)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded is None:
    st.info("Please upload the **Maternal Health Risk Data Set.csv** file using the sidebar to get started.")
    st.stop()

raw_df = load_data(uploaded)

# ─────────────────────────────────────────
# 2. PREPROCESSING  (mirrors the notebook)
# ─────────────────────────────────────────
@st.cache_data
def preprocess(raw):
    df = raw.copy()

    # Scale continuous features (exclude Age, RiskLevel, BodyTemp)
    cont_cols = [c for c in df.columns if c not in ["Age", "RiskLevel", "BodyTemp"]]
    ss = StandardScaler()
    df[cont_cols] = ss.fit_transform(df[cont_cols])

    # One-hot encode RiskLevel
    df = pd.get_dummies(df, columns=["RiskLevel"], drop_first=True, dtype="int")

    # Non-linear term for BS (notebook kept only BS2)
    df["BS2"] = df["BS"].values ** 2

    # Drop other quadratic terms that were found non-significant
    for col in ["SystolicBP2", "DiastolicBP2", "BodyTemp2", "HeartRate2"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Remove extreme Age outliers (top 1 %)
    df = df[df["Age"] < df["Age"].quantile(0.99)].copy()

    # Square-root transform on Age (target)
    df["Age"] = np.sqrt(df["Age"])

    return df, ss, cont_cols

df, scaler, cont_cols = preprocess(raw_df)

# ─────────────────────────────────────────
# 3. TRAIN / TEST SPLIT & MODEL
# ─────────────────────────────────────────
@st.cache_data
def fit_model(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    formula = "Age ~ SystolicBP + BS + BS2 + BodyTemp"
    lm = smf.ols(formula, data=train_df).fit()

    y_test_pred    = lm.predict(test_df)
    y_test_actual  = test_df["Age"]

    real_y_test = np.square(y_test_actual)
    real_y_pred = np.square(y_test_pred)

    mae  = mean_absolute_error(real_y_test, real_y_pred)
    rmse = np.sqrt(mean_squared_error(real_y_test, real_y_pred))
    r2   = r2_score(real_y_test, real_y_pred)
    error = real_y_test.values - real_y_pred.values

    return lm, mae, rmse, r2, error, real_y_test, real_y_pred, train_df, test_df

lm, mae, rmse, r2, error, real_y_test, real_y_pred, train_df, test_df = fit_model(df)

# ─────────────────────────────────────────
# 4. TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dataset", "📈 Model Summary", "🎯 Diagnostics", "🔮 Predict Age"]
)

# ── Tab 1: Dataset overview ──────────────
with tab1:
    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", raw_df.shape[0])
    col2.metric("Columns", raw_df.shape[1])
    col3.metric("Missing values", int(raw_df.isnull().sum().sum()))

    st.subheader("Descriptive Statistics")
    st.dataframe(raw_df.describe(), use_container_width=True)

    st.subheader("Risk Level Distribution")
    fig, ax = plt.subplots()
    raw_df["RiskLevel"].value_counts().plot(kind="bar", ax=ax, color=["#e74c3c","#f39c12","#2ecc71"])
    ax.set_xlabel("Risk Level"); ax.set_ylabel("Count"); ax.set_title("Risk Level Counts")
    plt.xticks(rotation=0)
    st.pyplot(fig)

# ── Tab 2: Model Summary ─────────────────
with tab2:
    st.subheader("OLS Regression Summary")
    st.markdown(f"**Formula:** `Age ~ SystolicBP + BS + BS2 + BodyTemp`  \n"
                f"*(Age is √-transformed; back-transformed for metrics)*")

    # Coefficients table
    coef_df = pd.DataFrame({
        "Coefficient": lm.params,
        "Std Error":   lm.bse,
        "t-value":     lm.tvalues,
        "p-value":     lm.pvalues,
    }).round(4)
    st.dataframe(coef_df, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² (test, back-transformed)", f"{r2:.4f}")
    col2.metric("MAE (years)",  f"{mae:.2f}")
    col3.metric("RMSE (years)", f"{rmse:.2f}")

    st.subheader("Coefficient Bar Chart")
    fig2, ax2 = plt.subplots()
    coef_df["Coefficient"].drop("Intercept").plot(
        kind="bar", ax=ax2, color="#3498db"
    )
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("OLS Coefficients (excl. Intercept)")
    ax2.set_ylabel("Coefficient value")
    plt.xticks(rotation=15)
    st.pyplot(fig2)

# ── Tab 3: Diagnostics ───────────────────
with tab3:
    st.subheader("Residual Analysis")

    col_a, col_b = st.columns(2)

    with col_a:
        fig3, ax3 = plt.subplots()
        ax3.hist(error, bins=30, color="#9b59b6", edgecolor="white")
        ax3.axvline(np.mean(error), color="red", linestyle="--",
                    label=f"Mean error = {np.mean(error):.2f}")
        ax3.set_title("Histogram of Residuals")
        ax3.set_xlabel("Error (actual − predicted)")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        st.pyplot(fig3)

    with col_b:
        fig4, ax4 = plt.subplots()
        ax4.scatter(real_y_pred, error, alpha=0.4, color="#1abc9c")
        ax4.axhline(0, color="red", linestyle="--")
        ax4.set_title("Residuals vs Predicted Age")
        ax4.set_xlabel("Predicted Age")
        ax4.set_ylabel("Residual")
        st.pyplot(fig4)

    st.subheader("Actual vs Predicted Age")
    fig5, ax5 = plt.subplots()
    ax5.scatter(real_y_test, real_y_pred, alpha=0.4, color="#e67e22")
    mn = min(real_y_test.min(), real_y_pred.min())
    mx = max(real_y_test.max(), real_y_pred.max())
    ax5.plot([mn, mx], [mn, mx], "r--", label="Perfect fit")
    ax5.set_xlabel("Actual Age"); ax5.set_ylabel("Predicted Age")
    ax5.set_title("Actual vs Predicted Age (test set)")
    ax5.legend()
    st.pyplot(fig5)

    st.subheader("Residual Correlations with Other Features")
    error_series = pd.Series(error, index=test_df.index)
    corr_cols = ["SystolicBP", "BS", "BodyTemp"]
    corr_data = []
    for col in corr_cols:
        if col in test_df.columns:
            c = np.corrcoef(error_series, test_df[col])[0, 1]
            corr_data.append({"Feature": col, "Correlation with Error": round(c, 4)})
    if "BS2" in test_df.columns:
        c = np.corrcoef(error_series, test_df["BS2"])[0, 1]
        corr_data.append({"Feature": "BS2", "Correlation with Error": round(c, 4)})
    st.dataframe(pd.DataFrame(corr_data), use_container_width=True)

# ── Tab 4: Predict Age ───────────────────
with tab4:
    st.subheader("Predict Maternal Age from Health Parameters")
    st.markdown("Enter raw (unscaled) values. The app scales them internally.")

    raw_stats = raw_df.describe()

    def num_input(label, col):
        lo  = float(raw_stats.loc["min", col])
        hi  = float(raw_stats.loc["max", col])
        med = float(raw_stats.loc["50%", col])
        return st.number_input(label, min_value=lo, max_value=hi, value=med, step=1.0)

    c1, c2 = st.columns(2)
    with c1:
        systolic  = num_input("Systolic BP (mmHg)",    "SystolicBP")
        body_temp = num_input("Body Temperature (°F)", "BodyTemp")
    with c2:
        bs        = num_input("Blood Sugar (mmol/L)",  "BS")

    if st.button("Predict Age", type="primary"):
        # Scale SystolicBP and BS (BodyTemp was NOT scaled in the notebook)
        dummy_row = pd.DataFrame([[systolic, 0, bs, 0, 0]],
                                 columns=cont_cols)
        scaled = scaler.transform(dummy_row)[0]
        syst_s = scaled[cont_cols.index("SystolicBP")]
        bs_s   = scaled[cont_cols.index("BS")]
        bs2_s  = bs_s ** 2

        pred_input = pd.DataFrame({
            "SystolicBP": [syst_s],
            "BS":         [bs_s],
            "BS2":        [bs2_s],
            "BodyTemp":   [body_temp],
        })
        sqrt_age = lm.predict(pred_input)[0]
        age_pred = sqrt_age ** 2

        st.success(f"### Predicted Maternal Age: **{age_pred:.1f} years**")
        st.caption("Model R² ≈ 0.41 — use as a statistical estimate, not a clinical decision.")
