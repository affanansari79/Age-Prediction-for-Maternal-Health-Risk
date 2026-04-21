import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Maternal Health Risk - Age Predictor", layout="wide")
st.title("Maternal Health Risk - Age Prediction (OLS Regression)")

# 1. DATA UPLOAD
st.sidebar.header("Upload Dataset")
uploaded = st.sidebar.file_uploader("Upload Maternal Health Risk Data Set.csv", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded is None:
    st.info("Please upload the Maternal Health Risk Data Set.csv file using the sidebar to get started.")
    st.stop()

raw_df = load_data(uploaded)

# 2. PREPROCESSING
@st.cache_data
def preprocess(raw):
    df = raw.copy()
    cont_cols = [c for c in df.columns if c not in ["Age", "RiskLevel", "BodyTemp"]]
    ss = StandardScaler()
    df[cont_cols] = ss.fit_transform(df[cont_cols])
    df = pd.get_dummies(df, columns=["RiskLevel"], drop_first=True, dtype="int")
    df["BS2"] = df["BS"].values ** 2
    df = df[df["Age"] < df["Age"].quantile(0.99)].copy()
    df["Age"] = np.sqrt(df["Age"])
    return df, ss, cont_cols

df, scaler, cont_cols = preprocess(raw_df)

# 3. MODEL
@st.cache_data
def fit_model(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    lm = smf.ols("Age ~ SystolicBP + BS + BS2 + BodyTemp", data=train_df).fit()
    y_pred    = lm.predict(test_df)
    y_actual  = test_df["Age"]
    real_test = np.square(y_actual)
    real_pred = np.square(y_pred)
    mae   = mean_absolute_error(real_test, real_pred)
    rmse  = np.sqrt(mean_squared_error(real_test, real_pred))
    r2    = r2_score(real_test, real_pred)
    error = real_test.values - real_pred.values
    return lm, mae, rmse, r2, error, real_test, real_pred, test_df

lm, mae, rmse, r2, error, real_y_test, real_y_pred, test_df = fit_model(df)

# 4. TABS
tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Model Summary", "Diagnostics", "Predict Age"])

# TAB 1
with tab1:
    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", raw_df.shape[0])
    c2.metric("Columns", raw_df.shape[1])
    c3.metric("Missing values", int(raw_df.isnull().sum().sum()))

    st.subheader("Descriptive Statistics")
    st.dataframe(raw_df.describe(), use_container_width=True)

    st.subheader("Risk Level Distribution")
    fig, ax = plt.subplots()
    counts = raw_df["RiskLevel"].value_counts()
    ax.bar(counts.index, counts.values, color=["#e74c3c", "#f39c12", "#2ecc71"])
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Count")
    ax.set_title("Risk Level Counts")
    st.pyplot(fig)
    plt.close(fig)

# TAB 2
with tab2:
    st.subheader("OLS Regression Summary")
    st.markdown("**Formula:** Age ~ SystolicBP + BS + BS2 + BodyTemp")
    st.markdown("Age is square-root transformed. Metrics are back-transformed.")

    coef_df = pd.DataFrame({
        "Coefficient": lm.params,
        "Std Error": lm.bse,
        "t-value": lm.tvalues,
        "p-value": lm.pvalues,
    }).round(4)
    st.dataframe(coef_df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("R2 (test)", f"{r2:.4f}")
    c2.metric("MAE (years)", f"{mae:.2f}")
    c3.metric("RMSE (years)", f"{rmse:.2f}")

    st.subheader("Coefficient Bar Chart")
    fig2, ax2 = plt.subplots()
    coef_vals = coef_df["Coefficient"].drop("Intercept")
    ax2.bar(coef_vals.index, coef_vals.values, color="#3498db")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("OLS Coefficients (excl. Intercept)")
    ax2.set_ylabel("Value")
    plt.xticks(rotation=15)
    st.pyplot(fig2)
    plt.close(fig2)

# TAB 3
with tab3:
    st.subheader("Residual Analysis")

    col_a, col_b = st.columns(2)

    with col_a:
        fig3, ax3 = plt.subplots()
        ax3.hist(error, bins=30, color="#9b59b6", edgecolor="white")
        ax3.axvline(np.mean(error), color="red", linestyle="--", label=f"Mean = {np.mean(error):.2f}")
        ax3.set_title("Histogram of Residuals")
        ax3.set_xlabel("Error")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        st.pyplot(fig3)
        plt.close(fig3)

    with col_b:
        fig4, ax4 = plt.subplots()
        ax4.scatter(real_y_pred, error, alpha=0.4, color="#1abc9c")
        ax4.axhline(0, color="red", linestyle="--")
        ax4.set_title("Residuals vs Predicted Age")
        ax4.set_xlabel("Predicted Age")
        ax4.set_ylabel("Residual")
        st.pyplot(fig4)
        plt.close(fig4)

    st.subheader("Actual vs Predicted Age")
    fig5, ax5 = plt.subplots()
    ax5.scatter(real_y_test, real_y_pred, alpha=0.4, color="#e67e22")
    mn = min(real_y_test.min(), real_y_pred.min())
    mx = max(real_y_test.max(), real_y_pred.max())
    ax5.plot([mn, mx], [mn, mx], "r--", label="Perfect fit")
    ax5.set_xlabel("Actual Age")
    ax5.set_ylabel("Predicted Age")
    ax5.set_title("Actual vs Predicted (test set)")
    ax5.legend()
    st.pyplot(fig5)
    plt.close(fig5)

# TAB 4
with tab4:
    st.subheader("Predict Maternal Age from Health Parameters")
    st.markdown("Enter raw unscaled values. The app scales them internally.")

    raw_stats = raw_df.describe()

    systolic = st.number_input(
        "Systolic BP (mmHg)",
        min_value=float(raw_stats.loc["min", "SystolicBP"]),
        max_value=float(raw_stats.loc["max", "SystolicBP"]),
        value=float(raw_stats.loc["50%", "SystolicBP"]),
        step=1.0,
        key="input_systolic"
    )

    bs = st.number_input(
        "Blood Sugar (mmol/L)",
        min_value=float(raw_stats.loc["min", "BS"]),
        max_value=float(raw_stats.loc["max", "BS"]),
        value=float(raw_stats.loc["50%", "BS"]),
        step=0.1,
        key="input_bs"
    )

    body_temp = st.number_input(
        "Body Temperature (F)",
        min_value=float(raw_stats.loc["min", "BodyTemp"]),
        max_value=float(raw_stats.loc["max", "BodyTemp"]),
        value=float(raw_stats.loc["50%", "BodyTemp"]),
        step=0.1,
        key="input_bodytemp"
    )

    if st.button("Predict Age", type="primary", key="predict_btn"):
        col_means = raw_df[cont_cols].mean()
        dummy_dict = {col: [float(col_means[col])] for col in cont_cols}
        dummy_dict["SystolicBP"] = [float(systolic)]
        dummy_dict["BS"] = [float(bs)]
        dummy_row = pd.DataFrame(dummy_dict)[cont_cols]

        scaled = scaler.transform(dummy_row)[0]
        syst_s = float(scaled[cont_cols.index("SystolicBP")])
        bs_s   = float(scaled[cont_cols.index("BS")])
        bs2_s  = bs_s ** 2

        pred_input = pd.DataFrame({
            "SystolicBP": [syst_s],
            "BS":         [bs_s],
            "BS2":        [bs2_s],
            "BodyTemp":   [float(body_temp)],
        })

        sqrt_age = lm.predict(pred_input)[0]
        age_pred = max(0.0, float(sqrt_age)) ** 2

        st.success(f"Predicted Maternal Age: {age_pred:.1f} years")
        st.caption("Model R2 = 0.41. Use as a statistical estimate, not a clinical decision.")
