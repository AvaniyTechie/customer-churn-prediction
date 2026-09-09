import streamlit as st
import pandas as pd
import joblib
from google import genai


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Customer Churn Intelligence",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")


model = load_model()


# =========================================================
# LOAD GEMINI
# =========================================================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


# =========================================================
# FUNCTION: GET INDIVIDUAL FEATURE CONTRIBUTIONS
# =========================================================

def get_feature_contributions(model, customer):

    # Get preprocessing pipeline
    preprocessor = model.named_steps["preprocessor"]

    # Get Logistic Regression model
    lr_model = model.named_steps["model"]

    # Transform customer using the SAME preprocessing
    # used during model training
    transformed_customer = preprocessor.transform(customer)

    # Get feature names after encoding/scaling
    feature_names = preprocessor.get_feature_names_out()

    # Get Logistic Regression coefficients
    coefficients = lr_model.coef_[0]

    # Convert sparse matrix to normal array if necessary
    if hasattr(transformed_customer, "toarray"):
        transformed_customer = transformed_customer.toarray()

    # Calculate individual feature contribution
    contributions = transformed_customer[0] * coefficients

    # Create dataframe
    contribution_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions
    })

    # Absolute contribution tells us the strength
    contribution_df["Absolute Contribution"] = (
        contribution_df["Contribution"].abs()
    )

    # Sort strongest contributions first
    contribution_df = contribution_df.sort_values(
        by="Absolute Contribution",
        ascending=False
    ).reset_index(drop=True)

    return contribution_df


# =========================================================
# FUNCTION: CLEAN FEATURE NAMES
# =========================================================

def clean_feature_name(feature):

    feature = feature.replace("num__", "")
    feature = feature.replace("cat__", "")
    feature = feature.replace("_", " ")

    return feature


# =========================================================
# HEADER
# =========================================================

st.title("🤖 AI Customer Churn Intelligence")

st.caption(
    "Machine Learning prediction + Generative AI retention recommendations"
)

st.divider()


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)


# ---------------------------------------------------------
# COLUMN 1
# ---------------------------------------------------------

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    Partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )


# ---------------------------------------------------------
# COLUMN 2
# ---------------------------------------------------------

with col2:

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    InternetService = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


# ---------------------------------------------------------
# COLUMN 3
# ---------------------------------------------------------

with col3:

    DeviceProtection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    StreamingTV = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


st.divider()


# =========================================================
# CONTRACT & BILLING
# =========================================================

st.header("💳 Contract & Billing")

col1, col2, col3 = st.columns(3)


with col1:

    Contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )


with col2:

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


with col3:

    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


col1, col2 = st.columns(2)


with col1:

    MonthlyCharges = st.number_input(
        "Monthly Charges ($)",
        min_value=18.0,
        max_value=120.0,
        value=70.0,
        step=1.0
    )


with col2:

    TotalCharges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=1000.0,
        step=10.0
    )


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

predict_button = st.button(
    "🔮 Analyze Customer Churn Risk",
    use_container_width=True,
    type="primary"
)


if predict_button:

    # =====================================================
    # CREATE CUSTOMER DATAFRAME
    # =====================================================

    new_customer = pd.DataFrame([{

        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges

    }])

    # =====================================================
    # MACHINE LEARNING PREDICTION
    # =====================================================

    prediction = model.predict(new_customer)

    probability = model.predict_proba(new_customer)

    churn_probability = probability[0][1] * 100

    # =====================================================
    # RISK LEVEL
    # =====================================================

    if churn_probability < 30:

        risk = "Low Risk"

    elif churn_probability < 60:

        risk = "Medium Risk"

    else:

        risk = "High Risk"

    # =====================================================
    # ACTUAL MODEL FEATURE CONTRIBUTIONS
    # =====================================================

    contribution_df = get_feature_contributions(
        model,
        new_customer
    )

    # Clean names for display
    contribution_df["Display Feature"] = (
        contribution_df["Feature"]
        .apply(clean_feature_name)
    )

    # -----------------------------------------------------
    # POSITIVE CONTRIBUTIONS
    # -----------------------------------------------------
    # Positive = pushes prediction toward churn

    churn_drivers = contribution_df[
        contribution_df["Contribution"] > 0
    ].copy()

    # -----------------------------------------------------
    # NEGATIVE CONTRIBUTIONS
    # -----------------------------------------------------
    # Negative = pushes prediction toward staying

    stay_drivers = contribution_df[
        contribution_df["Contribution"] < 0
    ].copy()

    # Strongest churn drivers
    top_churn_drivers = churn_drivers.head(5)

    # Strongest protective factors
    top_stay_drivers = (
        stay_drivers
        .sort_values(
            by="Absolute Contribution",
            ascending=False
        )
        .head(5)
    )

    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.divider()

    st.header("📈 Churn Risk Assessment")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )

    with metric2:

        st.metric(
            "Risk Level",
            risk
        )

    with metric3:

        prediction_text = (
            "Likely to Churn"
            if prediction[0] == "Yes"
            else "Likely to Stay"
        )

        st.metric(
            "ML Prediction",
            prediction_text
        )

    # =====================================================
    # PROBABILITY BAR
    # =====================================================

    st.subheader("Churn Probability")

    st.progress(
        min(int(churn_probability), 100)
    )

    if risk == "High Risk":

        st.error(
            f"⚠️ High churn risk — estimated probability: "
            f"{churn_probability:.2f}%"
        )

    elif risk == "Medium Risk":

        st.warning(
            f"⚠️ Medium churn risk — estimated probability: "
            f"{churn_probability:.2f}%"
        )

    else:

        st.success(
            f"✅ Low churn risk — estimated probability: "
            f"{churn_probability:.2f}%"
        )

    st.caption(
        "The probability is an estimate produced by the trained "
        "machine learning model and should not be interpreted as certainty."
    )

    # =====================================================
    # CUSTOMER PROFILE
    # =====================================================

    st.divider()

    st.header("👤 Customer Profile")

    profile1, profile2, profile3, profile4 = st.columns(4)

    with profile1:

        st.metric(
            "Tenure",
            f"{tenure} months"
        )

    with profile2:

        st.metric(
            "Monthly Charges",
            f"${MonthlyCharges:.2f}"
        )

    with profile3:

        st.metric(
            "Contract",
            Contract
        )

    with profile4:

        st.metric(
            "Internet",
            InternetService
        )

    # =====================================================
    # MODEL-BASED EXPLANATION
    # =====================================================

    st.divider()

    st.header("🔍 Why Did the Model Predict This?")

    st.caption(
        "These factors are calculated from the actual "
        "Logistic Regression model for this customer."
    )

    # =====================================================
    # FEATURE CONTRIBUTION VISUALIZATION
    # =====================================================

    st.subheader("📊 Model Contribution Overview")

    st.caption(
        "Positive values push the model toward churn, "
        "while negative values push the model toward staying."
    )

    # Select the 10 strongest contributors
    chart_df = contribution_df.head(10).copy()

    # Sort for easier reading
    chart_df = chart_df.sort_values(
        by="Contribution",
        ascending=True
    )

    # Prepare chart data
    chart_data = chart_df[
        ["Display Feature", "Contribution"]
    ].copy()

    chart_data = chart_data.set_index(
        "Display Feature"
    )

    # Display chart
    st.bar_chart(
        chart_data,
        horizontal=True
    )

    st.divider()

    # =====================================================
    # DRIVER COLUMNS
    # =====================================================

    driver_col1, driver_col2 = st.columns(2)

    # -----------------------------------------------------
    # CHURN DRIVERS
    # -----------------------------------------------------

    with driver_col1:

        st.subheader("🔴 Factors Pushing Toward Churn")

        if len(top_churn_drivers) > 0:

            for _, row in top_churn_drivers.iterrows():

                st.markdown(
                    f"**{row['Display Feature']}**"
                )

                st.caption(
                    f"Model contribution: "
                    f"+{row['Contribution']:.3f}"
                )

                st.divider()

        else:

            st.success(
                "No strong factors pushing the prediction toward churn."
            )

    # -----------------------------------------------------
    # STAY DRIVERS
    # -----------------------------------------------------

    with driver_col2:

        st.subheader("🟢 Factors Pushing Toward Stay")

        if len(top_stay_drivers) > 0:

            for _, row in top_stay_drivers.iterrows():

                st.markdown(
                    f"**{row['Display Feature']}**"
                )

                st.caption(
                    f"Model contribution: "
                    f"{row['Contribution']:.3f}"
                )

                st.divider()

        else:

            st.info(
                "No strong factors pushing the prediction toward staying."
            )

    # =====================================================
    # PREPARE MODEL DRIVERS FOR GEMINI
    # =====================================================

    actual_churn_drivers = "\n".join(
        [
            f"- {row['Display Feature']} "
            f"(model contribution: {row['Contribution']:+.3f})"
            for _, row in top_churn_drivers.iterrows()
        ]
    )

    actual_stay_drivers = "\n".join(
        [
            f"- {row['Display Feature']} "
            f"(model contribution: {row['Contribution']:+.3f})"
            for _, row in top_stay_drivers.iterrows()
        ]
    )

    if not actual_churn_drivers:

        actual_churn_drivers = "- None identified"

    if not actual_stay_drivers:

        actual_stay_drivers = "- None identified"

    # =====================================================
    # CUSTOMER CONTEXT FOR GEMINI
    # =====================================================

    customer_context = f"""

Customer Profile:

Senior Citizen: {"Yes" if SeniorCitizen == 1 else "No"}
Partner: {Partner}
Dependents: {Dependents}

Tenure: {tenure} months

Phone Service: {PhoneService}
Multiple Lines: {MultipleLines}
Internet Service: {InternetService}

Online Security: {OnlineSecurity}
Online Backup: {OnlineBackup}
Device Protection: {DeviceProtection}
Tech Support: {TechSupport}

Streaming TV: {StreamingTV}
Streaming Movies: {StreamingMovies}

Contract: {Contract}
Paperless Billing: {PaperlessBilling}
Payment Method: {PaymentMethod}

Monthly Charges: ${MonthlyCharges:.2f}
Total Charges: ${TotalCharges:.2f}


Machine Learning Assessment:

Churn Probability: {churn_probability:.2f}%
ML Prediction: {prediction[0]}
Risk Level: {risk}


ACTUAL MODEL CONTRIBUTIONS INCREASING CHURN RISK:

{actual_churn_drivers}


ACTUAL MODEL CONTRIBUTIONS REDUCING CHURN RISK:

{actual_stay_drivers}
"""

    # =====================================================
    # GEMINI PROMPT
    # =====================================================

    prompt = f"""
You are an AI customer-retention analyst assisting a telecom business.

Analyze the customer using ONLY the information explicitly provided below.

CUSTOMER INFORMATION:
{customer_context}

ML PREDICTION:
- Churn probability: {churn_probability:.2f}%
- Risk level: {risk}
- Prediction: {prediction[0]}

FACTORS PUSHING TOWARD CHURN:
{actual_churn_drivers}

FACTORS PUSHING TOWARD STAY:
{actual_stay_drivers}

IMPORTANT RULES:
1. Do NOT invent or assume customer information.
2. Do NOT infer any information that is not explicitly provided.
3. Always refer to the person as "the customer".
4. Do not introduce payment methods, services, contracts, or customer characteristics that are not present in the customer information.
5. Base the explanation primarily on the actual ML model contributions provided above.
6. Do not claim that a feature causes churn. Say that it contributes to the model's prediction.
7. Recommendations must be practical and relevant to the customer's actual profile.
8. Do not recommend unnecessary discounts to low-risk customers.
9. Do not contradict the ML prediction.
10. Do not make up numerical values.
11. Keep the response concise and professional.
12. If a recommendation requires information that is not available, clearly state that it would require additional business information.

Return exactly these four sections:

### 1. Churn Risk Explanation
Explain the predicted risk using the probability and the strongest model contributions.

### 2. Key Risk Drivers
List the 3–5 strongest factors pushing toward churn.
Include their exact model contributions.

### 3. Recommended Retention Actions
Give 2–3 practical business actions based only on the available customer information.

### 4. Personalized Retention Strategy
Create one concise retention strategy based primarily on the customer's strongest churn-risk factors and current services. Prioritize practical actions related to tenure, contract, internet service, payment method, and subscribed services when relevant.

Remember:
This is a probabilistic ML prediction, not a certainty.
Do not invent information.
"""

    # =====================================================
    # GEMINI RESPONSE
    # =====================================================

    st.divider()

    st.header("🤖 AI Retention Recommendations")

    st.caption(
        "Gemini converts the ML model's customer-specific "
        "insights into business-focused retention actions."
    )

    with st.spinner(
        "🤖 AI is analyzing the customer and generating "
        "retention recommendations..."
    ):

        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            ai_recommendation = response.text

            # Display Gemini response
            st.markdown(
                ai_recommendation
            )

        except Exception as e:

            st.error(
                "Unable to generate AI recommendations."
            )

            st.caption(
                "Please check your Gemini API configuration."
            )

    # =====================================================
    # HOW THE SYSTEM WORKS
    # =====================================================

    with st.expander(
        "🔍 How was this prediction and recommendation generated?"
    ):

        st.write(
            "1️⃣ Customer information is entered into the application."
        )

        st.write(
            "2️⃣ The trained Logistic Regression pipeline "
            "calculates the churn probability."
        )

        st.write(
            "3️⃣ The application calculates individual feature "
            "contributions using the transformed customer values "
            "and the model's coefficients."
        )

        st.write(
            "4️⃣ Positive contributions indicate factors pushing "
            "the model toward churn, while negative contributions "
            "push it toward staying."
        )

        st.write(
            "5️⃣ These model-derived insights are sent to Gemini."
        )

        st.write(
            "6️⃣ Gemini generates a human-readable explanation "
            "and personalized retention recommendations."
        )

    # =====================================================
    # FOOTER
    # =====================================================

    st.divider()

    st.caption(
        "⚠️ Decision-support tool only. Churn predictions are "
        "probabilistic estimates and should be validated using "
        "business context before taking customer action."
    )
