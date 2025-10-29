import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ========================
# App Configuration
# ========================
st.set_page_config(
    page_title="Smart Income Predictor",
    page_icon="💼",
    layout="centered",
)

st.title("💼 Smart Income Predictor")
st.markdown(
    """
    **Predict whether an individual earns more than \$50K per year**
    based on demographic and work-related information.
    
    ---
    """
)

# ========================
# Load Model + Encoder
# ========================
@st.cache_resource
def load_model():
    model = joblib.load("income_classifier.pkl")
    encoder = joblib.load("income_label_encoder.pkl")
    return model, encoder

try:
    model, label_encoder = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ========================
# User Input Section
# ========================
st.header("📋 Enter Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 80, 30)
    education = st.selectbox(
        "Education Level",
        ["Bachelors", "Masters", "HS-grad", "Doctorate", "Some-college"],
    )
    occupation = st.selectbox(
        "Occupation",
        ["Exec-managerial", "Tech-support", "Craft-repair", "Sales", "Other-service"],
    )

with col2:
    hours_per_week = st.slider("Hours Worked per Week", 1, 99, 40)
    gender = st.radio("Gender", ["Male", "Female"])
    marital_status = st.selectbox(
        "Marital Status",
        ["Never-married", "Married", "Divorced", "Separated", "Widowed"],
    )

# ========================
# Data Preparation
# ========================
user_input = pd.DataFrame(
    {
        "age": [age],
        "hours-per-week": [hours_per_week],
        "education": [education],
        "occupation": [occupation],
        "gender": [gender],
        "marital-status": [marital_status],
    }
)

# One-hot encode and align with model features
user_input = pd.get_dummies(user_input, drop_first=True)
for col in model.feature_names_in_:
    if col not in user_input.columns:
        user_input[col] = 0
user_input = user_input[model.feature_names_in_]

# ========================
# Prediction Logic
# ========================
if st.button("🔮 Predict Income"):
    pred = model.predict(user_input)[0]
    proba = model.predict_proba(user_input)[0]
    result = label_encoder.inverse_transform([pred])[0]

    st.subheader("🧾 Prediction Result")
    if result == ">50K":
        st.success("💰 Predicted Income Category: **>50K / year**")
    else:
        st.info("📉 Predicted Income Category: **<=50K / year**")

    # Confidence bar chart
    fig = go.Figure(
        go.Bar(
            x=label_encoder.inverse_transform([0, 1]),
            y=proba,
            text=[f"{p:.2%}" for p in proba],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Income Category",
        yaxis_title="Probability",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ========================
# Footer
# ========================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, scikit-learn, and Plotly.")
