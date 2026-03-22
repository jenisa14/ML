import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")

# Load model
@st.cache_resource
def load_model():
    model_columns = joblib.load("model_columns.pkl")
    lr_l1 = joblib.load("fraud_detection_model.pkl")
    return model_columns, lr_l1

model_columns, lr_l1 = load_model()

# Title
st.title("🔍 Insurance Fraud Detection System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Project Info")
    st.metric("Dataset Size", "12,002")
    st.metric("After Cleaning", "5,801")
    st.metric("Best Model Accuracy", "77.38%")
    st.info("Model: L1 Regularized Logistic Regression (LASSO)")

# Tabs
tab1, tab2 = st.tabs(["🔮 Fraud Prediction", "📈 Model Info"])

with tab1:
    st.header("Enter Claim Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        claim_number = st.number_input("Claim Number", value=700123456)
        age_of_driver = st.number_input("Age of Driver", value=38)
        gender = st.selectbox("Gender", ["M", "F"])
        marital_status = st.selectbox("Marital Status", [0, 1])
        safety_rating = st.number_input("Safety Rating", value=82)
        annual_income = st.number_input("Annual Income", value=62000.0)
    
    with col2:
        high_education = st.selectbox("High Education", [0, 1])
        address_change = st.selectbox("Address Change", [0, 1])
        property_status = st.selectbox("Property Status", ["Own", "Rent"])
        zip_code = st.number_input("Zip Code", value=50048)
        vehicle_category = st.selectbox("Vehicle Category", ["Small", "Medium", "Large", "Luxury"])
        vehicle_price = st.number_input("Vehicle Price", value=24500.0)
    
    with col3:
        vehicle_color = st.text_input("Vehicle Color", value="black")
        total_claim = st.number_input("Total Claim", value=27000.0)
        injury_claim = st.number_input("Injury Claim", value=5200.0)
        policy_deductible = st.number_input("Policy Deductible", value=1000)
        annual_premium = st.number_input("Annual Premium", value=1350.0)
        days_open = st.number_input("Days Open", value=9.5)
        form_defects = st.number_input("Form Defects", value=3)
    
    if st.button("🔍 Predict Fraud", type="primary", use_container_width=True):
        # Prepare data
        new_data = pd.DataFrame([{
            "claim_number": claim_number,
            "age_of_driver": age_of_driver,
            "gender": gender,
            "marital_status": marital_status,
            "safety_rating": safety_rating,
            "annual_income": annual_income,
            "high_education": high_education,
            "address_change": address_change,
            "property_status": property_status,
            "zip_code": zip_code,
            "vehicle_category": vehicle_category,
            "vehicle_price": vehicle_price,
            "vehicle_color": vehicle_color,
            "total_claim": total_claim,
            "injury_claim": injury_claim,
            "policy deductible": policy_deductible,
            "annual premium": annual_premium,
            "days open": days_open,
            "form defects": form_defects
        }])
        
        # Encode and predict
        new_data_encoded = pd.get_dummies(new_data)
        new_data_encoded = new_data_encoded.reindex(columns=model_columns, fill_value=0)
        prediction = lr_l1.predict(new_data_encoded)
        probability = lr_l1.predict_proba(new_data_encoded)
        
        result = "Fraud" if prediction[0] == 1 else "Not Fraud"
        fraud_prob = probability[0][1] * 100
        
        # Display result
        st.markdown("---")
        st.header("Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if result == "Fraud":
                st.error(f"⚠️ **{result}**")
            else:
                st.success(f"✅ **{result}**")
        
        with col2:
            st.metric("Fraud Probability", f"{fraud_prob:.1f}%")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob,
            title={'text': "Fraud Risk Score"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "red" if fraud_prob > 50 else "green"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgreen"},
                       {'range': [50, 100], 'color': "lightcoral"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}}))
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Algorithms Used")
        models_df = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "L1 (LASSO)", "L2 (RIDGE)"],
            "Accuracy": [75.39, 75.47, 77.38, 75.39]
        })
        st.dataframe(models_df, use_container_width=True)
    
    with col2:
        st.subheader("Dataset Info")
        st.write("**Original Shape:** (12002, 29)")
        st.write("**After Outlier Removal:** (5801, 29)")
        st.write("**Records Removed:** 6,201 (51.7%)")
        st.write("**Features Selected by L1:** 12")