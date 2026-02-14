import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

# Set page configuration
st.set_page_config(page_title="Medical Triage Assistant", page_icon="🏥", layout="wide")

# Session State Initialization
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Login Page Function
def login_page():
    st.title("🏥 AI Medical Triage Assistant")
    st.subheader("Secure Access Portal")
    st.write("Please sign in to access Triage Tools.")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Sign In"):
        if username == "admin" and password == "medical2026":
            st.session_state['logged_in'] = True
            st.rerun()
        else:
            st.error("Invalid credentials.")

# Main Application Logic (Protected)
if not st.session_state['logged_in']:
    login_page()
else:
    
    # Sidebar Logout
    with st.sidebar:
        st.header("Admin")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()
        st.divider()

    st.title("🏥 Medical Triage System")
    st.markdown("### ML-Powered Rapid Assessment & Department Routing")

    # Model Training and Data Loading
    @st.cache_resource
    def train_models():
        csv_path = "triage_data.csv"
        if not os.path.exists(csv_path):
            return None, None
        
        try:
            df = pd.read_csv(csv_path)
            X = df[['Age', 'Temperature', 'HeartRate']]
            y_risk = df['RiskLevel']
            y_dept = df['Department']
            
            clf_risk = RandomForestClassifier(n_estimators=100, random_state=42)
            clf_risk.fit(X, y_risk)
            
            clf_dept = RandomForestClassifier(n_estimators=100, random_state=42)
            clf_dept.fit(X, y_dept)
            
            return clf_risk, clf_dept
        except Exception as e:
            st.error(f"Error training models: {e}")
            return None, None

    # Load Models
    clf_risk, clf_dept = train_models()

    if clf_risk is None or clf_dept is None:
        st.error("⚠️ triage_data.csv not found or could not be loaded. Please ensure the dataset is in the current directory.")
    else:
        # Sidebar for Patient Vitals & History
        with st.sidebar:
            st.header("Patient Intake")
            patient_id = st.text_input("Enter Patient ID")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Other"])
            temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
            heart_rate = st.number_input("Heart Rate (BPM)", min_value=0, max_value=250, value=75)
            history = st.multiselect("Pre-existing Conditions", ["None", "Diabetes", "Hypertension", "Asthma", "Heart Disease"], default="None")
            st.divider()
            st.info("Ensure all intake data is recorded accurately.")

        # Main area for symptom description and severity
        st.subheader("Clinical Presentation")
        symptoms = st.text_area("Symptom Description", placeholder="Describe the symptoms in detail...", height=150)
        severity = st.slider("Manual Severity Level", min_value=1, max_value=10, value=5, help="1 (Minor) to 10 (Critical)")

        # Prediction Logic
        def get_prediction(age, temp, hr, history):
            input_data = pd.DataFrame([[age, temp, hr]], columns=['Age', 'Temperature', 'HeartRate'])
            risk_level = clf_risk.predict(input_data)[0]
            risk_probs = clf_risk.predict_proba(input_data)
            confidence = np.max(risk_probs) * 100
            dept = clf_dept.predict(input_data)[0]
            
            # Risk Adjustment for Pre-existing Conditions
            has_conditions = len(history) > 0 and "None" not in history
            if has_conditions:
                if risk_level == "Low":
                    risk_level = "Medium"
                    confidence = min(100.0, confidence + 5.0)
                elif risk_level == "Medium":
                    risk_level = "High"
                    confidence = min(100.0, confidence + 5.0)

            config = {
                "High": ("#dc3545", "white", "Critical indicators & medical history require immediate attention."),
                "Medium": ("#ffc107", "black", "Moderate risk factors or pre-existing conditions present. Requires timely assessment."),
                "Low": ("#28a745", "white", "Vitals appear stable according to historical data.")
            }
            bg_color, text_color, insight = config.get(risk_level, ("#6c757d", "white", "Undetermined risk level."))
            
            return risk_level, bg_color, text_color, dept, insight, confidence

        if st.button("Assess Risk"):
            if not symptoms.strip():
                st.warning("Please enter symptom description before assessment.")
            else:
                # Get Prediction
                risk, bg_color, text_color, dept, insight, confidence = get_prediction(age, temp, heart_rate, history)
                
                # Symptom-based overrides
                urgent_keywords = ["chest pain", "shortness of breath", "unconscious", "stroke", "severe bleeding"]
                symptoms_lower = symptoms.lower()
                if any(word in symptoms_lower for word in urgent_keywords):
                    risk = "High"
                    bg_color = "#dc3545"
                    text_color = "white"
                    dept = "Emergency Department (ED) - Priority 1"
                    insight = "Specific red-flag symptoms mentioned in description. Escalated to High priority regardless of vitals."
                    confidence = 100.0

                # Display Risk Result using standard Streamlit components
                st.subheader("Assessment Result")
                if risk == "High":
                    st.error(f"**Risk Level: {risk}**\n\nRecommended: {dept}\n\nPatient ID: {patient_id if patient_id else 'N/A'}")
                elif risk == "Medium":
                    st.warning(f"**Risk Level: {risk}**\n\nRecommended: {dept}\n\nPatient ID: {patient_id if patient_id else 'N/A'}")
                else:
                    st.success(f"**Risk Level: {risk}**\n\nRecommended: {dept}\n\nPatient ID: {patient_id if patient_id else 'N/A'}")
                
                st.write("") 

                # Assessment Summary Table
                st.subheader(f"📋 Assessment Summary: {patient_id if patient_id else 'Unknown Patient'}")
                summary_data = {
                    "Metric": ["Patient ID", "Pre-existing Conditions", "Predicted Risk", "Recommended Department", "AI Confidence Score"],
                    "Value": [patient_id, ", ".join(history), risk, dept, f"{confidence:.1f}%"]
                }
                st.table(pd.DataFrame(summary_data))
                
                with st.expander("🔍 AI Clinical Insight", expanded=True):
                    st.subheader("Clinical Context")
                    st.write(insight)
                    st.markdown("---")
                    st.markdown(f"**ML Baseline Confidence:** {confidence:.1f}%")
                    st.markdown(f"**Vitals Evaluated:** Age: {age}, Temp: {temp}°C, HR: {heart_rate} BPM")

    st.markdown("---")
    st.caption("⚠️ Disclaimer: This is a decision support tool. Always follow hospital protocols.")
