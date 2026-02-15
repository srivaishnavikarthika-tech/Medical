AI Medical Triage Assistant
A professional, machine-learning-powered web application designed to assist healthcare providers in prioritizing patients based on clinical urgency. This tool uses a Random Forest Classifier to analyze patient vitals and medical history to provide instant risk assessments.

🚀 Overview
In high-pressure clinical environments, rapid and accurate triage is critical. Our AI Medical Triage Assistant automates the initial screening process, ensuring that high-risk patients are identified and routed to the correct department (e.g., Cardiology, General Medicine) within seconds.

✨ Key Features
Secure Access: Custom-built login and registration system using st.session_state and a persistent local user database.

Intelligent Triage: Predicts Risk Level (Low/Medium/High) and Recommended Department using a Scikit-learn model trained on clinical data.

Comprehensive Patient Profiles: Supports manual entry of Patient IDs and Pre-existing Conditions (Diabetes, Heart Disease, etc.) to refine AI accuracy.

Medical Document Support: Integrated file uploader for health records and clinical notes.

Professional UI/UX: Styled with custom CSS, featuring interactive input highlighting, a centered login portal, and dynamic visual alerts (Red/Yellow/Green) based on patient risk.

🛠️ Tech Stack
Frontend: Streamlit (Python-based Web Framework)

Machine Learning: Scikit-learn (Random Forest Classification)

Data Handling: Pandas & NumPy

Styling: Custom CSS & Markdown

Data Persistence: CSV-based local storage for triage datasets and user credentials.

📊 How It Works (The Logic)
Data Processing: The app loads a 150-row clinical dataset (triage_data.csv).

Feature Engineering: It analyzes features like Age, Body Temperature, Heart Rate, and Comorbidities.

Inference: When a nurse clicks "Assess Risk," the inputs are fed into the trained ML model.

Risk Weighting: The logic specifically weights pre-existing conditions (e.g., a patient with Heart Disease and high HR is automatically prioritized).

Output: Generates a clinical insight report with a confidence score and department routing.

This application is a Hackathon Prototype. In a production environment, user passwords should be hashed (e.g., using bcrypt), and the system should be deployed on HIPAA-compliant cloud infrastructure.
