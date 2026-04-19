import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Placement & Salary projection", layout="centered")
classifier = joblib.load("artifacts/classifier_pipeline.pkl")
regressor = joblib.load("artifacts/regressor_pipeline.pkl")

st.title("Student Career & Salary Projection")

# Data form
with st.form("student_profile"):
    st.subheader("Data Akademik & Skill")
    col1, col2 = st.columns(2)
    
    with col1:
        ssc_p = st.number_input("SSC Percentage (Sekolah Menengah)", 0.0, 100.0, 75.0)
        hsc_p = st.number_input("HSC Percentage (SMA/K)", 0.0, 100.0, 70.0)
        degree_p = st.number_input("Degree Percentage (Kuliah)", 0.0, 100.0, 72.0)
        cgpa = st.number_input("CGPA", 0.0, 10.0, 8.0)
        attendance = st.number_input("Attendance Percentage", 0.0, 100.0, 90.0)
        backlogs = st.number_input("Jumlah Backlogs", 0, 10, 0)

    with col2:
        entrance_score = st.number_input("Entrance Exam Score", 0.0, 100.0, 75.0)
        tech_skill = st.number_input("Technical Skill Score (0-100)", 0.0, 100.0, 80.0)
        soft_skill = st.number_input("Soft Skill Score (0-100)", 0.0, 100.0, 85.0)
        certif = st.number_input("Jumlah Sertifikasi", 0, 10, 2)
        intern = st.number_input("Jumlah Internship", 0, 5, 1)
        projects = st.number_input("Jumlah Live Projects", 0, 5, 1)
        work_exp = st.number_input("Work Experience (Bulan)", 0, 60, 0)

    st.subheader("Data Kategorikal")
    c3, c4= st.columns(2)
    with c3:
        gender = st.selectbox("Gender", ["M", "F"])
    with c4:
        extra = st.selectbox("Ekstrakurikuler", ["Yes", "No"])
    submit = st.form_submit_button("Analisis Profil")

if submit:
    # dataframe untuk di pass ke model
    input_data = pd.DataFrame([{
        "ssc_percentage": ssc_p,
        "hsc_percentage": hsc_p,
        "degree_percentage": degree_p,
        "cgpa": cgpa,
        "entrance_exam_score": entrance_score,
        "technical_skill_score": tech_skill,
        "soft_skill_score": soft_skill,
        "internship_count": intern,
        "live_projects": projects,
        "work_experience_months": work_exp,
        "certifications": certif,
        "attendance_percentage": attendance,
        "backlogs": backlogs,
        "gender": gender,
        "extracurricular_activities": extra,
    }])

    st.divider()
    
    # do classification
    with st.spinner("peluang placement.."):
        prediction = classifier.predict(input_data)[0]
        prob = classifier.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"### PLACED! (Peluang: {prob*100:.1f}%)")
        
        # kalo statusnya placed, estimasi gajinya
        with st.spinner("estimasi salary.."):
            salary_est = regressor.predict(input_data)[0]
        st.metric(label="Estimasi Paket Gaji", value=f"{salary_est:.2f} LPA")
    else:
        st.error(f"### NOT PLACED (Peluang: {prob*100:.1f}%)")