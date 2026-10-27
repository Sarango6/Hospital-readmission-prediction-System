
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="MediPredict | Readmission AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = (
    Path(__file__).parent
    / "artifacts"
    / "readmission_model.joblib"
)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    bundle = load_model()
    model = bundle["model"]
except Exception:
    st.error(
        "Model file not found. Please run the notebook's "
        "model-save cell and check the artifacts folder."
    )
    st.stop()


# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #102A43;
        --teal: #087F8C;
        --mint: #DDF7F0;
        --bg: #F4F8FB;
        --text: #243B53;
        --muted: #627D98;
        --border: #D9E4EC;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--navy);
    }

    .hero {
        background: linear-gradient(125deg, #102A43 0%, #145C70 65%, #087F8C 100%);
        border-radius: 24px;
        padding: 34px 38px;
        color: white;
        margin-bottom: 26px;
        position: relative;
        overflow: hidden;
    }

    .hero-tag {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 30px;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.23);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 15px;
    }

    .hero h1 {
        color: white !important;
        font-size: clamp(27px, 4vw, 39px);
        line-height: 1.2;
        margin: 0 0 12px 0;
        font-weight: 800;
    }

    .hero p {
        color: #D9EDF2;
        font-size: 15px;
        line-height: 1.8;
        margin: 0;
        max-width: 660px;
    }

    .section-heading {
        font-family: 'Manrope', sans-serif;
        font-size: 21px;
        font-weight: 800;
        color: var(--navy);
        margin: 8px 0 5px 0;
    }

    .section-description {
        color: var(--muted);
        font-size: 14px;
        margin-bottom: 20px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: white;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px;
    }

    label {
        font-weight: 600 !important;
        color: var(--text) !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 10px;
        border-color: var(--border);
        background: #FBFDFF;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(100deg, #087F8C, #10A69A);
        color: white;
        border: none;
        border-radius: 12px;
        min-height: 50px;
        font-weight: 700;
        font-size: 15px;
        width: 100%;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #066874;
        color: white;
        border: none;
        box-shadow: 0 5px 16px rgba(8,127,140,0.2);
    }

    .result-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 27px;
        margin-top: 20px;
        box-shadow: 0 8px 25px rgba(16,42,67,0.04);
    }

    .result-label {
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        color: var(--muted);
        margin-bottom: 12px;
    }

    .result-title {
        font-family: 'Manrope', sans-serif;
        font-size: 25px;
        font-weight: 800;
        line-height: 1.3;
        margin-bottom: 12px;
    }

    .result-description {
        font-size: 14px;
        line-height: 1.7;
        color: var(--muted);
    }

    .probability-number {
        font-family: 'Manrope', sans-serif;
        font-size: 35px;
        font-weight: 800;
        color: var(--navy);
        margin: 5px 0;
    }

    .probability-track {
        width: 100%;
        height: 10px;
        border-radius: 20px;
        background: #E8EFF4;
        overflow: hidden;
        margin: 14px 0;
    }

    .probability-fill {
        height: 100%;
        border-radius: 20px;
        background: linear-gradient(90deg, #087F8C, #36C7A5);
    }

    .notice {
        background: #FFF8E7;
        border: 1px solid #F4DFAC;
        color: #795716;
        padding: 15px 17px;
        border-radius: 12px;
        font-size: 13px;
        line-height: 1.7;
        margin-top: 22px;
    }

    .footer {
        border-top: 1px solid var(--border);
        padding-top: 20px;
        margin-top: 34px;
        color: var(--muted);
        text-align: center;
        font-size: 12px;
        line-height: 1.8;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }

        .hero {
            padding: 25px 22px;
            border-radius: 18px;
        }

        .result-card {
            padding: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# HERO HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <div class="hero-tag">✦ MACHINE LEARNING · HEALTHCARE</div>
    <h1>Hospital Readmission<br>Prediction System</h1>
    <p>
        An educational machine learning application that estimates
        the likelihood of early hospital readmission from selected
        patient encounter characteristics.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# PATIENT INPUTS
# =====================================================
st.markdown(
    '<div class="section-heading">Patient Information</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-description">'
    'Enter the encounter details below to generate a model prediction.'
    '</div>',
    unsafe_allow_html=True
)

with st.form("patient_form"):
    with st.container(border=True):
        st.markdown("#### 👤 Demographic Details")

        col1, col2 = st.columns(2)

        with col1:
            age = st.selectbox(
                "Age Group",
                [
                    "[0-10)", "[10-20)", "[20-30)", "[30-40)",
                    "[40-50)", "[50-60)", "[60-70)", "[70-80)",
                    "[80-90)", "[90-100)"
                ],
                index=6
            )

        with col2:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male", "Unknown/Invalid"]
            )

    with st.container(border=True):
        st.markdown("#### 🏥 Hospital Encounter")

        col3, col4 = st.columns(2)

        with col3:
            time_in_hospital = st.number_input(
                "Days in Hospital",
                min_value=1,
                max_value=14,
                value=5,
                help="Length of the hospital stay in days."
            )

            num_lab_procedures = st.number_input(
                "Laboratory Procedures",
                min_value=0,
                max_value=200,
                value=40
            )

        with col4:
            num_medications = st.number_input(
                "Number of Medications",
                min_value=0,
                max_value=100,
                value=12
            )

            number_inpatient = st.number_input(
                "Previous Inpatient Visits",
                min_value=0,
                max_value=50,
                value=1
            )

    st.markdown("")
    submitted = st.form_submit_button(
        "✦  Generate Prediction",
        use_container_width=True
    )


# =====================================================
# PREDICTION
# =====================================================
if submitted:
    patient = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_medications": num_medications,
        "number_inpatient": number_inpatient
    }])

    # Fill numerical missing values using saved training medians
    for col in bundle["numeric_columns"]:
        patient[col] = pd.to_numeric(
            patient[col], errors="coerce"
        ).fillna(bundle["numeric_medians"][col])

    # Apply the same one-hot encoding and feature order as training
    patient = pd.get_dummies(
        patient,
        columns=["age", "gender"],
        dtype=int
    )

    patient = patient.reindex(
        columns=bundle["feature_columns"],
        fill_value=0
    )

    prediction = int(model.predict(patient)[0])
    probability = float(model.predict_proba(patient)[0, 1])
    percentage = probability * 100

    st.markdown("---")
    st.markdown(
        '<div class="section-heading">Prediction Summary</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-description">'
        'Your submitted details have been processed by the trained model.'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        if prediction == 1:
            title = "Early Readmission Predicted"
            symbol = "↗"
            accent = "#B45309"
            description = (
                "The model assigned this encounter to the class "
                "associated with readmission within 30 days."
            )
        else:
            title = "No Early Readmission Predicted"
            symbol = "✓"
            accent = "#087F8C"
            description = (
                "The model assigned this encounter to the class "
                "associated with readmission after 30 days or no readmission."
            )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">MODEL OUTPUT</div>
                <div style="font-size:34px; color:{accent}; margin-bottom:10px;">
                    {symbol}
                </div>
                <div class="result-title" style="color:{accent};">
                    {title}
                </div>
                <div class="result-description">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">MODEL ESTIMATE</div>
                <div class="probability-number">{percentage:.1f}%</div>
                <div class="result-description">
                    Estimated probability for the early-readmission class
                </div>
                <div class="probability-track">
                    <div class="probability-fill"
                         style="width:{percentage:.2f}%;">
                    </div>
                </div>
                <div class="result-description">
                    Model: {bundle["model_name"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="notice">
            <strong>Important notice</strong><br>
            This application is an academic demonstration using historical
            data. Its predictions are not medical diagnoses, are not
            clinically validated, and must not be used to make patient-care
            decisions.
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    <strong>MediPredict</strong> · Hospital Readmission Prediction System<br>
    Academic Machine Learning Project · Dataset: Diabetes 130-US Hospitals
    (1999–2008)<br>
    Built for educational and demonstration purposes only.
</div>
""", unsafe_allow_html=True)
