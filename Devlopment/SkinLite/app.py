import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from fpdf import FPDF, XPos, YPos
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# PAGE CONFIGURATION & THEME
# =============================================================================
st.set_page_config(
    page_title="SkinLite AI - Clinical Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom UI CSS Framework
custom_css = """
<style>
    /* Hide default Streamlit footer and main menu, keep top header active */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Keep Streamlit sidebar collapse button visible */
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        display: block !important;
    }
    
    /* Medical Gradient Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #3B82F6 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        color: #FFFFFF !important;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.25);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF !important;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        color: #93C5FD !important;
        margin-top: 0.4rem;
    }
    
    /* Risk Badge Mapping */
    .badge {
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .badge-high { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
    .badge-medium { background-color: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
    .badge-low { background-color: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# =============================================================================
# CONSTANTS & MEDICAL REFERENCE DATABASES
# =============================================================================
MODEL_PATH = "best_densenet121.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Benign Tumors", "Skin Cancer", "Unknown Normal", "Vascular Tumors"]

DISEASE_INFO = {
    "Benign Tumors": {
        "Description": "Non-cancerous skin growths or benign spots, such as seborrheic keratoses or benign nevi.",
        "Symptoms": "Stable, slow-growing spots with regular shapes, distinct borders, and uniform color."
    },
    "Skin Cancer": {
        "Description": "Malignant skin lesions requiring immediate professional dermatological evaluation.",
        "Symptoms": "Irregular borders, changing colors, asymmetrical shapes, diameter >6mm, or non-healing sores."
    },
    "Unknown Normal": {
        "Description": "Healthy skin areas, typical beauty marks, or non-diseased dermatological variations.",
        "Symptoms": "Common skin markings showing zero signs of active disease, inflammation, or irritation."
    },
    "Vascular Tumors": {
        "Description": "Growths formed by clusters of blood vessels, such as hemangiomas or cherry angiomas.",
        "Symptoms": "Red, blue, or deep purple raised bumps or spots on the skin surface."
    }
}

if "history" not in st.session_state:
    st.session_state["history"] = []

# Helper Function: Clean Unicode for Standard PDF Latin-1 Compatibility
def sanitize_text(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "•": "* ",
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "\xa0": " ",
        "**": "",
        "###": "",
        "##": "",
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text.encode("latin-1", "replace").decode("latin-1")

# =============================================================================
# ADVANCED PDF REPORT ENGINE
# =============================================================================
class AdvancedClinicalReport(FPDF):
    def header(self):
        # Professional Primary Top Banner
        self.set_fill_color(15, 23, 42) # Dark Slate Blue
        self.rect(0, 0, 210, 32, "F")
        
        # Decorative Blue Accent Line
        self.set_fill_color(37, 99, 235)
        self.rect(0, 32, 210, 3, "F")
        
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 18)
        self.set_xy(12, 8)
        self.cell(0, 10, "SKINLITE CLINICAL AI ASSESSMENT", new_x=XPos.RIGHT, new_y=YPos.TOP)
        
        self.set_font("Helvetica", "", 9)
        self.set_xy(12, 18)
        self.cell(0, 8, "Automated Dermatological Image Analysis & Triage System", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(12)

    def footer(self):
        self.set_y(-22)
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y(), 200, self.get_y())
        
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        footer_text = sanitize_text("CONFIDENTIAL MEDICAL AI EVALUATION REPORT * NOT FOR DIRECT CLINICAL DIAGNOSIS")
        self.cell(0, 5, footer_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.cell(0, 5, f"Page {self.page_no()}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

def build_advanced_pdf(patient_info, filename, pred, conf, probs, info, ai_text):
    try:
        pdf = AdvancedClinicalReport()
        pdf.set_auto_page_break(auto=True, margin=25)
        pdf.add_page()
        
        # ---------------------------------------------------------------------
        # SECTION 1: METADATA & PATIENT PROFILE TABLE
        # ---------------------------------------------------------------------
        report_id = f"SL-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"
        current_time = datetime.now().strftime("%B %d, %Y - %I:%M %p")
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 6, "1. PATIENT & SCREENING METADATA", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
        
        # Table Header Styling
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(241, 245, 249)
        pdf.set_draw_color(203, 213, 225)
        
        # Patient Data Rows
        pdf.cell(47, 6, " Patient Name", 1, new_x=XPos.RIGHT, new_y=YPos.TOP, fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(48, 6, f" {sanitize_text(patient_info['Name'])}", 1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(47, 6, " Assessment ID", 1, new_x=XPos.RIGHT, new_y=YPos.TOP, fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(48, 6, f" {report_id}", 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(47, 6, " Age / Gender", 1, new_x=XPos.RIGHT, new_y=YPos.TOP, fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(48, 6, f" {patient_info['Age']} yrs / {sanitize_text(patient_info['Gender'])}", 1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(47, 6, " Generated On", 1, new_x=XPos.RIGHT, new_y=YPos.TOP, fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(48, 6, f" {current_time}", 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(47, 6, " Lesion Location", 1, new_x=XPos.RIGHT, new_y=YPos.TOP, fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(48, 6, f" {sanitize_text(patient_info['Location'])}", 1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(47, 6, " Image Source File", 1, new_x=XPos.RIGHT, new_y=YPos.TOP, fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(48, 6, f" {sanitize_text(filename)}", 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        if patient_info.get("Notes"):
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(47, 6, " Clinical Notes", 1, new_x=XPos.RIGHT, new_y=YPos.TOP, fill=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(143, 6, f" {sanitize_text(patient_info['Notes'])}", 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
        pdf.ln(6)

        # ---------------------------------------------------------------------
        # SECTION 2: AI SCREENING RESULTS & PROBABILITY BREAKDOWN
        # ---------------------------------------------------------------------
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 6, "2. PRIMARY CLASSIFICATION SUMMARY", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
        
        # Result Box Banner
        pdf.set_fill_color(239, 246, 255)
        pdf.set_draw_color(191, 219, 254)
        pdf.rect(10, pdf.get_y(), 190, 16, "DF")
        
        pdf.set_xy(14, pdf.get_y() + 3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(90, 5, f"Detected Condition: {sanitize_text(pred).upper()}", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(90, 5, f"Confidence Score: {conf:.2f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
        pdf.ln(10)
        
        # Class Probability Table
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(110, 6, " Condition Category", 1, new_x=XPos.RIGHT, new_y=YPos.TOP, fill=True)
        pdf.cell(80, 6, " Probability Confidence", 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C", fill=True)
        
        pdf.set_font("Helvetica", "", 9)
        for class_name, prob in probs.items():
            pdf.cell(110, 6, f"  {sanitize_text(class_name)}", 1, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.cell(80, 6, f"{prob:.2f}%", 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            
        pdf.ln(6)

        # ---------------------------------------------------------------------
        # SECTION 3: PATHOLOGY & SYMPTOM PROFILE
        # ---------------------------------------------------------------------
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 6, "3. PATHOLOGY CONTEXT & CLINICAL OVERVIEW", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
        
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(51, 65, 85)
        pathology_text = sanitize_text(f"Description: {info['Description']}\nTypical Symptoms: {info.get('Symptoms', 'N/A')}")
        pdf.multi_cell(0, 5, pathology_text)
        pdf.ln(6)

        # ---------------------------------------------------------------------
        # SECTION 4: AI CONSULTATION & GUIDANCE
        # ---------------------------------------------------------------------
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 6, "4. AI TRIAGE RECOMMENDATIONS & ADVICE", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
        
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(51, 65, 85)
        clean_ai_text = sanitize_text(ai_text)
        pdf.multi_cell(0, 5, clean_ai_text)
        pdf.ln(8)

        # ---------------------------------------------------------------------
        # SECTION 5: CLINICAL DISCLAIMER
        # ---------------------------------------------------------------------
        pdf.set_fill_color(254, 242, 242)
        pdf.set_draw_color(254, 202, 202)
        pdf.rect(10, pdf.get_y(), 190, 18, "DF")
        
        pdf.set_xy(12, pdf.get_y() + 2)
        pdf.set_text_color(153, 27, 27)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(0, 4, "IMPORTANT MEDICAL DISCLAIMER:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 7.5)
        disclaimer_msg = sanitize_text("This screening report is generated by a computer vision model for educational and preliminary assessment purposes only. It does not constitute a formal diagnosis, medical advice, or clinical path. Always consult a licensed dermatologist for professional lesion evaluation.")
        pdf.multi_cell(0, 3.5, disclaimer_msg)
        
        return bytes(pdf.output())
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# =============================================================================
# MACHINE LEARNING ENGINE & PIPELINES
# =============================================================================
@st.cache_resource
def load_nn_classifier():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return load_model(MODEL_PATH, compile=False)
    except Exception:
        return None

def process_raw_image(image: Image.Image):
    img = image.convert("RGB").resize(IMG_SIZE)
    img_arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_arr, axis=0)

def calculate_risk(disease, confidence):
    if disease == "Skin Cancer":
        return "High" if confidence > 0.50 else "Medium"
    elif disease in ["Benign Tumors", "Vascular Tumors"]:
        return "Medium" if confidence > 0.75 else "Low"
    return "Low"

def fetch_groq_summary(disease, confidence, risk):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ **Groq API Configuration Notice:** API Key not found in environment registry file."
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a helpful, simple health assistant for a school project interface. "
                        "CRITICAL RULES:\n"
                        "- Do NOT suggest clinical treatments, drugs, or surgeries.\n"
                        "- Clearly declare if the condition is dangerous or safe.\n"
                        "- If the risk profile is High or Medium, state plainly that the user must consult a doctor.\n"
                        "- If the risk profile is Low or Benign, tell them it is safe and there is no urgent need for a doctor.\n"
                        "- Present your analysis ONLY in basic bullet points under two simple headings: 'What to Do' and 'What Not to Do'."
                    )
                },
                {
                    "role": "user", 
                    "content": f"The vision system detected '{disease}' with an automated risk tier of '{risk}'. Please provide simple advice following your rules."
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=350
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ **Groq Hub Connection Fault:** {e}"

# =============================================================================
# INTERFACE CONTROL TERMINAL
# =============================================================================
with st.sidebar:
    st.markdown("<h2 style='color:#3182ce; text-align:center;'>SkinLite AI</h2>", unsafe_allow_html=True)
    view = st.radio("Navigation Matrix", ["🏠 Home Portal", "📤 Analysis Core", "📖 Condition Library", "📊 Activity Logs"])
    st.markdown("---")
    st.write("**Core Backbone:** DenseNet121")

classifier_model = load_nn_classifier()

# =============================================================================
# VIEW 1: HOME PANEL
# =============================================================================
if view == "🏠 Home Portal":
    st.markdown("""
    <div class='hero-container'>
        <div class='hero-title'>SkinLite AI Dashboard</div>
        <div class='hero-subtitle'>Clinical Neural Assistance for Cutaneous Lesion Categorization</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Operational Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Model Optimization Target", value="83.67%")
    m2.metric(label="Registered Conditions", value="4 Core Classes")
    m3.metric(label="Neural Spine Structure", value="DenseNet121 Base")

# =============================================================================
# VIEW 2: PREDICT PANEL
# =============================================================================
elif view == "📤 Analysis Core":
    st.markdown("## 📤 Medical Image Categorization Engine")
    
    if classifier_model is None:
        st.error(f"🚨 **Model Weight Exception:** The file `{MODEL_PATH}` could not be resolved from your local path.")
    else:
        st.markdown("### 📋 Patient Assessment Intake")
        
        # Intake Form Columns
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            p_name = st.text_input("Patient Full Name", value="John Doe")
            p_age = st.number_input("Patient Age", min_value=1, max_value=120, value=35)
        with col_p2:
            p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            p_loc = st.selectbox("Lesion Location", ["Arm / Hand", "Leg / Foot", "Torso / Back", "Head / Neck", "Other"])
        with col_p3:
            p_notes = st.text_area("Patient History / Notes", value="No prior history of skin lesions.", height=108)
            
        st.markdown("---")
        uploaded_img = st.file_uploader("Upload dermatoscopic image files (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
        
        if uploaded_img:
            c1, c2 = st.columns([1, 1.5])
            img_open = Image.open(uploaded_img)
            
            with c1:
                st.image(img_open, caption="Target Image Profile", width="stretch")
                trigger_analysis = st.button("Initialize Machine Screening", type="primary", width="stretch")
            
            with c2:
                if trigger_analysis:
                    with st.spinner("Processing image matrix through neural pipelines..."):
                        tensor_input = process_raw_image(img_open)
                        raw_out = classifier_model.predict(tensor_input)[0]
                        
                        top_id = np.argmax(raw_out)
                        selected_disease = CLASS_NAMES[top_id]
                        confidence_score = float(raw_out[top_id])
                        risk_level = calculate_risk(selected_disease, confidence_score)
                        
                        probability_dict = {CLASS_NAMES[i]: float(raw_out[i] * 100) for i in range(len(CLASS_NAMES))}
                    
                    st.markdown("### Primary Screening Classification")
                    st.subheader(f"{selected_disease}")
                    st.info(f"**Classification Confidence:** {confidence_score*100:.2f}%")
                    
                    # Risk Badges
                    if risk_level == "High":
                        st.markdown("<span class='badge badge-high'>🚨 High Risk Profile</span>", unsafe_allow_html=True)
                    elif risk_level == "Medium":
                        st.markdown("<span class='badge badge-medium'>⚠️ Moderate Risk Profile</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='badge badge-low'>✅ Low Risk Profile</span>", unsafe_allow_html=True)
                    
                    # Distribution Metrics
                    st.markdown("#### Probability Vector Trackers")
                    for condition, score in probability_dict.items():
                        st.write(f"**{condition}** — {score:.1f}%")
                        st.progress(score / 100.0)
                        
                    # Cache session record parameters
                    st.session_state["history"].append({
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Patient Name": p_name,
                        "Condition Result": selected_disease,
                        "Confidence Score": f"{confidence_score*100:.1f}%",
                        "Risk Level": risk_level
                    })
                    
                    # Render Groq Output cleanly via markdown
                    st.markdown("---")
                    st.markdown("### 🤖 Live AI Consultation Reference Insights")
                    ai_response = fetch_groq_summary(selected_disease, confidence_score * 100, risk_level)
                    st.markdown(ai_response)
                    
                    # Safe dictionary fallback mapping
                    fallback_info = DISEASE_INFO.get(selected_disease, {"Description": "Skin condition data description node.", "Symptoms": "N/A"})
                    
                    # Package Patient Profile Dictionary
                    patient_profile = {
                        "Name": p_name,
                        "Age": p_age,
                        "Gender": p_gender,
                        "Location": p_loc,
                        "Notes": p_notes
                    }
                    
                    # Document Generation Block
                    pdf_output = build_advanced_pdf(
                        patient_profile, 
                        uploaded_img.name, 
                        selected_disease, 
                        confidence_score*100, 
                        probability_dict, 
                        fallback_info, 
                        ai_response
                    )
                    
                    if pdf_output:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.download_button(
                            "📥 Download Clinical Assessment PDF Report",
                            data=pdf_output,
                            file_name=f"SkinLite_Report_{p_name.replace(' ', '_')}.pdf",
                            mime="application/pdf",
                            width="stretch"
                        )

# =============================================================================
# VIEW 3: DISEASE REFERENCE LIBRARY
# =============================================================================
elif view == "📖 Condition Library":
    st.markdown("## 📖 Cutaneous Condition Encyclopedia")
    for condition_key, data in DISEASE_INFO.items():
        with st.expander(f"Medical Reference Profile: {condition_key}"):
            st.markdown(f"**Pathology Description:** {data['Description']}")
            st.markdown(f"**Symptoms:** {data['Symptoms']}")

# =============================================================================
# VIEW 4: LOGS HISTORY
# =============================================================================
elif view == "📊 Activity Logs":
    st.markdown("## 📊 Session Audit Trail Logs")
    if not st.session_state["history"]:
        st.info("No inference instances captured in this container segment yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state["history"]), width="stretch")

# =============================================================================
# GLOBAL SYSTEM FOOTER
# =============================================================================
st.markdown("""
<br><hr style='border:none;border-top:1px solid #CBD5E1;'>
<p style='text-align:center;color:#64748B;font-size:0.8rem;'>
    SkinLite AI Framework Core Engine • Academic Proof-of-Concept Project Setup
</p>
""", unsafe_allow_html=True)