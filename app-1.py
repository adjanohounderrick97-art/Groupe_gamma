import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import openai
from openai import OpenAI

import os
from io import BytesIO







# --- CONFIGURATION & DESIGN ---
st.set_page_config(page_title="NephroExpert AI - Expertise XAI", page_icon="🏥", layout="wide")

st.sidebar.header("🔐 Configuration OpenAI")

user_api_key = st.sidebar.text_input(
    "Entrez votre clé OpenAI",
    type="password"
)

if user_api_key:
    client = OpenAI(api_key=user_api_key)
    llm_available = True
    st.sidebar.success("Clé API chargée ✅")
else:
    llm_available = False
    st.sidebar.warning("Mode IA désactivé")


# CSS Haute Visibilité Médicale (Correction Contrastes et Sidebar)
st.markdown("""
    <style>
    .stApp, [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child { 
        background-color: #FFFFFF !important; 
    }
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stCheckbox div[data-baseweb="checkbox"] {
        border: 2px solid #1E3A8A !important;
    }
    h1, h2, h3, h4, h5, h6, p, label, .stWidgetLabel p {
        color: #1E293B !important;
        font-weight: 600 !important;
    }
    .main-header { 
        color: #1E3A8A !important; 
        font-size: 32px; font-weight: bold; border-bottom: 3px solid #1E3A8A; 
        text-align: center; padding-bottom: 15px; margin-bottom: 25px; 
    }
    input[type="number"], .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
    }
    .doctor-section, .patient-section { 
        background-color: #FFFFFF !important; 
        border-radius: 12px; padding: 25px; border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 25px; 
    }
    .doctor-section { border-left: 10px solid #1E3A8A !important; }
    .patient-section { border-left: 10px solid #10B981 !important; }
    .metric-card { 
        background: #F8FAFC !important; padding: 20px; border-radius: 10px; 
        border: 1px solid #E2E8F0 !important; text-align: center; 
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important; color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE MÉDICALE ---
def compute_egfr(creatinine_mg_dl, age, sex):
    if pd.isna(creatinine_mg_dl) or age <= 0: return 0
    K, alpha = (0.7, -0.241) if sex == "F" else (0.9, -0.302)
    sex_f = 1.012 if sex == "F" else 1.0
    scr_k = creatinine_mg_dl / K
    return 142 * min(scr_k, 1)**alpha * max(scr_k, 1)**-1.200 * (0.9938**age) * sex_f

@st.cache_resource
def load_assets():
    clf = joblib.load("xgboost_ckd_pipeline.joblib")
    features = joblib.load('features_list.joblib')
    return clf, features

# --- FONCTION SHAP AUGMENTÉE (RETOURNE LE GRAPHIQUE ET LES DONNÉES) ---
def get_shap_analysis(clf, input_df):
    try:
        model = clf.named_steps['model']
        preprocessor = clf.named_steps['preprocessor']
        X_transformed = preprocessor.transform(input_df)
        feature_names = preprocessor.get_feature_names_out()
        clean_features = [f.split('__')[-1] for f in feature_names]
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_transformed)

        # Convertir en array 1D pour le patient unique
        vals = shap_values.values.flatten()  # chaque feature

        prediction = model.predict(X_transformed)[0]
        class_idx = list(model.classes_).index(prediction)

        if isinstance(shap_values, list): vals = shap_values[class_idx][0]
        elif len(shap_values.shape) == 3: vals = shap_values[0, :, class_idx]
        else: vals = shap_values[0]

        shap_df = pd.DataFrame({'Feature': clean_features, 'Influence': vals})
        shap_df_sorted = shap_df.sort_values(by='Influence', ascending=False)
        
        # Création du graphique Plotly
        top_plot = shap_df_sorted.head(10).iloc[::-1] # Top 10 pour le graphique
        fig = px.bar(top_plot, x='Influence', y='Feature', orientation='h',
                     title=f"Poids des variables sur la prédiction (Stade {prediction})",
                     color='Influence', color_continuous_scale='RdBu_r')
        fig.update_layout(height=400, template='plotly_white')
        
        return fig, shap_df_sorted
    except Exception as e:
        return None, None
    
def generate_ai_interpretation(report_text, egfr, stade, justif):
    ai_analysis = "Analyse IA non disponible pour le moment."  # valeur par défaut

    if llm_available:
        try:
            prompt_llm = f"""
            Patient avec eGFR = {egfr}
            Stade IRC = {stade}
            Justification = {justif}
            """
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tu es un néphrologue expert."},
                    {"role": "user", "content": prompt_llm}
                ],
                temperature=0.3
            )
            ai_analysis = response.choices[0].message.content
        except Exception as e:   # <- attrape toutes les erreurs OpenAI
            print(f'⚠️ Analyse IA indisponible : {str(e)}')
            ai_analysis = "⚠️ Analyse IA indisponible"

    return ai_analysis



def generate_ai_patient_text(pat_name, egfr, stade, justif):
    """
    Génère un texte clair pour le patient via OpenAI.
    """
    if not llm_available:
        return "Analyse IA non disponible pour le moment."

    prompt = f"""
    Tu es un néphrologue expert. Explique à un patient nommé {pat_name} :
    - Son eGFR = {egfr:.1f}
    - Son stade d'insuffisance rénale = {stade}
    - La justification médicale : {justif}

    Rédige un texte clair, pédagogique et rassurant.
    Inclue :
    - Explication simple de la créatinine, urée, tension, âge.
    - Pourquoi il est à ce stade.
    - Conseils de vie personnalisés selon son stade.
    Limite le texte à tenir sur une page PDF.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un néphrologue expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        ai_text = response.choices[0].message.content
    except Exception as e:
        ai_text = f"Impossible de générer le texte IA : {e}"

    return ai_text


def generate_patient_report(pat_name, age, sexe, egfr, stade, creat, uree, sys, dia, ai_text=None):
    """
    Génère un PDF patient à partir d'un texte IA ou fallback.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []

    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]

    # Texte fallback si IA indisponible
    if not ai_text:
        ai_text = (f"Bonjour {pat_name},\n"
                   f"Votre bilan indique un eGFR de {egfr:.1f} mL/min "
                   f"et un stade IRC {stade}.\n"
                   f"Créatinine : {creat} mg/dL | Urée : {uree} g/L | "
                   f"Tension : {sys}/{dia} mmHg\n\n"
                   "Veuillez suivre les recommandations médicales et consulter votre néphrologue.")

    # On split le texte en paragraphes
    for line in ai_text.split("\n"):
        elements.append(Paragraph(line.replace(" ", "&nbsp;"), normal_style))
        elements.append(Spacer(1, 0.2*inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer


def plot_biomarkers_radar(data):
    categories = ['Créatinine', 'Urée', 'Tension Sys.', 'Tension Dias.', 'Âge']
    values = [data['creat']/1.1, data['uree']/0.45, data['sys']/130, data['dia']/85, data['age']/65]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Patient', fillcolor='rgba(30, 58, 138, 0.5)', line=dict(color='#1E3A8A')))
    fig.add_trace(go.Scatterpolar(r=[1]*5, theta=categories, line=dict(color='red', dash='dash'), name='Seuil de Norme'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 2])), showlegend=True, title="Profil Physiologique vs Normes")
    return fig

def main():
    st.markdown("<div class='main-header'>🏥 NEPHROEXPERT AI : ANALYSE CLINIQUE AVANCÉE</div>", unsafe_allow_html=True)
    
    try:
        clf, features_list = load_assets()
    except:
        st.error("⚠️ Fichiers du modèle (.joblib) introuvables.")
        return

    with st.sidebar:
        st.header("📋 Dossier Patient")
        doc_name = st.text_input("Médecin Référent", "Dr. S. Koné")
        pat_name = st.text_input("Identité Patient", "M. Jean Robert")
        age = st.number_input("Âge", 1, 110, 65)
        sexe = st.selectbox("Sexe biologique", ["M", "F"])
        st.divider()
        tabac = st.checkbox("Tabagisme")
        alcool = st.checkbox("Alcool")
        phytotherapie = st.checkbox("Phytothérapie")
        ains = st.checkbox("Usage régulier d'AINS")

    tab_med, tab_pat = st.tabs(["🩺 ESPACE MÉDECIN (Expertise)", "👤 ESPACE PATIENT (Accompagnement)"])

    with tab_med:
        col_in, col_res = st.columns([1, 1.8])
        with col_in:
            st.subheader("📥 Données Biologiques")
            creat = st.number_input("Créatinine (mg/dL)", 0.1, 20.0, 1.8)
            uree = st.number_input("Urée (g/L)", 0.1, 5.0, 0.6)
            sys = st.number_input("TA Systolique (mmHg)", 80, 220, 150)
            dia = st.number_input("TA Diastolique (mmHg)", 40, 130, 95)
            alb = st.selectbox("Albuminurie (BU)", ["Non", "Oui"])
            diurese = st.selectbox("Diurèse", ["Normale", "Oligurie", "Anurie"])
            hta = st.checkbox("Hypertension déclarée")
            diab = st.checkbox("Diabète déclaré")
            
        with col_res:
            if st.button("LANCER L'ANALYSE EXPERTE"):
                egfr = compute_egfr(creat, age, sexe)
                input_data = {
                    'Age': [age], 'Sexe': [1 if sexe == "F" else 0], 'creatinine_mg_dl': [creat],
                    'Uree_mgdl': [uree], 'Tension_systolique_mmhg': [sys], 'Tension_diastolique_mmhg': [dia],
                    'Hypertension': [1 if hta else 0], 'Diabete_Type1': [1 if diab else 0],
                    'Diabete_Type2': [0], 'Tabac': [1 if tabac else 0], 'Alcool': [1 if alcool else 0], 
                    'Consommation_AINS': [1 if ains else 0], 'Phytotherapie': [1 if phytotherapie else 0], 
                    'Hematurie_n': [0], 'Albuminurie_n': [1 if alb == "Oui" else 0],
                    'diurese_encoded': [0 if diurese == "Normale" else 1]
                }
                input_df = pd.DataFrame(input_data)[features_list]
                stade = clf.predict(input_df)[0]
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                with m1: st.markdown(f"<div class='metric-card'><b>Stade Prédit (IA)</b><br><h2>Stade {stade}</h2></div>", unsafe_allow_html=True)
                with m2: st.markdown(f"<div class='metric-card'><b>eGFR (Filtration)</b><br><h2>{egfr:.1f}</h2><small>mL/min</small></div>", unsafe_allow_html=True)
                with m3: 
                    color = "#EF4444" if stade >= 4 else ("#F59E0B" if stade >= 3 else "#10B981")
                    st.markdown(f"<div class='metric-card'><b>État Patient</b><br><h2 style='color:{color}'>{'Critique' if stade >= 4 else ('Surveillance' if stade >= 3 else 'Stable')}</h2></div>", unsafe_allow_html=True)

                # SHAP
                st.markdown("<div class='doctor-section'>", unsafe_allow_html=True)
                st.subheader("🔍 Analyse d'Interprétabilité (SHAP)")
                shap_fig, shap_data = get_shap_analysis(clf, input_df)

                if shap_data is not None:
                    shap_report = "\n".join([f"- {row['Feature']} : {row['Influence']:.4f} (Poids)"
                                            for index, row in shap_data.head(8).iterrows()])
                else:
                    shap_report = "Analyse SHAP non disponible."
                if shap_fig:
                    st.plotly_chart(shap_fig, use_container_width=True)
                else:
                    st.warning("🔍 Analyse SHAP non disponible pour ce patient.")


                st.caption("💡 Le graphique ci-dessus justifie mathématiquement le stade prédit.")
                st.markdown("</div>", unsafe_allow_html=True)

                # Justification
                txt_hta = "L'hypertension sévère " if sys > 140 else "La tension artérielle "
                justif = f"Le patient présente une insuffisance de Stade {stade} avec un eGFR à {egfr:.1f}. {txt_hta} couplée à une créatinine de {creat}mg/dL justifie cette classification."
                st.write(justif)
                
                # --- GÉNÉRATION DU RAPPORT TECHNIQUE DÉTAILLÉ ---
                st.divider()
                
                # Construction du texte SHAP pour le rapport
                if shap_data is not None:
                    shap_report = "\n".join([f"- {row['Feature']} : {row['Influence']:.4f} (Poids)"
                                            for index, row in shap_data.head(8).iterrows()])
                else:
                    shap_report = "Analyse SHAP non disponible."
                if shap_fig is not None:
                    st.plotly_chart(shap_fig, use_container_width=True)
                else:
                    st.warning("🔍 Analyse SHAP non disponible pour ce patient.")


                                
                full_report = f"""
                
RAPPORT MÉDICAL TECHNIQUE - NEPHROEXPERT AI
-------------------------------------------
DATE : {datetime.date.today()}
MÉDECIN : {doc_name}
PATIENT : {pat_name}
ÂGE : {age} | SEXE : {sexe}

BILAN BIOLOGIQUE :
- Créatinine : {creat} mg/dL
- Urée : {uree} g/L
- eGFR : {egfr:.2f} mL/min/1.73m²
- TA : {sys}/{dia} mmHg

RÉSULTATS PRÉDICTIFS :
- STADE IRC PRÉDIT : {stade}
- INTERPRÉTATION : {justif}

--- ANALYSE D'INTERPRÉTABILITÉ (SHAP) ---
Ce tableau représente l'importance de chaque variable sur la décision finale de l'IA.
Plus le poids est élevé, plus la variable a contribué à définir le stade {stade}.

{shap_report}


DISPOSITIONS RECOMMANDÉES :
- Réévaluation sous 48h.
- Bilan ionique complet (K+, Na+).
- Éviction stricte des AINS et produits néphrotoxiques.
-------------------------------------------
Note : Ce document est une aide à la décision générée par IA.
"""
                # 🔹 Appel sécurisé de la fonction IA
                #ai_analysis = generate_ai_interpretation(full_report)

                with st.spinner("Analyse IA en cours..."):
                    ai_analysis = generate_ai_interpretation(full_report,egfr,stade,justif)
                    st.subheader("🧠 Analyse IA Expert")
                    st.write(ai_analysis)

            
                # --- GÉNÉRATION DU PDF ---
                from io import BytesIO

                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer)
                elements = []

                styles = getSampleStyleSheet()
                normal_style = styles["Normal"]

                for line in full_report.split("\n"):
                    elements.append(Paragraph(line.replace(" ", "&nbsp;"), normal_style))
                    elements.append(Spacer(1, 0.2 * inch))

                doc.build(elements)
                buffer.seek(0)

                st.download_button(
                    "📥 Télécharger le Rapport Médical Technique (PDF)",
                    buffer,
                    file_name=f"Rapport_Expert_{pat_name}.pdf",
                    mime="application/pdf"
                )

                st.session_state['resultat'] = {"stade": stade, "egfr": egfr, "justif": justif}

    with tab_pat:
        if 'resultat' in st.session_state:
                res = st.session_state['resultat']
                st.markdown("<div class='patient-section'>", unsafe_allow_html=True)
                st.header(f"Bonjour {pat_name}")
                st.write(f"Votre bilan indique que vos reins fonctionnent à environ **{int(res['egfr'])}%** de leur capacité.")
                st.table(pd.DataFrame({"Semaine": ["S1", "S2", "S3", "S4"], "Action": ["Bilan biologique", "Mesure ta consommation quotidienne", "Consultation Diététique", "Contrôle Néphrologue"]}))
                st.markdown("</div>", unsafe_allow_html=True)
                with st.spinner("🧠 Génération du rapport IA patient..."):
                    ai_text_patient = None
                    if llm_available:
                        with st.spinner("Génération du rapport IA patient..."):
                            try:
                                ai_text_patient = generate_ai_patient_text(
                                    pat_name, res['egfr'], res['stade'], res['justif']
                                )
                            except Exception as e:
                                st.warning(f"⚠️ Impossible de générer le rapport patient : {str(e)}")   
                                ai_text_patient=None

                    st.subheader("🧠 Analyse IA Expert")
                    if ai_text_patient:
                        st.write(ai_text_patient)
                    else:
                         # Texte fallback standard si l'IA ne fonctionne pas
                        st.write(f"Bonjour {pat_name},\n"
                                f"Votre bilan indique un eGFR de {res['egfr']:.1f} mL/min "
                                f"et un stade IRC {res['stade']}.\n"
                                f"{res['justif']}\n\n"
                                "Veuillez suivre les recommandations médicales générales et consulter votre néphrologue régulièrement.")

                    # --- Génération PDF patient avec texte IA ou fallback ---
                    patient_pdf = generate_patient_report(
                        pat_name, age, sexe, res['egfr'], res['stade'],
                        creat, uree, sys, dia, ai_text=ai_text_patient
                    )

                    st.download_button(
                        "📥 Télécharger votre rapport patient (PDF)",
                        patient_pdf,
                        file_name=f"Rapport_Patient_{pat_name}.pdf",
                        mime="application/pdf"
                    )

                    
        else:
            st.info("Veuillez d'abord lancer l'analyse dans l'onglet Médecin.")

if __name__ == "__main__":
    main()