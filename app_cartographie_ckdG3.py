import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Cartographie CKD - Bénin",
    page_icon="",
    layout="wide"
)

# Titre principal
st.title(" Cartographie des Zones à Risque - Maladie Rénale Chronique (CKD)")
st.markdown("### Analyse géographique des patients au Bénin")

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv('Data_AI4CKD_-_Original.csv')
    return df

df = load_data()

# Nettoyer la colonne département
df['Département'] = df['Adresse (Département)'].fillna('Non spécifié')
df['Stage_CKD'] = df["Stage de l'IRC"].fillna('Non défini')

# Filtrer les valeurs aberrantes
df = df[~df['Département'].isin(['18%'])]
df = df[~df['Stage_CKD'].isin(['0%'])]

# Coordonnées approximatives des départements du Bénin
departements_coords = {
    'Littoral': {'lat': 6.3654, 'lon': 2.4183, 'nom_complet': 'Littoral (Cotonou)'},
    'Atlantique': {'lat': 6.6833, 'lon': 2.3500, 'nom_complet': 'Atlantique'},
    'Ouémé': {'lat': 6.4969, 'lon': 2.6289, 'nom_complet': 'Ouémé (Porto-Novo)'},
    'Zou': {'lat': 7.1833, 'lon': 2.1500, 'nom_complet': 'Zou (Abomey)'},
    'Mono': {'lat': 6.4833, 'lon': 1.6833, 'nom_complet': 'Mono (Lokossa)'},
    'Plateau': {'lat': 7.0167, 'lon': 2.6167, 'nom_complet': 'Plateau (Pobè)'},
    'Couffo': {'lat': 7.0000, 'lon': 1.7500, 'nom_complet': 'Couffo (Aplahoué)'},
    'Collines': {'lat': 8.0000, 'lon': 2.3333, 'nom_complet': 'Collines (Savalou)'},
    'Alibori': {'lat': 11.1333, 'lon': 2.6167, 'nom_complet': 'Alibori (Kandi)'},
}

# Calcul du score de risque par département
def calculer_score_risque(stage):
    """Convertir le stage CKD en score numérique"""
    scores = {
        'CKD 1': 1,
        'CKD 2': 2,
        'CKD 3a': 3,
        'CKD 3b': 4,
        'CKD 4': 5,
        'CKD 5': 6
    }
    return scores.get(stage, 0)

df['Score_Risque'] = df['Stage_CKD'].apply(calculer_score_risque)

# Agrégation par département
dept_stats = df.groupby('Département').agg({
    'ID': 'count',
    'Score_Risque': 'mean',
    'Stage_CKD': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Non défini'
}).reset_index()

dept_stats.columns = ['Département', 'Nombre_Patients', 'Score_Risque_Moyen', 'Stage_Dominant']

# Ajouter les coordonnées
dept_stats['Latitude'] = dept_stats['Département'].map(lambda x: departements_coords.get(x, {}).get('lat', 0))
dept_stats['Longitude'] = dept_stats['Département'].map(lambda x: departements_coords.get(x, {}).get('lon', 0))
dept_stats['Nom_Complet'] = dept_stats['Département'].map(lambda x: departements_coords.get(x, {}).get('nom_complet', x))

# Filtrer les départements sans coordonnées
dept_stats = dept_stats[dept_stats['Latitude'] != 0]

# Déterminer le niveau de risque
def niveau_risque(score):
    if score < 2.5:
        return 'Faible'
    elif score < 4.5:
        return 'Moyen'
    else:
        return 'Élevé'

dept_stats['Niveau_Risque'] = dept_stats['Score_Risque_Moyen'].apply(niveau_risque)

# Couleurs pour les niveaux de risque
couleurs_risque = {
    'Faible': '#2ecc71',  # Vert
    'Moyen': '#f39c12',   # Orange
    'Élevé': '#e74c3c'    # Rouge
}

# Sidebar - Filtres
st.sidebar.header("🔍 Filtres")
niveau_filtre = st.sidebar.multiselect(
    "Niveau de risque",
    options=['Faible', 'Moyen', 'Élevé'],
    default=['Faible', 'Moyen', 'Élevé']
)

dept_filtre = st.sidebar.multiselect(
    "Départements",
    options=dept_stats['Département'].unique(),
    default=dept_stats['Département'].unique()
)

# Appliquer les filtres
dept_stats_filtered = dept_stats[
    (dept_stats['Niveau_Risque'].isin(niveau_filtre)) &
    (dept_stats['Département'].isin(dept_filtre))
]

# Métriques principales
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(" Départements", len(dept_stats_filtered))

with col2:
    st.metric(" Total Patients", int(dept_stats_filtered['Nombre_Patients'].sum()))

with col3:
    risque_moyen_global = dept_stats_filtered['Score_Risque_Moyen'].mean()
    st.metric(" Score Risque Moyen", f"{risque_moyen_global:.2f}/6")

with col4:
    zones_elevees = len(dept_stats_filtered[dept_stats_filtered['Niveau_Risque'] == 'Élevé'])
    st.metric(" Zones à Risque Élevé", zones_elevees)

st.markdown("---")

# Carte interactive principale
st.subheader(" Carte Interactive des Zones à Risque")

# Créer la carte avec Plotly
fig_map = go.Figure()

# Ajouter les marqueurs pour chaque département
for niveau in ['Faible', 'Moyen', 'Élevé']:
    dept_niveau = dept_stats_filtered[dept_stats_filtered['Niveau_Risque'] == niveau]
    
    if not dept_niveau.empty:
        fig_map.add_trace(go.Scattermapbox(
            lat=dept_niveau['Latitude'],
            lon=dept_niveau['Longitude'],
            mode='markers',
            marker=dict(
                size=dept_niveau['Nombre_Patients'] * 0.5,  # Taille proportionnelle
                color=couleurs_risque[niveau],
                opacity=0.8,
                sizemode='diameter'
            ),
            text=dept_niveau.apply(
                lambda row: f"<b>{row['Nom_Complet']}</b><br>" +
                           f"Patients: {int(row['Nombre_Patients'])}<br>" +
                           f"Score: {row['Score_Risque_Moyen']:.2f}/6<br>" +
                           f"Niveau: {row['Niveau_Risque']}<br>" +
                           f"Stage dominant: {row['Stage_Dominant']}",
                axis=1
            ),
            hovertemplate='%{text}<extra></extra>',
            name=f"Risque {niveau}",
            showlegend=True
        ))

# Configuration de la carte
fig_map.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=9.3, lon=2.3),
        zoom=5.5
    ),
    height=600,
    margin={"r":0,"t":0,"l":0,"b":0},
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.8)"
    )
)

st.plotly_chart(fig_map, use_container_width=True)

# Graphiques supplémentaires
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader(" Distribution des Patients par Département")
    
    # Trier par nombre de patients
    dept_sorted = dept_stats_filtered.sort_values('Nombre_Patients', ascending=True)
    
    fig_bar = px.bar(
        dept_sorted,
        x='Nombre_Patients',
        y='Département',
        orientation='h',
        color='Niveau_Risque',
        color_discrete_map=couleurs_risque,
        text='Nombre_Patients',
        title="Nombre de patients par département"
    )
    
    fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
    fig_bar.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("Score de Risque Moyen par Département")
    
    fig_scatter = px.scatter(
        dept_stats_filtered,
        x='Nombre_Patients',
        y='Score_Risque_Moyen',
        size='Nombre_Patients',
        color='Niveau_Risque',
        color_discrete_map=couleurs_risque,
        text='Département',
        title="Corrélation Patients vs Risque"
    )
    
    fig_scatter.update_traces(textposition='top center')
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

# Distribution des stages CKD
st.markdown("---")
st.subheader(" Répartition des Stages CKD")

col1, col2 = st.columns(2)

with col1:
    # Distribution globale
    df_filtered = df[df['Département'].isin(dept_filtre)]
    stage_counts = df_filtered['Stage_CKD'].value_counts()
    
    fig_pie = px.pie(
        values=stage_counts.values,
        names=stage_counts.index,
        title="Distribution globale des stages CKD",
        color_discrete_sequence=px.colors.sequential.Reds_r
    )
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Distribution par département
    stage_dept = df_filtered.groupby(['Département', 'Stage_CKD']).size().reset_index(name='Count')
    
    fig_stacked = px.bar(
        stage_dept,
        x='Département',
        y='Count',
        color='Stage_CKD',
        title="Répartition des stages par département",
        color_discrete_sequence=px.colors.sequential.Reds_r
    )
    fig_stacked.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_stacked, use_container_width=True)

# Tableau de données détaillé
st.markdown("---")
st.subheader(" Données Détaillées par Département")

# Préparer le tableau avec mise en forme
dept_display = dept_stats_filtered[['Nom_Complet', 'Nombre_Patients', 'Score_Risque_Moyen', 
                                      'Niveau_Risque', 'Stage_Dominant']].copy()
dept_display.columns = ['Département', 'Patients', 'Score Moyen', 'Niveau Risque', 'Stage Dominant']
dept_display['Score Moyen'] = dept_display['Score Moyen'].round(2)
dept_display = dept_display.sort_values('Patients', ascending=False)

# Afficher le tableau avec style
st.dataframe(
    dept_display.style.background_gradient(subset=['Patients'], cmap='YlOrRd'),
    use_container_width=True,
    height=400
)

# Recommandations
st.markdown("---")
st.subheader("Recommandations d'Intervention")

# Identifier les zones prioritaires
zones_elevees_list = dept_stats_filtered[dept_stats_filtered['Niveau_Risque'] == 'Élevé'].sort_values(
    'Nombre_Patients', ascending=False
)

if not zones_elevees_list.empty:
    st.warning(" **Zones Prioritaires à Risque Élevé**")
    for idx, row in zones_elevees_list.iterrows():
        st.markdown(f"""
        - **{row['Nom_Complet']}**: {int(row['Nombre_Patients'])} patients - Score: {row['Score_Risque_Moyen']:.2f}/6
        """)
    
    st.markdown("""
    **Actions recommandées:**
    - Déploiement prioritaire de campagnes de dépistage
    - Renforcement des capacités médicales locales
    - Programmes de prévention et sensibilisation ciblés
    - Suivi rapproché des patients à haut risque
    """)
else:
    st.success("Aucune zone à risque élevé identifiée avec les filtres actuels")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><b>Hackathon IA - Bootcamp Cohorte 1</b></p>
    <p>Intelligence Artificielle au service de la Maladie Rénale Chronique (CKD)</p>
    <p>Données: CNHU/HKM - Bénin</p>
</div>
""", unsafe_allow_html=True)
