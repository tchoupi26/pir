import streamlit as st
import pandas as pd
import joblib

# Chargement du modèle entraîné avec mise en cache
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('best_avc_model.pkl')

model = load_model()

# Titre de l'application
st.title("Estimation du Risque d'AVC")
st.write("Remplissez le formulaire ci-dessous pour estimer votre risque d'AVC.")

# Formulaire de saisie des données utilisateur
def user_input_features():
    age = st.number_input('Âge', min_value=18, max_value=100, value=50)
    avg_glucose_level = st.number_input('Niveau moyen de glucose (mg/dL)', min_value=50.0, max_value=300.0, value=120.0)
    bmi = st.number_input('Indice de Masse Corporelle (BMI)', min_value=10.0, max_value=70.0, value=25.0)
    
    gender = st.selectbox('Genre', ('Male', 'Female', 'Other'))
    hypertension = st.selectbox('Hypertension', (0, 1))
    heart_disease = st.selectbox('Maladie cardiaque', (0, 1))
    ever_married = st.selectbox('Marié(e)', ('Yes', 'No'))
    work_type = st.selectbox('Type de travail', ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'))
    Residence_type = st.selectbox('Type de résidence', ('Urban', 'Rural'))
    smoking_status = st.selectbox('Statut tabagique', ('never smoked', 'formerly smoked', 'smokes', 'Unknown'))
    
    data = {
        'age': age,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'gender': gender,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'smoking_status': smoking_status
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Bouton de prédiction
if st.button('Estimer le Risque d\'AVC'):
    try:
        # Prédiction du risque
        risk = model.predict_proba(input_df)[:, 1][0] * 100
        
        # Détermination de la catégorie de risque avec emojis Unicode
        if risk < 20:
            risk_category = "Faible"
            risk_color = "green"
            risk_icon = "🟢"
        elif 20 <= risk < 50:
            risk_category = "Modéré"
            risk_color = "orange"
            risk_icon = "🟠"
        else:
            risk_category = "Élevé"
            risk_color = "🔴"
        
        # Affichage du risque avec stylisation
        st.markdown(
            f"<h2 style='color:{risk_color}; font-size: 36px;'>{risk_icon} Le risque prédit d'AVC est de {risk:.2f}% ({risk_category})</h2>",
            unsafe_allow_html=True
        )
        
        # Génération des conseils de santé
        conseils = []
        
        # Conseils basés sur le niveau de glucose
        if input_df['avg_glucose_level'].iloc[0] > 126:
            conseils.append("• **Contrôlez votre glycémie** : Un niveau élevé de glucose peut augmenter le risque d'AVC.")
        
        # Conseils basés sur l'hypertension
        if input_df['hypertension'].iloc[0] == 1:
            conseils.append("• **Contrôlez votre tension artérielle** : L'hypertension est un facteur de risque majeur d'AVC.")
        
        # Conseils basés sur le BMI
        if input_df['bmi'].iloc[0] >= 30:
            conseils.append("• **Gérez votre poids** : Un indice de masse corporelle élevé augmente le risque d'AVC.")
        elif 25 <= input_df['bmi'].iloc[0] < 30:
            conseils.append("• **Maintenez un poids santé** : Un indice de masse corporelle modéré peut réduire le risque d'AVC.")
        
        # Conseils basés sur le statut tabagique
        if input_df['smoking_status'].iloc[0] in ['smokes', 'formerly smoked']:
            conseils.append("• **Arrêtez de fumer** : Le tabagisme augmente significativement le risque d'AVC.")
        
        # Conseils basés sur la maladie cardiaque
        if input_df['heart_disease'].iloc[0] == 1:
            conseils.append("• **Surveillez votre santé cardiaque** : Les maladies cardiaques sont liées à un risque accru d'AVC.")
        
        # Affichage des conseils
        if conseils:
            st.markdown("### **Conseils de santé :**")
            for conseil in conseils:
                st.markdown(conseil)
        else:
            st.markdown("### **Félicitations !** Vos données ne montrent pas de facteurs de risque majeurs pour l'AVC.")
    
    except Exception as e:
        st.error("Une erreur est survenue lors de la prédiction. Veuillez vérifier vos entrées et réessayer.")
