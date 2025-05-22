import streamlit as st
import pandas as pd
from scraping.scraper import scrap_initial, get_user_rate, get_item_description
import ML.data.data
from ML.ia import VintedDealPredictor
from ML.descriptionAnalyze.descriptionAnalyze import DescriptionStatusToScoreModel, predict_score_for_single_item
import time
import torch
import numpy as np
import base64
import os

# Configuration de la page
st.set_page_config(
    page_title="Smartbargain - Trouvez les meilleures affaires",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour encoder l'image en base64
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Chemin vers votre logo (adaptez selon votre structure de dossiers)
logo_path = os.path.join(os.path.dirname(__file__), "Document/logo.png")
# Ou un chemin direct comme:
# logo_path = "./logo.png"

# Encoder l'image
try:
    logo_base64 = get_base64_encoded_image(logo_path)

    # Markdown avec l'image encodée en base64
    st.markdown(f"""
    <div class="app-header" style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px;">
        <img src="data:image/png;base64,{logo_base64}" width="150" style="margin-bottom: 10px;">
        <h1 class="app-title">Smartbargain</h1>
        <p>Trouvez les meilleures affaires sur Vinted grâce à l'IA</p>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Erreur lors du chargement de l'image: {e}")
    # Solution de repli si l'image ne peut pas être chargée
    st.markdown("""
    <div class="app-header" style="text-align: center; padding: 20px;">
        <h1 class="app-title">Smartbargain</h1>
        <p>Trouvez les meilleures affaires sur Vinted grâce à l'IA</p>
    </div>
    """, unsafe_allow_html=True)

def display_items(df, title, is_good_deal=False):
    """Affiche les items filtrés dans une grille de 3 colonnes avec un titre."""
    st.markdown(f'<div class="items-section"><h2>{title}</h2>', unsafe_allow_html=True)

    num_cols = 3
    rows = [df[i:i+num_cols] for i in range(0, len(df), num_cols)]

    for row in rows:
        cols = st.columns(num_cols)
        for idx, (_, item) in enumerate(row.iterrows()):
            if idx < len(cols):
                with cols[idx]:
                    st.markdown('<div class="item-card">', unsafe_allow_html=True)

                    # Badge pour bonnes affaires
                    if is_good_deal:
                        st.markdown('<span class="badge-good-deal">Bonne affaire</span>', unsafe_allow_html=True)

                    # Image et détails de l'article
                    st.markdown(f'<a href="{item["URL"]}" target="_blank">'
                                f'<img src="{item["photo_url"]}" class="item-image"></a>', unsafe_allow_html=True)
                    st.markdown(f'<p class="item-title">{item["Title"]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="item-price" style="font-size: 22px; font-weight: bold;">'
                                f'{item["Price"]} {item["currency"]}</p>', unsafe_allow_html=True)

                    # Afficher la marque si disponible
                    if "Brand" in item and item["Brand"]:
                        st.markdown(f'<p style="color: #6c757d; font-size: 14px;">{item["Brand"]}</p>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Création de la structure de l'interface
header_section = st.container()  # Pour le formulaire
results_section = st.container()  # Pour les résultats

# Formulaire de recherche avec style amélioré
with header_section:
    st.markdown('<div class="search-form">', unsafe_allow_html=True)
    st.subheader("Rechercher des articles")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        titre_filter = st.text_input("Titre", placeholder="Ex: Veste en cuir...")
    with col2:
        marque_options = ["Carhartt", "The North Face", "Nike", "Adidas", "Zara", "H&M"]  # Ajoutez ici les marques populaires
        marque_filter = st.selectbox("Marque", options=marque_options)
    with col3:
        categorie_options = ["bonnets", "sweats", "t-shirts", "robes", "pantalons", "chaussures"]  # Ajoutez ici les catégories populaires
        categorie_filter = st.selectbox("Catégorie", options=categorie_options)
    with col4:
        genre_options = ["hommes", "femmes", "enfants"]  # Ajoutez ici les genres disponibles
        genre_filter = st.selectbox("Genre", options=genre_options)

    st.markdown('<div style="padding: 10px 0;"></div>', unsafe_allow_html=True)
    search_button = st.button("🔍 Rechercher", help="Lancer la recherche d'articles")
    st.markdown('</div>', unsafe_allow_html=True)

# Dans la section des résultats
# Dans la section des résultats
with results_section:
    # Créer des emplacements pour les bonnes affaires et tous les articles
    good_deals_placeholder = st.empty()
    all_items_placeholder = st.container()

    # Si le bouton est cliqué, effectuer la recherche
    if search_button:
        research_text = titre_filter
        marque_recherche = marque_filter if marque_filter else None
        categorie_recherche = categorie_filter if categorie_filter else None
        genre_recherche = genre_filter if genre_filter else None

        # Animation de recherche
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            st.markdown('<div class="loading-animation">', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            st.markdown('<p style="text-align: center;">Recherche des articles...</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Simulation de progression (à remplacer par le vrai scraping)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            progress_placeholder.empty()

        # Option pour basculer entre le scraping et le chargement d'un CSV
        use_csv = False  # Changez cette valeur à True pour charger les données depuis un CSV

        if use_csv:
            try:
                # Charger les données depuis un fichier CSV
                csv_path = "sweatTNF.csv"  # Chemin vers le fichier CSV
                initial_items_df = pd.read_csv(csv_path)
                #st.success("Données chargées depuis le fichier CSV.")

                # Si les données sont chargées depuis le CSV, pas besoin de récupérer les détails
                detailed_df = initial_items_df.copy()
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier CSV: {e}")
                initial_items_df = pd.DataFrame()  # DataFrame vide en cas d'erreur
        else:
            try:
                # Récupérer les articles via le scraping
                with st.spinner("Récupération des articles..."):
                    initial_items_df = scrap_initial(research_text, genre_recherche, categorie_recherche, marque_recherche, 10)
                    print(f"Taille du dataset initial: {initial_items_df.shape[0]} articles")
                    initial_items_df = initial_items_df.drop_duplicates(subset=['URL'])
                    print(f"Taille du dataset sans les doublons: {initial_items_df.shape[0]} articles")


            except Exception as e:
                st.error(f"Erreur lors de la récupération des articles: {e}")
                initial_items_df = pd.DataFrame()  # DataFrame vide en cas d'erreur

        if not initial_items_df.empty:
            # Afficher tous les articles d'abord
            with all_items_placeholder:
                display_items(initial_items_df, "📦 Tous les articles trouvés")

            # Traitement des prédictions
            try:
                if not use_csv:
                    # Afficher l'indicateur de chargement pendant la récupération des détails
                    with good_deals_placeholder.container():
                        st.markdown('<div class="items-section">', unsafe_allow_html=True)
                        st.markdown('<div style="display: flex; align-items: center; gap: 10px;">', unsafe_allow_html=True)
                        st.markdown('<h2>⏳ Récupération des détails en cours...</h2>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        progress_text_details = st.empty()
                        progress_bar_details = st.progress(0)

                        detailed_data = []
                        for index, row in initial_items_df.iterrows():
                            progress = (index + 1) / len(initial_items_df)
                            progress_bar_details.progress(min(progress, 1.0))
                            progress_text_details.text(f"Récupération des détails ({index+1}/{len(initial_items_df)}): {row['Title']}")

                            try:
                                user_evaluations, user_rating = get_user_rate(row['user_url'])
                                item_data = {
                                    "URL": row['URL'],
                                    "NbEvalUser": user_evaluations,
                                    "UserRating": user_rating
                                }

                                description = get_item_description(row['URL'])
                                item_data["Description"] = description

                                detailed_data.append(item_data)
                            except Exception as e:
                                #st.warning(f"Erreur pour {row['Title']}: {e}", icon="⚠️")
                                detailed_data.append({
                                    "URL": row['URL'],
                                    "NbEvalUser": None,
                                    "UserRating": None,
                                    "Description": ""
                                })

                        # Une fois la récupération terminée, vider la barre de progression
                        progress_text_details.empty()
                        progress_bar_details.empty()

                        st.markdown('</div>', unsafe_allow_html=True) # Fermer la div items-section

                    # Créer un nouveau conteneur pour l'analyse des bonnes affaires avec la barre de progression
                    with good_deals_placeholder.container():

                        progress_bar_prediction = st.progress(0)

                        detailed_df = pd.DataFrame(detailed_data)
                        df_for_prediction = pd.merge(initial_items_df, detailed_df, on='URL', how='left')

                        # Préparation et nettoyage des données
                        df_cleaned = data.nettoyer_outliers(df_for_prediction.copy())
                        df_encoded = data.encode_df(df_cleaned.copy())

                        # Chargement des modèles
                        predictor = VintedDealPredictor(df_encoded)
                        predictor.load_model('deal_predictor_model.keras')

                        trusmodel = DescriptionStatusToScoreModel()
                        trusmodel.load_state_dict(torch.load("descriptionAnalyze/model_description_status.pt"))

                        predictor.load_encoders({
                            'Brand': 'Brand_label_encoder.pkl',
                            'Type': 'Type_label_encoder.pkl'
                        })

                        # Prédictions
                        predictions_list = []
                        trust_scores_list = []

                        for index, row in df_encoded.iterrows():
                            progress = (index + 1) / len(df_encoded)
                            progress_bar_prediction.progress(min(progress, 1.0))

                            prediction = predictor.predict_deal(row)
                            description = row['Description']
                            status = row['ItemStatus']
                            trust_score = predict_score_for_single_item(trusmodel, description, status)
                            trust_scores_list.append(trust_score)
                            predictions_list.append(prediction)

                        # Une fois la prédiction terminée, vider la barre de progression
                        progress_bar_prediction.empty()


                        df_cleaned['prediction'] = predictions_list
                        df_cleaned['trust_score'] = trust_scores_list

                        # Fusion pondérée
                        prediction_weight = 0.8
                        trust_score_weight = 0.2
                        df_cleaned['weighted_score'] = (
                            prediction_weight * df_cleaned['prediction'] +
                            trust_score_weight * df_cleaned['trust_score']
                        )

                        # Seuil pour déterminer une bonne affaire
                        threshold = 0.8
                        df_cleaned['bonne_affaire'] = df_cleaned['weighted_score'] >= threshold

                        good_deals_df = df_cleaned[df_cleaned['bonne_affaire']]

                        st.markdown('</div>', unsafe_allow_html=True) # Fermer la div items-section

                        # # Une fois les bonnes affaires affichées, vider le conteneur
                        # good_deals_placeholder.empty()

                        # Animation de complétion
                        st.balloons()

                        # Mettre à jour l'espace réservé avec les bonnes affaires
                        if not good_deals_df.empty:
                            display_items(good_deals_df, "✅ Bonnes affaires prédites par l'IA", is_good_deal=True)
                        else:
                            st.info("Aucune bonne affaire n'a été prédite par l'IA.")

                else:
                    # Si nous utilisons CSV, afficher quand même l'indicateur
                    with good_deals_placeholder.container():
                        st.markdown('<div class="items-section">', unsafe_allow_html=True)
                        progress_bar_prediction = st.progress(0)
                        df_for_prediction = initial_items_df.copy()
                        # Préparation et nettoyage des données
                        df_cleaned = data.nettoyer_outliers(df_for_prediction.copy())
                        df_encoded = data.encode_df(df_cleaned.copy())

                        # Chargement des modèles
                        predictor = VintedDealPredictor(df_encoded)
                        predictor.load_model('deal_predictor_model.keras')

                        trusmodel = DescriptionStatusToScoreModel()
                        trusmodel.load_state_dict(torch.load("descriptionAnalyze/model_description_status.pt"))

                        predictor.load_encoders({
                            'Brand': 'Brand_label_encoder.pkl',
                            'Type': 'Type_label_encoder.pkl'
                        })

                        # Prédictions
                        predictions_list = []
                        trust_scores_list = []

                        for index, row in df_encoded.iterrows():
                            progress = (index + 1) / len(df_encoded)
                            progress_bar_prediction.progress(min(progress, 1.0))

                            prediction = predictor.predict_deal(row)
                            description = row['Description']
                            status = row['ItemStatus']
                            trust_score = predict_score_for_single_item(trusmodel, description, status)
                            trust_scores_list.append(trust_score)
                            predictions_list.append(prediction)

                        # Une fois la prédiction terminée, vider la barre de progression
                        progress_bar_prediction.empty()

                        df_cleaned['prediction'] = predictions_list
                        df_cleaned['trust_score'] = trust_scores_list

                        # Fusion pondérée
                        prediction_weight = 0.8
                        trust_score_weight = 0.2
                        df_cleaned['weighted_score'] = (
                            prediction_weight * df_cleaned['prediction'] +
                            trust_score_weight * df_cleaned['trust_score']
                        )

                        # Seuil pour déterminer une bonne affaire
                        threshold = 0.8
                        df_cleaned['bonne_affaire'] = df_cleaned['weighted_score'] >= threshold

                        good_deals_df = df_cleaned[df_cleaned['bonne_affaire']]

                        # Supprimer le texte d'analyse une fois terminé
                        st.markdown('</div>', unsafe_allow_html=True)  # Fermer la div items-section

                        # Animation de complétion
                        st.balloons()

                        # Mettre à jour l'espace réservé avec les bonnes affaires
                        if not good_deals_df.empty:
                            display_items(good_deals_df, "✅ Bonnes affaires prédites par l'IA", is_good_deal=True)
                        else:
                            st.info("Aucune bonne affaire n'a été prédite par l'IA.")
                    
            except Exception as e:
                st.error(f"Erreur lors de l'analyse: {e}")
                # En cas d'erreur, effacer l'indicateur de chargement
                with good_deals_placeholder.container():
                    st.error("Une erreur est survenue lors de l'analyse des bonnes affaires.")
        else:
            st.info("Aucun article trouvé avec ces critères.")

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown('© 2025 Smartbargain - Trouvez les meilleures affaires sur Vinted', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)