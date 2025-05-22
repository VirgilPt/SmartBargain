from vinted_scraper import VintedScraper
import json
import requests
from bs4 import BeautifulSoup
import re, time

import pandas as pd

brandjson = 'scraping/brand.json'
catalog ='scraping/catalog_correspondences.json'


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-CA,en-US;q=0.7,en;q=0.3",
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Priority': 'u=0, i',
}

def get_brand_id_by_title(json_file_path, title) -> str:
    title = title.lower()
    if not title:
        return None
    with open(json_file_path, 'r') as file:
        brands = json.load(file)
        for brand in brands:
            if brand.get("title").lower() == title:
                return brand.get("id")
    return None

def get_catalog_id_by_title(json_file_path, gender, title) -> str:
    title = title.lower()
    gender = gender.lower()
    with open(json_file_path, 'r') as file:
        catalogs = json.load(file)
        for catalog_title, catalog_id in catalogs[gender].items():
            if catalog_title.lower() == title:
                return catalog_id
    return None

def get_data_frame_initial(items, category):
    data = []
    for item in items:
        data.append({
            "Title": item.title,
            "Brand": item.brand.title,
            'user_url': item.user.profile_url,
            "Type": category,
            "Price": item.price,
            "ItemStatus": item.status,
            "NbFavourite": item.favourite_count,
            "Iduser": item.user.id,
            "URL": item.url,
            "photo_url": item.photos[0].url if item.photos else None,
            "currency": item.currency
        })
    return pd.DataFrame(data)

def scrap_initial(item_title ='', gender ='', category ='', brand ='', max_page = 1):
    scraper = VintedScraper("https://www.vinted.fr")  # init the scraper with the baseurl
    nb_pages = max_page
    all_items = []
    page = 0
    while page < nb_pages:
        params = {
            "search_text": item_title,
            "brand_ids": get_brand_id_by_title(brandjson, brand),
            "catalog_ids" : get_catalog_id_by_title(catalog, gender, category),
            "page": page
            # Add other query parameters like the pagination and so on
        }
        items = scraper.search(params)  # get all the items
        status_mapping = {"New with tags": 1, "New without tags": 2, "Very good": 3, "Good": 4}
        for item in items:
            item.status = status_mapping.get(item.status, item.status)
            #print(item.title)
        all_items += items
        print(f"Page {page} terminée, {len(all_items)} items récupérés.")
        page += 1
        time.sleep(0)

    return get_data_frame_initial(all_items,category)

# Les fonctions get_user_rate et get_item_description restent inchangées pour l'instant
def get_user_rate(url) -> tuple:
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    rating_element = soup.select_one('.web_ui__Rating__rating[aria-label]')
    evaluations_element = soup.select_one('.web_ui__Text__text.web_ui__Text__body.web_ui__Text__left')
    rating = rating_element['aria-label'] if rating_element else 'No rating'
    evaluations_text = evaluations_element.text.strip() if evaluations_element else 'No evaluations'
    rating_match = re.search(r'(\d+(\.\d+)?) sur 5', rating)
    rating = rating_match.group(1) if rating_match else 'No rating'
    evaluations_match = re.search(r'(\d+)', evaluations_text)
    evaluations = evaluations_match.group(1) if evaluations_match else 'No evaluations'
    return evaluations, rating

def get_item_description(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    description = soup.select_one('.web_ui__Text__text.web_ui__Text__body.web_ui__Text__left.web_ui__Text__format')
    return description.text.strip() if description else 'No description'

if __name__ == "__main__":
    brandjson = 'brand.json'
    catalog ='catalog_correspondences.json'

    # Étape 1: Scraper les informations initiales
    initial_df = scrap_initial('', 'hommes', 'bonnet', 'Carhartt', 15)
    print("Informations initiales des items récupérées:")
    print(initial_df.head())

    # Ici, vous pouvez utiliser initial_df pour votre premier affichage

    # Étape 2: Scraper les informations supplémentaires (taux utilisateur et description)
    # Cette partie peut être exécutée ultérieurement ou en arrière-plan

    # Supprimer les doublons dans initial_df
    initial_df = initial_df.drop_duplicates(subset=['URL'], keep='first')

    detailed_data = []
    for index, row in initial_df.iterrows():
        print(f"Récupération des détails pour l'item: {row['Title']}")
        try:
            user_evaluations, user_rating = get_user_rate(row['user_url']) # Ajustement de l'URL pour le profil
            description = get_item_description(row['URL']) # Ajustement de l'URL pour l'item
            detailed_data.append({
                "URL": row['URL'],
                "NbEvalUser": user_evaluations,
                "UserRating": user_rating,
                "Description": description
            })
            time.sleep(0.1) # Pour ne pas surcharger le serveur lors des requêtes détaillées
        except Exception as e:
            print(f"Erreur lors de la récupération des détails pour {row['Title']}: {e}")
            detailed_data.append({
                "URL": row['URL'],
                "NbEvalUser": None,
                "UserRating": None,
                "Description": None
            })

    detailed_df = pd.DataFrame(detailed_data)

    # Fusionner les informations détaillées avec le DataFrame initial
    final_df = pd.merge(initial_df, detailed_df, on='URL', how='left')
    print("\nDataFrame final avec les détails:")
    print(final_df.head())

    # enregistrer le DataFrame final dans un fichier CSV
    final_df.to_csv('pantalonC.csv', index=False)

    # Maintenant, final_df contient toutes les informationssweats