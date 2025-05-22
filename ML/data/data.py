import pandas as pd
import os
import json

file_list = ['manteaux','casquettes','sweats'] # Ajouter les autres fichiers ici


def load_and_merge_datasets(file_list):
    dataframes = []
    for file in file_list:
        print(f"Processing file: {file}")
        df = pd.read_csv(f"ML/data/{file}.csv")
        df['Type'] = file
        dataframes.append(df)
    merged_data = pd.concat(dataframes, ignore_index=True)
    return merged_data

# Chargement du dataset
data = load_and_merge_datasets(file_list)

# Afficher la taille du dataframe
print(f"Taille du dataframe: {data.shape}")

#Title,Brand,Price,ItemStatus,NbFavourite,UserRating,NbEvalUser,Description,URL

# Calcul des métriques
metrics = {
'mean_price': data['Price'].mean(),
'median_price': data['Price'].median(),
'max_price': data['Price'].max(),
'min_price': data['Price'].min(),
'std_dev_price': data['Price'].std()
}

# Remplacer les valeurs "no evaluation" par NaN
data['NbEvalUser'] = data['NbEvalUser'].replace('No evaluations', pd.NA)
data['UserRating'] = data['UserRating'].replace('No rating', pd.NA)
data['Description'] = data['Description'].replace('No description', pd.NA)

data['NbEvalUser'] = pd.to_numeric(data['NbEvalUser'], errors='coerce')
data['UserRating'] = pd.to_numeric(data['UserRating'], errors='coerce')


def filter_group(x):
    print(f"Group: {x.name}, Size: {len(x)}")
    return x[
        (x['Price'] < x['Price'].quantile(0.95)) &
        (x['NbFavourite'] < x['NbFavourite'].quantile(0.95))
    ]

def nettoyer_outliers(df):
    # Vérifier le nombre de types différents dans le dataset
    unique_types = df['Type'].nunique()
    print(f"Nombre de types différents dans le dataset: {unique_types}")

    #Nettoyer les outliers de prix et de favoris par marque et type

    df = df.groupby(['Brand', 'Type']).apply(lambda x: x[
        (x['Price'] < x['Price'].quantile(0.90)) &
        (x['NbFavourite'] < x['NbFavourite'].quantile(0.95))
    ]).reset_index(drop=True)

    df = df.groupby(['Brand', 'Type']).apply(filter_group).reset_index(drop=True)
    # print(df.shape)
    return df

def encode_df(df):
    # Charger le fichier JSON contenant les IDs des marques
    # with open('../scraping/brand.json', 'r') as f:
    #     brand_data = json.load(f)
    #     brand_ids = {brand['title']: brand['id'] for brand in brand_data}

    # # Encodage des marques
    # df['Brand'] = df['Brand'].map(brand_ids)
    return df


# Fonction de labellisation automatique
def labeliser_bonne_affaire(df):
    # Calculer les métriques PAR MARQUE ET TYPE
    # Ensure no NaN values in the grouping columns and Price column
    df = df.dropna(subset=['Brand', 'Type', 'Price']).copy()
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['NbFavourite'] = pd.to_numeric(df['NbFavourite'], errors='coerce')

    # Calculer les métriques PAR MARQUE ET TYPE
    df['prix_median_groupe'] = df.groupby(['Brand', 'Type'])['Price'].transform('median')
    df['favoris_median_groupe'] = df.groupby(['Brand', 'Type'])['NbFavourite'].transform('median')

    for (brand, type_), group in df.groupby(['Brand', 'Type']):
        median_price = group['Price'].median()
        print(f"Brand: {brand}, Type: {type_}, Median Price: {median_price}")
    
    
    # Calculer le z-score du prix PAR GROUPE
    df['z_score_prix'] = df.groupby(['Brand', 'Type'])['Price'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    def est_bonne_affaire(row):
        
        # Règles relatives au groupe
        favoris_ok = row['NbFavourite'] > 1.1 * row['favoris_median_groupe']
        z_score_ok = row['z_score_prix'] < -0.5  # La règle basée sur le z-score suffit pour évaluer le prix
        
        # Règles absolues
        etat_ok = row['ItemStatus'] in [2, 3, 4]
        rating_ok = row['UserRating'] > 4.0 and row['NbEvalUser'] > 20
        
        return int(favoris_ok and z_score_ok and etat_ok and rating_ok)
    
    df['bonne_affaire'] = df.apply(est_bonne_affaire, axis=1)
    return df


# Convertir la colonne ItemStatus en entiers
data['ItemStatus'] = pd.to_numeric(data['ItemStatus'], errors='coerce')


# Appliquer la fonction de nettoyage des outliers
data = nettoyer_outliers(data)

# Appliquer la fonction d'encodage
data = encode_df(data)

# Appliquer la labellisation
df = labeliser_bonne_affaire(data)

# Calculer la part de bonnes affaires dans le dataset
total_rows = len(df)
bonne_affaire_count = df['bonne_affaire'].sum()
bonne_affaire_ratio = bonne_affaire_count / total_rows

print(f"Part de bonnes affaires sur le dataset total: {bonne_affaire_ratio:.2%}")

#print les liens ainsi que le label de bonne affaire

print("\n\n\n______________________________________________________\n\n\n")

# Sous-échantillonnage par groupe
grouped = df.groupby(['Brand', 'Type'])
balanced_groups = []

for (brand, type_), group in grouped:
    bonnes_affaires = group[group['bonne_affaire'] == 1]
    non_bonnes_affaires = group[group['bonne_affaire'] == 0]

    # Calculer la taille cible pour atteindre une différence max de 30% entre les deux classes
    target_bonnes_affaires_ratio = 0.35
    target_total_size = len(bonnes_affaires) / target_bonnes_affaires_ratio
    target_non_bonnes_affaires_size = int(target_total_size - len(bonnes_affaires))

    # Sous-échantillonner la classe majoritaire
    if len(non_bonnes_affaires) > target_non_bonnes_affaires_size:
        non_bonnes_affaires_sampled = non_bonnes_affaires.sample(n=target_non_bonnes_affaires_size, random_state=42)
    else:
        non_bonnes_affaires_sampled = non_bonnes_affaires

    # Combiner les deux classes
    balanced_group = pd.concat([bonnes_affaires, non_bonnes_affaires_sampled])
    balanced_groups.append(balanced_group)

# Combiner tous les groupes équilibrés
df_balanced = pd.concat(balanced_groups)

# Mélanger les données
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Taille du dataset équilibré: {df_balanced.shape}")

# Calculer la part de bonnes affaires dans le dataset
total_rows = len(df_balanced)
bonne_affaire_count = df_balanced['bonne_affaire'].sum()
bonne_affaire_ratio = bonne_affaire_count / total_rows

print(f"Part de bonnes affaires dans le dataset: {bonne_affaire_ratio:.2%}")


# Afficher les statistiques par groupe
for (brand, type_), group in df.groupby(['Brand', 'Type']):
    print(f"\n\nGroupe {brand} - {type_}:")
    print(f"- Médiane prix: {group['prix_median_groupe'].iloc[0]}€")
    print(f"- Bonnes affaires: {group['bonne_affaire'].mean():.0%}")
    print(f"- Exemple: {group.iloc[0]['URL']}")




if len(file_list) > 1:
    # Créer le nom du fichier
    new_csv_file_path = "ML/data/merged.csv"

    # Sauvegarder le nouveau dataset
    #print(f"Current working directory: {os.getcwd()}")
    df_balanced.to_csv(new_csv_file_path, index=False)

else:
    # Obtenir le nom du fichier sans l'extension
    file_name = os.path.splitext(os.path.basename(file_list[0]))[0]

    # Créer le nom du nouveau fichier
    new_csv_file_path = f"ML/data/{file_name}_labelise.csv"

    # Sauvegarder le nouveau dataset
    df_balanced.to_csv(new_csv_file_path, index=False)