# To Bee or Not To Bee — ML Project

Projet de Machine Learning IG.2412 : classification d'insectes pollinisateurs (abeilles, bourdons, autres) à partir de 347 images haute résolution avec leurs masques de segmentation.

## Structure du projet

```
Machine learning/
├── data/
│   ├── train_images/      # Images 1 à 250 (.jpg/.png)
│   ├── train_masks/       # Masques correspondants
│   ├── test_images/       # Images 251 à 347 (fournies plus tard)
│   ├── test_masks/        # Masques correspondants
│   └── labels.xlsx        # Fichier Excel avec colonnes ID, bug type, species
├── notebooks/
│   └── to_bee_or_not_to_bee.ipynb   # Pipeline principal
├── src/
│   ├── features.py        # Extraction des features (III.1)
│   ├── data_loader.py     # Chargement images/masques + DataFrame
│   ├── visualization.py   # PCA, t-SNE, UMAP, distribution des classes (III.2)
│   └── models.py          # SVM, KNN, RF, KMeans, Agglomerative, DBSCAN, CNN (III.3)
├── outputs/               # Figures et CSV générés
├── report/                # Rapport PDF final
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Préparation des données

1. Placer les **images d'entraînement** (1 à 250) dans `data/train_images/`
2. Placer les **masques d'entraînement** dans `data/train_masks/` (mêmes noms ou IDs que les images)
3. Quand disponibles : placer les images de test dans `data/test_images/` et leurs masques dans `data/test_masks/`

## Utilisation

Ouvrir le notebook `notebooks/to_bee_or_not_to_bee.ipynb` et exécuter les cellules dans l'ordre. Les sections correspondent au cahier des charges :

- **Section 3** : III.1 Extraction des features (7 pts)
- **Section 4** : III.2 Visualisation des données (5 pts)
- **Section 5** : III.3 Méthodes ML/DL (8+1 pts)
- **Section 6** : Production du CSV final pour les images 251-347

## Features extraites

**Obligatoires (sujet III.1)**
- Ratio de pixels du bug / image totale
- Min, max, mean, median, std des canaux R, G, B dans le masque
- Symétrie de forme (horizontale et verticale)
- Symétrie couleur (horizontale et verticale)

**Additionnelles (au moins 2 requises, on en met plus)**
- Descripteurs de forme : aspect ratio, extent, eccentricity, solidity, compactness
- Statistiques HSV (mean/std de H, S, V) dans le masque
- Features de texture GLCM (contrast, homogeneity, energy, correlation)
- Couleur moyenne du fond (hors masque)

## Méthodes ML

- **2 supervisées non-DL, non-ensemble** : SVM (RBF), KNN
- **1 ensemble** : Random Forest
- **2 clustering** : KMeans, Agglomerative (+ DBSCAN en bonus)
- **DL optionnel** : CNN simple (à activer si TensorFlow est installé)

## Livrables (8 juin 2026, 12h30)

- Rapport PDF dans `report/`
- CSV final dans `outputs/submission.csv` (colonnes `ID`, `bug type`)
