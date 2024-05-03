# Projet de Détection de Tumeurs Cérébrales par IRM

Ce projet vise à développer des modèles de machine learning pour la classification des tumeurs cérébrales à partir d'images par résonance magnétique (IRM). Le code Python fourni couvre plusieurs aspects du projet, de la collecte des données à l'évaluation des modèles.

## Prérequis et Installation

- Assurez-vous d'avoir Python 3.11.X installé sur votre ordinateur.
- Installez les dépendances en utilisant `pip install -r requirements.txt`

package :
- tensorflow
- opencv-python
- matplotlib
- pandas
- Pillow
- Pillow
- numpy
- requests
- tensorflow-addons


## Architecture des dossiers
```
Projet_ML\ # Dossier principal du projet
    code\  # 3mplacement des scripts python
    data_aug\ # Données augmentées
		IRM\
			oui\
			non\
		test\
			oui\
			non\
    test\ # Données de test
		oui_test\
		non_test\
    test_final\
		img0.png\ 
		img1.png\ 
		img2.png\
		img3.png\
		img4.png\
		img5.png\ 
```

## Description du Projet

- **Data Augmentation et Collecte de Données**
  - Les images IRM sont collectées sur kaggle voici le lien : - -https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection/code.
  - Les données sont augmentées pour renforcer la capacité du modèle.

- **Random Forest Classifier**
  - Un modèle Random Forest est entraîné et évalué pour la classification des tumeurs cérébrales.

- **Modèle CNN (Convolutional Neural Network)**
  - Un modèle CNN est construit et entraîné pour la classification des tumeurs cérébrales.
  
- **Modèle VGG16**
  - Un modèle pré-entraîné VGG16 est utilisé pour la classification des tumeurs cérébrales.

- **Évaluation des Modèles**
  - L'évaluation des modèles est effectuée avec des métriques telles que l'exactitude, la matrice de confusion et le rapport de classification.

- **Visualisation des Résultats**
  - Des graphiques sont générés pour visualiser l'entraînement et la validation des modèles.

## Description fonctionnelle

**Collecte et Augmentation des Données**
   - Assurez-vous d'avoir les données images IRM dans le répertoire `IRM,test`.

A partir du dossier `code`, ouvrez une console **Cmder ou terminal ** et exectuer le code `data_augmentation_projet.py`en utilisant la commend suivant :
```bash
pyhton data_augmentation_projet.py
```
le script prend les images du dossier `IRM` et `test`, les augmente, et les sauvegarde dans le dossier `data_aug`

**Entraînement et évaluation des modèles**
pour ce faire, il suffit de lancer le script `projet_final.py` avec la command suivant
```bash
pyhton projet_final.py
```
Il commence par charger et prétraiter un ensemble d'images, puis divise les données en ensembles d'entraînement et de validation. Ensuite, il entraîne un modèle Random Forest pour la classification. Le code construit également un modèle VGG16 pré-entraîné et un modèle CNN, les entraîne, évalue leurs performances sur des données de test, et sauvegarde les modèles. Enfin, il effectue des prédictions sur de nouvelles images et affiche les résultats, tout en fournissant des visualisations telles que des matrices de confusion et des graphiques de précision.



## Remarques

- Les modèles entraînés sont sauvegardés pour une utilisation ultérieure. 
- VVoici le lien vers ces modèles https://we.tl/t-Tx33Ds50PC
- Attention!!!, c'est un lien WeTransfer valable seulement une semaine 

## Auteur

BERKANI Yacine

---
