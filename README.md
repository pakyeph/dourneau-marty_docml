# Hospital Data Analysis & Documentation Generator

Projet d'analyse de données hospitalières générant automatiquement des Data Cards, Model Cards et une Documentation Technique complète.

**Auteurs** : Paul-Henri DOURNEAU & Dorian MARTY  
**Date** : Janvier 2026

## 📋 Description

Ce projet vise à analyser un jeu de données de monitoring hospitalier (`hospital_deterioration_hourly_panel.csv`) pour :
1. **Explorer les données** : Statistiques, distributions, corrélations.
2. **Entraîner des modèles** :
   - **Classification** : Prédire les événements de détérioration (`deterioration_event`).
   - **Régression** : Estimer la durée de séjour hospitalier (`los_hours`).
3. **Générer de la documentation** : Création automatique de rapports au format Markdown (Data Card, Model Cards, Documentation Technique).

## 📂 Structure du Projet

```
dourneau-marty_docml/
├── data/       # Contient les données (CSV)
├── docs/       # Documentation générée (Data Card, Model Cards...)
├── figures/    # Graphiques générés (PNG)
├── models/     # Modèles entraînés (.joblib)
├── scripts/    # Scripts Python
│   ├── datacard.py           # Génère la Data Card et les visualisations
│   ├── modelcard.py          # Entraîne les modèles et génère les Model Cards
│   ├── technicalcard.py      # Génère la Documentation Technique
│   └── generate_all_docs.py  # Script maître pour tout exécuter
├── requirements.txt
└── README.md
```

## ⚙️ Installation

1. Assurez-vous d'avoir Python 3.10+ installé.
2. Installez les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

## 🚀 Utilisation

### Exécution automatique
Pour générer toute la documentation et entraîner les modèles en une seule fois :

```bash
cd scripts
python generate_all_docs.py
```

### Exécution manuelle
Vous pouvez lancer chaque script individuellement :

```bash
# Génération de l'analyse exploratoire (Data Card)
python datacard.py

# Entraînement des modèles et Model Cards
python modelcard.py

# Génération de la documentation technique
python technicalcard.py
```

## ⚠️ Configuration Importante

Les scripts actuels contiennent des chemins codés en dur pointant vers `c:\Users\Ph\Documents\.EPSI\Documentations`. 

Avant l'exécution, vous devrez peut-être **modifier les variables `INPUT_FILE` et `OUTPUT_DIR`** au début des scripts (`scripts/*.py`) pour correspondre à votre structure de dossiers actuelle, par exemple :

```python
# Exemple de modification dans les scripts :
INPUT_FILE = r'../data/hospital_deterioration_hourly_panel.csv'
OUTPUT_DIR = r'../docs'
```
