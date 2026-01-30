"""
Documentation Technique - Générateur
====================================
Ce script génère automatiquement la documentation technique complète
du projet d'analyse de données hospitalières.

Auteur: Paul-Henri DOURNEAU & Dorian MARTY
Date: 30/01/2026
"""

import os
from datetime import datetime

OUTPUT_DIR = r'c:\Users\Ph\Documents\.EPSI\Documentations'

print("[INFO] Génération de la Documentation Technique Complète...")

doc_content = f"""# Documentation Technique
## Projet d'Analyse de Données Hospitalières

---

**Auteurs** : Paul-Henri DOURNEAU & Dorian MARTY  
**Date de création** : 09/01/2026  
**Dernière mise à jour** : {datetime.now().strftime('%d/%m/%Y')}  
**Version** : 2.0

---

## Table des Matières

1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Architecture du Projet](#2-architecture-du-projet)
3. [Les Données](#3-les-données)
4. [Les Modèles](#4-les-modèles)
5. [Guide d'Utilisation](#5-guide-dutilisation)
6. [Annexes](#6-annexes)

---

## 1. Introduction et Contexte

### 1.1 Problématique Médicale

Dans le contexte hospitalier, deux enjeux majeurs se posent quotidiennement :

1. **La détection précoce des détériorations** : Comment identifier les patients dont l'état de santé risque de se dégrader rapidement (choc septique, arrêt cardiaque, détresse respiratoire) ?

2. **L'optimisation de la gestion des lits** : Comment anticiper la durée de séjour des patients pour mieux planifier les admissions et les sorties ?

### 1.2 Objectifs du Projet

Ce projet vise à répondre à ces deux problématiques en développant des **modèles prédictifs** basés sur les données de monitoring hospitalier :

| Objectif | Type de Problème | Modèle Utilisé |
|----------|------------------|----------------|
| Prédire une détérioration | Classification binaire | Régression Logistique |
| Estimer la durée de séjour | Régression | Régression Linéaire |

### 1.3 Périmètre

- **Données** : Mesures horaires de signes vitaux, analyses biologiques, scores cliniques
- **Population** : Patients adultes hospitalisés (18-90 ans)
- **Horizon temporel** : Prédiction à l'instant T basée sur l'état actuel

---

## 2. Architecture du Projet

### 2.1 Diagramme de Flux

```mermaid
flowchart TB
    subgraph DONNÉES["📊 DONNÉES"]
        A[("hospital_deterioration_hourly_panel.csv<br/>50 Mo - 417,866 lignes")]
    end
    
    subgraph EXPLORATION["🔍 EXPLORATION"]
        B["datacard.py"]
        B1[["Statistiques descriptives"]]
        B2[["Valeurs manquantes"]]
        B3[["Matrice de corrélation"]]
        B4[["Distributions"]]
    end
    
    subgraph PREPARATION["⚙️ PRÉPARATION"]
        C["Imputation<br/>(moyenne/mode)"]
        D["Encodage<br/>(LabelEncoder)"]
        E["Normalisation<br/>(StandardScaler)"]
    end
    
    subgraph MODELISATION["🤖 MODÉLISATION"]
        F["modelcard.py"]
        F1["Régression Logistique<br/>(Classification)"]
        F2["Régression Linéaire<br/>(Régression)"]
    end
    
    subgraph OUTPUTS["📁 OUTPUTS"]
        G["Data_Card.md"]
        H["Model_Card_Classification.md"]
        I["Model_Card_Regression.md"]
        J["Visualisations PNG"]
        K["Modèles .joblib"]
    end
    
    A --> B
    B --> B1 & B2 & B3 & B4
    B1 & B2 & B3 & B4 --> G
    B3 & B4 --> J
    
    A --> C --> D --> E
    E --> F
    F --> F1 & F2
    F1 --> H
    F2 --> I
    F1 & F2 --> J & K
```

### 2.2 Structure des Fichiers

```
📁 C:\\Users\\Ph\\Documents\\.EPSI\\Documentations\\
│
├── 📄 hospital_deterioration_hourly_panel.csv   # Dataset source (50 Mo)
├── 📄 hospital_data_cleaned_normalized.csv      # Dataset nettoyé (189 Mo)
│
├── 🐍 datacard.py           # Script génération Data Card
├── 🐍 modelcard.py          # Script génération Model Cards
├── 🐍 generate_all_docs.py  # Script maître
│
├── 📑 Data_Card.md                    # Fiche des données
├── 📑 Model_Card_Classification.md   # Fiche modèle classification
├── 📑 Model_Card_Regression.md       # Fiche modèle régression
├── 📑 Documentation_Technique.md     # Ce document
├── 📑 transformation_log.md          # Log des transformations
│
├── 📊 heatmap_correlation.png            # Matrice corrélation
├── 📊 heatmap_correlation_annotated.png  # Matrice annotée
├── 📊 distributions_signes_vitaux.png    # Histogrammes vitaux
├── 📊 distributions_analyses_labo.png    # Histogrammes labo
├── 📊 boxplots_outliers.png              # Détection outliers
├── 📊 valeurs_manquantes.png             # Graphique manquants
├── 📊 distribution_cibles.png            # Répartition classes
├── 📊 confusion_matrix.png               # Matrice confusion
├── 📊 roc_curve.png                      # Courbe ROC
├── 📊 feature_importance.png             # Importance variables
├── 📊 predictions_vs_reality.png         # Prédictions régression
├── 📊 residuals_analysis.png             # Analyse résidus
├── 📊 regression_coefficients.png        # Coefficients régression
│
├── 🧠 model_classification.joblib   # Modèle sérialisé (classification)
└── 🧠 model_regression.joblib       # Modèle sérialisé (régression)
```

### 2.3 Technologies Utilisées

| Catégorie | Technologie | Version | Usage |
|-----------|-------------|---------|------|
| Langage | Python | 3.10+ | Traitement et modélisation |
| Données | Pandas | 2.x | Manipulation de données |
| Visualisation | Matplotlib | 3.x | Graphiques |
| Visualisation | Seaborn | 0.12+ | Graphiques statistiques |
| ML | Scikit-learn | 1.x | Modèles et métriques |
| Sérialisation | Joblib | 1.x | Sauvegarde des modèles |

---

## 3. Les Données

### 3.1 Source et Description

Le dataset `hospital_deterioration_hourly_panel.csv` contient des mesures **horaires** 
collectées auprès de patients hospitalisés.

> 📖 **Documentation détaillée** : [Data_Card.md](./Data_Card.md)

### 3.2 Visualisations d'Exploration

#### Matrice de Corrélation

![Matrice de Corrélation](./heatmap_correlation.png)

**Lecture** : Les couleurs indiquent la force et le sens de la corrélation :
- 🔴 Rouge intense = Corrélation positive forte (+1)
- 🔵 Bleu intense = Corrélation négative forte (-1)
- ⚪ Blanc = Pas de corrélation (0)

#### Corrélations Annotées (Variables Clés)

![Corrélations Annotées](./heatmap_correlation_annotated.png)

#### Distribution des Signes Vitaux

![Distributions Vitaux](./distributions_signes_vitaux.png)

#### Distribution des Analyses Biologiques

![Distributions Labo](./distributions_analyses_labo.png)

#### Détection des Outliers

![Boxplots](./boxplots_outliers.png)

### 3.3 Résumé des Transformations

Le dataset brut a subi les transformations suivantes :

| Étape | Méthode | Justification |
|-------|---------|---------------|
| **Valeurs manquantes** | Imputation par la moyenne (numériques) ou le mode (catégorielles) | Conserver le maximum de données |
| **Encodage** | LabelEncoder pour `gender`, `oxygen_device`, `admission_type` | Conversion en format numérique |
| **Normalisation** | StandardScaler (μ=0, σ=1) | Équilibrer l'influence des variables |

> 📋 **Log détaillé** : [transformation_log.md](./transformation_log.md)

---

## 4. Les Modèles

### 4.1 Modèle de Classification (Détérioration)

#### Objectif
Prédire si un patient va subir un événement de détérioration (choc, arrêt cardiaque, etc.)

#### Algorithme
- **Régression Logistique** avec pondération des classes (`class_weight='balanced'`)

#### Performances Clés

| Métrique | Valeur |
|----------|--------|
| Accuracy | ~89% |
| AUC-ROC | ~0.85 |
| Recall (Détérioration) | ~53% |

![Matrice de Confusion](./confusion_matrix.png)

![Courbe ROC](./roc_curve.png)

> 📖 **Documentation détaillée** : [Model_Card_Classification.md](./Model_Card_Classification.md)

### 4.2 Modèle de Régression (Durée de Séjour)

#### Objectif
Estimer la durée totale d'hospitalisation en heures

#### Algorithme
- **Régression Linéaire Multiple**

#### Performances Clés

| Métrique | Valeur |
|----------|--------|
| RMSE | ~14 heures |
| R² | ~0.23 |

![Prédictions vs Réalité](./predictions_vs_reality.png)

![Analyse des Résidus](./residuals_analysis.png)

> 📖 **Documentation détaillée** : [Model_Card_Regression.md](./Model_Card_Regression.md)

### 4.3 Importance des Variables

Les variables les plus influentes pour la prédiction sont :

![Feature Importance](./feature_importance.png)

![Coefficients Régression](./regression_coefficients.png)

---

## 5. Guide d'Utilisation

### 5.1 Prérequis

```bash
# Créer un environnement virtuel
python -m venv venv
venv\\Scripts\\activate  # Windows

# Installer les dépendances
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 5.2 Exécution des Scripts

#### Générer la Data Card

```bash
python datacard.py
```

**Outputs** :
- `Data_Card.md`
- 6 fichiers PNG de visualisation

#### Générer les Model Cards

```bash
python modelcard.py
```

**Outputs** :
- `Model_Card_Classification.md`
- `Model_Card_Regression.md`
- 6 fichiers PNG de visualisation
- 2 modèles `.joblib`
- `transformation_log.md`

#### Tout régénérer

```bash
python generate_all_docs.py
```

### 5.3 Utiliser les Modèles Entraînés

```python
import joblib
import pandas as pd

# Charger le modèle de classification
clf = joblib.load('model_classification.joblib')

# Charger le modèle de régression
reg = joblib.load('model_regression.joblib')

# Prédiction sur de nouvelles données
# ATTENTION : les données doivent être normalisées de la même façon
nouvelles_donnees = pd.DataFrame(...)  # Vos données
prediction_deterioration = clf.predict(nouvelles_donnees)
prediction_duree = reg.predict(nouvelles_donnees)
```

---

## 6. Annexes

### 6.1 Liens vers les Documents

| Document | Description | Lien |
|----------|-------------|------|
| **Data Card** | Fiche complète du dataset | [Data_Card.md](./Data_Card.md) |
| **Model Card Classification** | Détails du modèle de détérioration | [Model_Card_Classification.md](./Model_Card_Classification.md) |
| **Model Card Régression** | Détails du modèle de durée de séjour | [Model_Card_Regression.md](./Model_Card_Regression.md) |
| **Log Transformations** | Historique des transformations | [transformation_log.md](./transformation_log.md) |

### 6.2 Glossaire

| Terme | Définition |
|-------|------------|
| **AUC** | Area Under Curve - Aire sous la courbe ROC |
| **FN** | Faux Négatif - Cas positif prédit comme négatif |
| **FP** | Faux Positif - Cas négatif prédit comme positif |
| **LOS** | Length Of Stay - Durée de séjour |
| **MAE** | Mean Absolute Error - Erreur absolue moyenne |
| **RMSE** | Root Mean Square Error - Erreur quadratique moyenne |
| **ROC** | Receiver Operating Characteristic |
| **SOFA** | Sequential Organ Failure Assessment - Score de défaillance |
| **SpO2** | Saturation pulsée en oxygène |

### 6.3 Références

- Scikit-learn Documentation : https://scikit-learn.org/
- Pandas Documentation : https://pandas.pydata.org/
- Seaborn Documentation : https://seaborn.pydata.org/

---

*Document généré automatiquement par `technicalcard.py`*
"""

doc_path = os.path.join(OUTPUT_DIR, 'Documentation_Technique.md')
with open(doc_path, 'w', encoding='utf-8') as f:
    f.write(doc_content)

print(f"[OK] Documentation Technique sauvegardée: {doc_path}")
