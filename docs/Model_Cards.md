# Model Cards - Vue Combinée

Ce fichier combine les deux Model Cards pour référence.
Pour les versions détaillées, voir:

- [Model_Card_Classification.md](./Model_Card_Classification.md)
- [Model_Card_Regression.md](./Model_Card_Regression.md)

---

# Model Card : Prédiction de Détérioration

> **Document auto-généré** - Dernière mise à jour: 30/01/2026 à 10:17

- **Auteur**: Paul-Henri DOURNEAU & Dorian MARTY
- **Date de création**: 09/01/2026
- **Version**: 2.0

---

## 📋 Résumé du Modèle

| Propriété      | Valeur                                                          |
| -------------- | --------------------------------------------------------------- |
| **Type**       | Classification Binaire                                          |
| **Algorithme** | Régression Logistique                                           |
| **Objectif**   | Alerter le personnel soignant en cas de risque de détérioration |
| **Cible**      | `deterioration_event` (0 = Stable, 1 = Détérioration)           |

### Cas d'Usage Médical

Ce modèle a pour but d'**alerter le personnel soignant** en cas de risque imminent
de détérioration de l'état du patient. Une détérioration peut inclure :

- Choc septique
- Arrêt cardiaque
- Insuffisance respiratoire aiguë

> ⚠️ **ATTENTION** : Ce modèle est un outil d'aide à la décision.
> Il ne remplace pas le jugement clinique du médecin.

---

## 📊 Données d'Entraînement

| Métrique                | Valeur                                    |
| ----------------------- | ----------------------------------------- |
| **Dataset source**      | Hospital Deterioration (Version Nettoyée) |
| **Taille totale**       | 417,866 observations                      |
| **Split train/test**    | 80% / 20%                                 |
| **Taille entraînement** | 334,292 observations                      |
| **Taille test**         | 83,574 observations                       |

### Features Utilisées (22 variables)

Variables physiologiques et biologiques, **excluant** :

- `patient_id` (identifiant)
- Variables cibles (éviter fuite de données)

Lien vers le détail des features : [Data_Card.md](./Data_Card.md)

---

## ⚙️ Hyperparamètres

| Paramètre      | Valeur   | Justification                        |
| -------------- | -------- | ------------------------------------ |
| `max_iter`     | 1000     | Assurer la convergence               |
| `solver`       | lbfgs    | Par défaut, performant               |
| `C`            | 1.0      | Régularisation standard              |
| `class_weight` | balanced | Compenser le déséquilibre de classes |
| `random_state` | 42       | Reproductibilité                     |

---

## 📈 Performance et Analyse

### Métriques Globales

| Métrique     | Valeur |
| ------------ | ------ |
| **Accuracy** | 82.9%  |
| **AUC-ROC**  | 0.877  |

### Rapport de Classification

```
              precision    recall  f1-score   support

           0       0.92      0.86      0.89     66006
           1       0.58      0.72      0.64     17568

    accuracy                           0.83     83574
   macro avg       0.75      0.79      0.76     83574
weighted avg       0.85      0.83      0.84     83574

```

### Matrice de Confusion

![Matrice de Confusion](./confusion_matrix.png)

**Interprétation** :

- **TN (Vrais Négatifs)** : 56,663 patients stables correctement identifiés
- **TP (Vrais Positifs)** : 12,655 détériorations correctement détectées
- **FP (Faux Positifs)** : 9,343 fausses alertes
- **FN (Faux Négatifs)** : 4,913 détériorations manquées ⚠️

> 🔴 **Point Critique** : En milieu médical, les **Faux Négatifs** sont plus graves que les Faux Positifs.
> Un FN signifie qu'on rate une urgence potentielle.

### Courbe ROC

![Courbe ROC](./roc_curve.png)

Un AUC de **0.877** indique une capacité discriminante bonne.

---

## 🔍 Importance des Variables

![Feature Importance](./feature_importance.png)

### Top 5 Variables Prédictives

| Rang | Variable              | Importance |
| ---- | --------------------- | ---------- |
| 1    | `lactate`             | 1.271      |
| 2    | `spo2_pct`            | 0.943      |
| 3    | `hour_from_admission` | 0.728      |
| 4    | `comorbidity_index`   | 0.635      |
| 5    | `crp_level`           | 0.523      |

---

## 💡 Recommandations

### Pour améliorer le modèle

1. **Augmenter le rappel** : Ajuster le seuil de décision (actuellement 0.5) vers 0.3-0.4
   pour réduire les Faux Négatifs au prix de plus de Faux Positifs
2. **Tester des modèles non-linéaires** : Random Forest, XGBoost
3. **Feature engineering** : Créer des variables temporelles (tendances)

### Limites connues

- Entraîné sur un seul établissement hospitalier
- Ne prend pas en compte l'historique du patient au-delà des mesures horaires
- Performance dépendante de la qualité des données entrantes

---

## 📁 Fichiers Associés

- **Modèle sérialisé** : [model_classification.joblib](./model_classification.joblib)
- **Data Card** : [Data_Card.md](./Data_Card.md)
- **Log transformations** : [transformation_log.md](./transformation_log.md)
- **Script de génération** : [modelcard.py](./modelcard.py)

---

# Model Card : Estimation Durée de Séjour

> **Document auto-généré** - Dernière mise à jour: 30/01/2026 à 10:17

- **Auteur**: Paul-Henri DOURNEAU & Dorian MARTY
- **Date de création**: 09/01/2026
- **Version**: 2.0

---

## 📋 Résumé du Modèle

| Propriété      | Valeur                                    |
| -------------- | ----------------------------------------- |
| **Type**       | Régression                                |
| **Algorithme** | Régression Linéaire Multiple              |
| **Objectif**   | Estimer la durée totale d'hospitalisation |
| **Cible**      | `los_hours` (heures)                      |

### Cas d'Usage Hospitalier

Ce modèle permet d'**anticiper la gestion des lits** en prédisant la durée totale
d'hospitalisation d'un patient en fonction de son état clinique actuel.

**Applications** :

- Planification des sorties
- Optimisation de l'occupation des lits
- Estimation des ressources nécessaires

---

## 📊 Données d'Entraînement

| Métrique                    | Valeur                                    |
| --------------------------- | ----------------------------------------- |
| **Dataset source**          | Hospital Deterioration (Version Nettoyée) |
| **Taille totale**           | 417,866 observations                      |
| **Split train/test**        | 80% / 20%                                 |
| **Durée moyenne (test)**    | 49.2 heures                               |
| **Écart-type durée (test)** | 16.0 heures                               |

Lien vers le détail des features : [Data_Card.md](./Data_Card.md)

---

## ⚙️ Paramètres du Modèle

| Paramètre          | Valeur                     |
| ------------------ | -------------------------- |
| **Algorithme**     | LinearRegression (sklearn) |
| **Régularisation** | Aucune                     |
| **Intercept**      | 49.1930                    |

---

## 📈 Performance et Analyse

### Métriques de Régression

| Métrique | Valeur       | Interprétation             |
| -------- | ------------ | -------------------------- |
| **RMSE** | 14.02 heures | Erreur moyenne quadratique |
| **MAE**  | 11.52 heures | Erreur moyenne absolue     |
| **R²**   | 0.2300       | Variance expliquée (23.0%) |

### Prédictions vs Valeurs Réelles

![Prédictions vs Réalité](./predictions_vs_reality.png)

**Interprétation** :

- Un R² de 0.23 signifie que le modèle explique **23.0%** de la variance
- C'est un score modeste, typique sur des données médicales complexes
- L'erreur moyenne est de **11.5 heures** (environ 0.5 jours)

### Analyse des Résidus

![Analyse des Résidus](./residuals_analysis.png)

**Observations** :

- Distribution des résidus centrée autour de zéro ✓
- Pas d'hétéroscédasticité visible

---

## 🔍 Coefficients du Modèle

![Coefficients de Régression](./regression_coefficients.png)

### Interprétation des Coefficients

- **Coefficient positif** (vert) : Augmente la durée de séjour prédite
- **Coefficient négatif** (rouge) : Diminue la durée de séjour prédite

### Top 5 Variables Influentes

| Rang | Variable              | Coefficient | Effet   |
| ---- | --------------------- | ----------- | ------- |
| 1    | `baseline_risk_score` | 0.003       | ↑ Durée |
| 2    | `comorbidity_index`   | 0.160       | ↑ Durée |
| 3    | `nurse_alert`         | -0.205      | ↓ Durée |
| 4    | `oxygen_device`       | 0.116       | ↑ Durée |
| 5    | `temperature_c`       | 0.032       | ↑ Durée |

---

## 💡 Recommandations

### Pour améliorer le modèle

1. **Modèles non-linéaires** : Random Forest ou XGBoost captureront mieux les interactions
2. **Feature engineering** : Inclure des variables temporelles (heure du jour, jour de la semaine)
3. **Régularisation** : Tester Ridge ou Lasso pour réduire l'overfitting

### Limites connues

- Modèle linéaire : ne capture pas les relations complexes
- Prédictions peuvent être négatives (non bornées)
- Performance variable selon le type d'admission

---

## 📁 Fichiers Associés

- **Modèle sérialisé** : [model_regression.joblib](./model_regression.joblib)
- **Data Card** : [Data_Card.md](./Data_Card.md)
- **Log transformations** : [transformation_log.md](./transformation_log.md)
- **Script de génération** : [modelcard.py](./modelcard.py)
