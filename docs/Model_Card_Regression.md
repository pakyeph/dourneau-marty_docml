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
