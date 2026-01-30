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
