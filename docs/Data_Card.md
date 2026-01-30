# Fiche de Données : Hospital Deterioration

> **Document auto-généré** - Dernière mise à jour: 30/01/2026 à 10:17

- **Auteur du document**: Paul-Henri DOURNEAU & Dorian MARTY
- **Date de création**: 09/01/2026
- **Version**: 2.0

---

## 📋 Informations Générales

| Propriété | Valeur |
|-----------|--------|
| **Nom du dataset** | Hospital Deterioration Hourly Panel |
| **Domaine** | Santé / Suivi Clinique |
| **Nombre d'entrées** | 417,866 |
| **Nombre de colonnes** | 28 |
| **Doublons** | 0 lignes |

### Objectif du Dataset

Ce jeu de données sert principalement à :
1. **Identifier les signes avant-coureurs** de la détérioration de l'état de santé des patients (choc septique, arrêt cardiaque, etc.)
2. **Estimer la durée d'hospitalisation** restante pour optimiser la gestion des lits

---

## 🏥 Provenance et Contexte

Ces données proviennent d'un **monitoring hospitalier continu**. Elles agrègent :
- **Signes vitaux** : fréquence cardiaque, pression sanguine, saturation O2
- **Résultats de laboratoire** : lactate, créatinine, CRP, hémoglobine
- **Scores cliniques** : SOFA, NEWS, score de risque de sepsis

| Caractéristique | Détail |
|-----------------|--------|
| **Période couverte** | Non spécifiée (échelle horaire) |
| **Granularité** | Horaire (`hour_from_admission`) |
| **Population** | Patients hospitalisés (adultes, 18-90 ans) |

---

## 📊 Qualité des Données

### Valeurs Manquantes

Certaines variables biologiques ne sont pas mesurées à chaque heure (prises de sang non horaires), 
ce qui explique les taux de valeurs manquantes.

![Valeurs Manquantes](./valeurs_manquantes.png)

### Répartition des Classes Cibles

![Distribution des Cibles](./distribution_cibles.png)

> ⚠️ **Déséquilibre de classes** : La classe "Détérioration" représente environ 21% des observations, 
> ce qui nécessite des techniques de rééquilibrage pour l'entraînement des modèles.

---

## 📖 Dictionnaire des Variables

| Variable | Type | Description / Plage Normale |
|---|---|---|
| `patient_id` | int64 | Identifiant unique du patient |
| `hour_from_admission` | int64 | Heures écoulées depuis l'admission |
| `heart_rate` | float64 | Fréquence cardiaque (battements/min) - Normal: 60-100 |
| `respiratory_rate` | float64 | Fréquence respiratoire (/min) - Normal: 12-20 |
| `spo2_pct` | float64 | Saturation en oxygène (%) - Normal: >95% |
| `temperature_c` | float64 | Température corporelle (°C) - Normal: 36.5-37.5 |
| `systolic_bp` | float64 | Pression artérielle systolique (mmHg) - Normal: 90-120 |
| `diastolic_bp` | float64 | Pression artérielle diastolique (mmHg) - Normal: 60-80 |
| `oxygen_device` | str | Type de dispositif d'oxygénation utilisé |
| `oxygen_flow` | float64 | Débit d'oxygène administré (L/min) |
| `mobility_score` | int64 | Score de mobilité du patient (0-4) |
| `nurse_alert` | int64 | Alerte infirmière déclenchée (0=Non, 1=Oui) |
| `wbc_count` | float64 | Numération des globules blancs (10³/µL) - Normal: 4-11 |
| `lactate` | float64 | Lactate sanguin (mmol/L) - Normal: <2 |
| `creatinine` | float64 | Créatinine (mg/dL) - Normal: 0.7-1.3 |
| `crp_level` | float64 | Protéine C-réactive (mg/L) - Normal: <10 |
| `hemoglobin` | float64 | Hémoglobine (g/dL) - Normal: 12-17 |
| `sepsis_risk_score` | float64 | Score de risque de sepsis (0-1) |
| `age` | int64 | Âge du patient (années) |
| `gender` | str | Sexe du patient (M/F) |
| `comorbidity_index` | int64 | Index de comorbidité (Charlson modifié) |
| `admission_type` | str | Type d'admission (Urgence, Programmée, etc.) |
| `baseline_risk_score` | float64 | Score de risque initial à l'admission (0-1) |
| `los_hours` | int64 | **TARGET** - Durée totale de séjour (heures) |
| `deterioration_event` | int64 | **TARGET** - Événement de détérioration (0=Non, 1=Oui) |
| `deterioration_within_12h_from_admission` | int64 | **TARGET** - Détérioration dans les 12h post-admission |
| `deterioration_hour` | int64 | Heure de l'événement de détérioration (-1 si aucun) |
| `deterioration_next_12h` | int64 | **TARGET** - Détérioration dans les 12h suivantes |


---

## 📈 Exploration Statistique

### Statistiques Descriptives

|  | Moyenne | Écart-Type | Min | Médiane | Max |
|---|---|---|---|---|---|
| patient_id | 4978.44 | 2889.2 | 1.0 | 4986.0 | 10000.0 |
| hour_from_admission | 24.1 | 16.94 | 0.0 | 21.0 | 71.0 |
| heart_rate | 89.26 | 21.13 | 40.0 | 86.33 | 180.0 |
| respiratory_rate | 20.18 | 6.85 | 8.0 | 19.33 | 45.0 |
| spo2_pct | 93.5 | 5.95 | 70.0 | 94.83 | 100.0 |
| temperature_c | 36.97 | 0.45 | 35.24 | 36.92 | 40.5 |
| systolic_bp | 113.4 | 17.11 | 70.0 | 114.83 | 184.56 |
| diastolic_bp | 70.52 | 10.67 | 40.0 | 71.47 | 110.0 |
| oxygen_flow | 7.97 | 16.25 | 0.0 | 0.0 | 56.19 |
| mobility_score | 2.28 | 0.93 | 0.0 | 2.0 | 4.0 |
| nurse_alert | 0.22 | 0.42 | 0.0 | 0.0 | 1.0 |
| wbc_count | 9.2 | 4.08 | 2.0 | 8.49 | 30.0 |
| lactate | 1.99 | 1.51 | 0.5 | 1.61 | 8.0 |
| creatinine | 1.32 | 0.67 | 0.4 | 1.19 | 4.5 |
| crp_level | 34.23 | 36.28 | 0.0 | 26.63 | 250.0 |
| hemoglobin | 13.28 | 1.16 | 7.0 | 13.36 | 17.0 |
| sepsis_risk_score | 0.49 | 0.22 | 0.02 | 0.47 | 1.0 |
| age | 53.86 | 20.95 | 18.0 | 54.0 | 90.0 |
| comorbidity_index | 4.0 | 2.6 | 0.0 | 4.0 | 8.0 |
| baseline_risk_score | 0.5 | 0.21 | 0.03 | 0.5 | 0.98 |
| los_hours | 49.2 | 15.99 | 12.0 | 52.0 | 72.0 |
| deterioration_event | 0.21 | 0.41 | 0.0 | 0.0 | 1.0 |
| deterioration_within_12h_from_admission | 0.03 | 0.17 | 0.0 | 0.0 | 1.0 |
| deterioration_hour | 5.53 | 14.52 | -1.0 | -1.0 | 70.0 |
| deterioration_next_12h | 0.05 | 0.23 | 0.0 | 0.0 | 1.0 |

### Distribution des Signes Vitaux

![Distributions Signes Vitaux](./distributions_signes_vitaux.png)

### Distribution des Analyses Biologiques

![Distributions Analyses Labo](./distributions_analyses_labo.png)

### Détection des Outliers

![Boxplots Outliers](./boxplots_outliers.png)

---

## 🔗 Analyse des Corrélations

### Matrice de Corrélation Complète

![Matrice de Corrélation](./heatmap_correlation.png)

### Corrélations avec la Cible (Détérioration)

Variables les plus corrélées avec `deterioration_event` :

- `deterioration_hour`: 0.872
- `lactate`: 0.588
- `spo2_pct`: -0.563
- `creatinine`: 0.531
- `crp_level`: 0.516

### Matrice Annotée (Variables Clés)

![Corrélations Annotées](./heatmap_correlation_annotated.png)

---

## 📝 Notes pour l'Utilisation

1. **Pré-traitement recommandé** :
   - Imputation des valeurs manquantes (moyenne ou médiane)
   - Encodage des variables catégorielles (`gender`, `oxygen_device`, `admission_type`)
   - Normalisation (StandardScaler) pour les algorithmes sensibles à l'échelle

2. **Variables à exclure de l'entraînement** :
   - `patient_id` (identifiant, risque de surapprentissage)
   - `deterioration_hour` (fuite d'information si on prédit la détérioration)

3. **Attention aux fuites de données** :
   - Certaines variables comme `deterioration_hour` contiennent implicitement la réponse

---

*Lien vers les scripts de génération : [datacard.py](./datacard.py)*
