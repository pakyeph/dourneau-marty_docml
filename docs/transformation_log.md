# Log des Transformations de Données

**Date**: 30/01/2026 à 10:17

## Étapes de Prétraitement

### 1. Gestion des Valeurs Manquantes

- **Stratégie numériques**: Imputation par la moyenne
- **Stratégie catégorielles**: Imputation par le mode (valeur la plus fréquente)
- **Justification**: Conserver le maximum de données sans introduire de biais significatif

### 2. Encodage des Variables Catégorielles

Variables encodées avec LabelEncoder:

- `oxygen_device`: 5 classes
- `gender`: 2 classes
- `admission_type`: 3 classes

### 3. Normalisation (StandardScaler)

- **Méthode**: Centrage (moyenne=0) et réduction (écart-type=1)
- **Variables normalisées**: 22 features
- **Justification**: Nécessaire pour la régression logistique (sensible à l'échelle)

### 4. Variables Exclues de l'Entraînement

- `deterioration_event`
- `los_hours`
- `patient_id`
- `deterioration_hour`
- `deterioration_next_12h`
- `deterioration_within_12h_from_admission`

## Fichier de Sortie

- **Chemin**: `c:\Users\Ph\Documents\.EPSI\Documentations\hospital_data_cleaned_normalized.csv`
- **Taille**: 417866 lignes × 28 colonnes
