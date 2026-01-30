"""
Data Card Generator - Analyse de Données Hospitalières
=======================================================
Ce script génère automatiquement une fiche de données (Data Card) complète
avec visualisations pour le dataset hospital_deterioration_hourly_panel.csv.

Auteur: Paul-Henri DOURNEAU & Dorian MARTY
Date: 30/01/2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def df_to_markdown(df, index=True):
    """Convert a DataFrame to markdown table format without tabulate."""
    if index:
        df = df.reset_index()
        df.columns = [''] + list(df.columns[1:])
    
    # Header
    header = '| ' + ' | '.join(str(col) for col in df.columns) + ' |'
    separator = '|' + '|'.join(['---' for _ in df.columns]) + '|'
    
    # Rows
    rows = []
    for _, row in df.iterrows():
        row_str = '| ' + ' | '.join(str(val) for val in row.values) + ' |'
        rows.append(row_str)
    
    return '\n'.join([header, separator] + rows)

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = r'c:\Users\Ph\Documents\.EPSI\Documentations\hospital_deterioration_hourly_panel.csv'
OUTPUT_DIR = r'c:\Users\Ph\Documents\.EPSI\Documentations'

# Créer le dossier de sortie si nécessaire
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Configuration du style des graphiques (français)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

print(f"[INFO] Chargement des données depuis {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f"[INFO] Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")

# =============================================================================
# EXPLORATION DES DONNÉES
# =============================================================================

print("[INFO] Calcul des statistiques descriptives...")

# Statistiques de base
n_rows, n_cols = df.shape
columns_info = df.dtypes.to_dict()
missing_values = df.isnull().sum()
missing_pct = (missing_values / len(df)) * 100
duplicates = df.duplicated().sum()

# Identification des colonnes par type
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Variables cibles identifiées
target_cols = ['deterioration_event', 'los_hours', 'deterioration_next_12h', 
               'deterioration_within_12h_from_admission', 'deterioration_hour']

# Variables physiologiques clés pour visualisation
vital_cols = ['heart_rate', 'respiratory_rate', 'spo2_pct', 'temperature_c', 
              'systolic_bp', 'diastolic_bp']
labo_cols = ['wbc_count', 'lactate', 'creatinine', 'crp_level', 'hemoglobin']
score_cols = ['sepsis_risk_score', 'baseline_risk_score', 'mobility_score']

# =============================================================================
# VISUALISATION 1 : MATRICE DE CORRÉLATION ANNOTÉE
# =============================================================================

print("[INFO] Génération de la matrice de corrélation...")

corr_matrix = df[numerical_cols].corr()

# Heatmap complète
fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Corrélation'},
            vmin=-1, vmax=1, ax=ax)
ax.set_title('Matrice de Corrélation - Variables Numériques', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, 'heatmap_correlation.png')
plt.savefig(heatmap_path, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Heatmap sauvegardée: {heatmap_path}")

# Heatmap annotée (subset des variables les plus importantes)
important_vars = vital_cols + labo_cols + ['deterioration_event', 'los_hours', 'age']
important_vars = [v for v in important_vars if v in df.columns]
corr_important = df[important_vars].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_important, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Corrélation'},
            vmin=-1, vmax=1, ax=ax, annot_kws={'size': 8})
ax.set_title('Corrélations - Variables Clés (annotées)', fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
heatmap_annotated_path = os.path.join(OUTPUT_DIR, 'heatmap_correlation_annotated.png')
plt.savefig(heatmap_annotated_path, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Heatmap annotée sauvegardée: {heatmap_annotated_path}")

# =============================================================================
# VISUALISATION 2 : DISTRIBUTIONS DES SIGNES VITAUX
# =============================================================================

print("[INFO] Génération des distributions des signes vitaux...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

vital_labels = {
    'heart_rate': 'Fréquence Cardiaque (bpm)',
    'respiratory_rate': 'Fréquence Respiratoire (/min)',
    'spo2_pct': 'Saturation O2 (%)',
    'temperature_c': 'Température (°C)',
    'systolic_bp': 'Pression Systolique (mmHg)',
    'diastolic_bp': 'Pression Diastolique (mmHg)'
}

for i, col in enumerate(vital_cols):
    if col in df.columns:
        ax = axes[i]
        data = df[col].dropna()
        
        # Histogramme avec KDE
        sns.histplot(data, kde=True, ax=ax, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {data.mean():.1f}')
        ax.axvline(data.median(), color='green', linestyle='-.', linewidth=2, label=f'Médiane: {data.median():.1f}')
        ax.set_xlabel(vital_labels.get(col, col), fontsize=10)
        ax.set_ylabel('Fréquence', fontsize=10)
        ax.set_title(col.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)

plt.suptitle('Distribution des Signes Vitaux', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
dist_vital_path = os.path.join(OUTPUT_DIR, 'distributions_signes_vitaux.png')
plt.savefig(dist_vital_path, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Distributions signes vitaux sauvegardées: {dist_vital_path}")

# =============================================================================
# VISUALISATION 3 : DISTRIBUTIONS DES ANALYSES BIOLOGIQUES
# =============================================================================

print("[INFO] Génération des distributions des analyses labo...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

labo_labels = {
    'wbc_count': 'Globules Blancs (10³/µL)',
    'lactate': 'Lactate (mmol/L)',
    'creatinine': 'Créatinine (mg/dL)',
    'crp_level': 'CRP (mg/L)',
    'hemoglobin': 'Hémoglobine (g/dL)'
}

for i, col in enumerate(labo_cols):
    if col in df.columns and i < len(axes):
        ax = axes[i]
        data = df[col].dropna()
        
        sns.histplot(data, kde=True, ax=ax, color='darkorange', alpha=0.7, edgecolor='white')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {data.mean():.2f}')
        ax.set_xlabel(labo_labels.get(col, col), fontsize=10)
        ax.set_ylabel('Fréquence', fontsize=10)
        ax.set_title(col.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)

# Masquer les axes inutilisés
for j in range(len(labo_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution des Analyses Biologiques', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
dist_labo_path = os.path.join(OUTPUT_DIR, 'distributions_analyses_labo.png')
plt.savefig(dist_labo_path, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Distributions analyses labo sauvegardées: {dist_labo_path}")

# =============================================================================
# VISUALISATION 4 : BOXPLOTS POUR DÉTECTION D'OUTLIERS
# =============================================================================

print("[INFO] Génération des boxplots pour détection d'outliers...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Boxplot signes vitaux
ax1 = axes[0]
vital_data = df[vital_cols].dropna()
vital_normalized = (vital_data - vital_data.mean()) / vital_data.std()
bp1 = ax1.boxplot([vital_normalized[col].dropna() for col in vital_cols], 
                   labels=[v.replace('_', '\n') for v in vital_cols],
                   patch_artist=True, showfliers=True)
for patch in bp1['boxes']:
    patch.set_facecolor('lightblue')
ax1.set_title('Boxplots des Signes Vitaux (Normalisés)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Valeur standardisée (Z-score)', fontsize=10)
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax1.axhline(y=3, color='red', linestyle='--', alpha=0.7, label='Seuil outlier (±3σ)')
ax1.axhline(y=-3, color='red', linestyle='--', alpha=0.7)
ax1.legend()

# Boxplot analyses labo
ax2 = axes[1]
labo_data = df[labo_cols].dropna()
labo_normalized = (labo_data - labo_data.mean()) / labo_data.std()
bp2 = ax2.boxplot([labo_normalized[col].dropna() for col in labo_cols],
                   labels=[v.replace('_', '\n') for v in labo_cols],
                   patch_artist=True, showfliers=True)
for patch in bp2['boxes']:
    patch.set_facecolor('lightsalmon')
ax2.set_title('Boxplots des Analyses Biologiques (Normalisés)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Valeur standardisée (Z-score)', fontsize=10)
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax2.axhline(y=3, color='red', linestyle='--', alpha=0.7, label='Seuil outlier (±3σ)')
ax2.axhline(y=-3, color='red', linestyle='--', alpha=0.7)
ax2.legend()

plt.tight_layout()
boxplot_path = os.path.join(OUTPUT_DIR, 'boxplots_outliers.png')
plt.savefig(boxplot_path, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Boxplots sauvegardés: {boxplot_path}")

# =============================================================================
# VISUALISATION 5 : MATRICE DES VALEURS MANQUANTES
# =============================================================================

print("[INFO] Génération de la matrice des valeurs manquantes...")

fig, ax = plt.subplots(figsize=(12, 6))
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

if len(missing_data) > 0:
    colors = plt.cm.Reds(missing_data / missing_data.max())
    bars = ax.barh(missing_data.index, missing_data.values, color=colors)
    ax.set_xlabel('Nombre de valeurs manquantes', fontsize=11)
    ax.set_title('Valeurs Manquantes par Variable', fontsize=14, fontweight='bold')
    
    # Ajouter les pourcentages
    for i, (val, name) in enumerate(zip(missing_data.values, missing_data.index)):
        pct = (val / len(df)) * 100
        ax.text(val + 100, i, f'{pct:.1f}%', va='center', fontsize=9)
else:
    ax.text(0.5, 0.5, 'Aucune valeur manquante détectée!', 
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.tight_layout()
missing_path = os.path.join(OUTPUT_DIR, 'valeurs_manquantes.png')
plt.savefig(missing_path, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Graphique valeurs manquantes sauvegardé: {missing_path}")

# =============================================================================
# VISUALISATION 6 : RÉPARTITION DE LA CIBLE (DÉTÉRIORATION)
# =============================================================================

print("[INFO] Génération de la répartition des classes cibles...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Deterioration event
ax1 = axes[0]
if 'deterioration_event' in df.columns:
    counts = df['deterioration_event'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    labels = ['Stable (0)', 'Détérioration (1)']
    explode = (0, 0.05)
    ax1.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, 
            explode=explode, shadow=True, startangle=90)
    ax1.set_title('Répartition Événements de Détérioration', fontsize=12, fontweight='bold')

# Distribution durée de séjour
ax2 = axes[1]
if 'los_hours' in df.columns:
    sns.histplot(df['los_hours'].dropna(), kde=True, ax=ax2, color='purple', alpha=0.7)
    ax2.axvline(df['los_hours'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Moyenne: {df["los_hours"].mean():.1f}h')
    ax2.set_xlabel('Durée de Séjour (heures)', fontsize=11)
    ax2.set_ylabel('Fréquence', fontsize=11)
    ax2.set_title('Distribution de la Durée de Séjour', fontsize=12, fontweight='bold')
    ax2.legend()

plt.tight_layout()
target_dist_path = os.path.join(OUTPUT_DIR, 'distribution_cibles.png')
plt.savefig(target_dist_path, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Distribution des cibles sauvegardée: {target_dist_path}")

# =============================================================================
# GÉNÉRATION DE LA DATA CARD (MARKDOWN)
# =============================================================================

print("[INFO] Génération de la Data Card au format Markdown...")

# Préparer les statistiques formatées
desc_stats = df.describe().T
desc_stats = desc_stats[['mean', 'std', 'min', '50%', 'max']]
desc_stats.columns = ['Moyenne', 'Écart-Type', 'Min', 'Médiane', 'Max']
desc_stats_markdown = df_to_markdown(desc_stats.round(2))

# Créer le dictionnaire des variables avec descriptions enrichies
variable_descriptions = {
    'patient_id': ('int64', 'Identifiant unique du patient'),
    'hour_from_admission': ('int64', 'Heures écoulées depuis l\'admission'),
    'heart_rate': ('float64', 'Fréquence cardiaque (battements/min) - Normal: 60-100'),
    'respiratory_rate': ('float64', 'Fréquence respiratoire (/min) - Normal: 12-20'),
    'spo2_pct': ('float64', 'Saturation en oxygène (%) - Normal: >95%'),
    'temperature_c': ('float64', 'Température corporelle (°C) - Normal: 36.5-37.5'),
    'systolic_bp': ('float64', 'Pression artérielle systolique (mmHg) - Normal: 90-120'),
    'diastolic_bp': ('float64', 'Pression artérielle diastolique (mmHg) - Normal: 60-80'),
    'oxygen_device': ('object', 'Type de dispositif d\'oxygénation utilisé'),
    'oxygen_flow': ('float64', 'Débit d\'oxygène administré (L/min)'),
    'mobility_score': ('int64', 'Score de mobilité du patient (0-4)'),
    'nurse_alert': ('int64', 'Alerte infirmière déclenchée (0=Non, 1=Oui)'),
    'wbc_count': ('float64', 'Numération des globules blancs (10³/µL) - Normal: 4-11'),
    'lactate': ('float64', 'Lactate sanguin (mmol/L) - Normal: <2'),
    'creatinine': ('float64', 'Créatinine (mg/dL) - Normal: 0.7-1.3'),
    'crp_level': ('float64', 'Protéine C-réactive (mg/L) - Normal: <10'),
    'hemoglobin': ('float64', 'Hémoglobine (g/dL) - Normal: 12-17'),
    'sepsis_risk_score': ('float64', 'Score de risque de sepsis (0-1)'),
    'age': ('int64', 'Âge du patient (années)'),
    'gender': ('object', 'Sexe du patient (M/F)'),
    'comorbidity_index': ('int64', 'Index de comorbidité (Charlson modifié)'),
    'admission_type': ('object', 'Type d\'admission (Urgence, Programmée, etc.)'),
    'baseline_risk_score': ('float64', 'Score de risque initial à l\'admission (0-1)'),
    'los_hours': ('int64', '**TARGET** - Durée totale de séjour (heures)'),
    'deterioration_event': ('int64', '**TARGET** - Événement de détérioration (0=Non, 1=Oui)'),
    'deterioration_within_12h_from_admission': ('int64', '**TARGET** - Détérioration dans les 12h post-admission'),
    'deterioration_hour': ('int64', 'Heure de l\'événement de détérioration (-1 si aucun)'),
    'deterioration_next_12h': ('int64', '**TARGET** - Détérioration dans les 12h suivantes')
}

# Générer le tableau des variables
variable_table = "| Variable | Type | Description / Plage Normale |\n|---|---|---|\n"
for col in df.columns:
    dtype = str(df[col].dtype)
    desc = variable_descriptions.get(col, (dtype, 'Variable mesurée'))[1]
    variable_table += f"| `{col}` | {dtype} | {desc} |\n"

# Calculer les corrélations les plus fortes avec la cible
if 'deterioration_event' in df.columns:
    target_corr = corr_matrix['deterioration_event'].abs().sort_values(ascending=False)
    top_correlations = target_corr[1:6]  # Top 5 (excluant la cible elle-même)
    corr_text = "\n".join([f"- `{var}`: {corr_matrix.loc[var, 'deterioration_event']:.3f}" 
                           for var in top_correlations.index])
else:
    corr_text = "Non disponible"

# Contenu de la Data Card
datacard_content = f"""# Fiche de Données : Hospital Deterioration

> **Document auto-généré** - Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y à %H:%M')}

- **Auteur du document**: Paul-Henri DOURNEAU & Dorian MARTY
- **Date de création**: 09/01/2026
- **Version**: 2.0

---

## 📋 Informations Générales

| Propriété | Valeur |
|-----------|--------|
| **Nom du dataset** | Hospital Deterioration Hourly Panel |
| **Domaine** | Santé / Suivi Clinique |
| **Nombre d'entrées** | {n_rows:,} |
| **Nombre de colonnes** | {n_cols} |
| **Doublons** | {duplicates} lignes |

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

{variable_table}

---

## 📈 Exploration Statistique

### Statistiques Descriptives

{desc_stats_markdown}

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

{corr_text}

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
"""

# Sauvegarder la Data Card
datacard_path = os.path.join(OUTPUT_DIR, 'Data_Card.md')
with open(datacard_path, 'w', encoding='utf-8') as f:
    f.write(datacard_content)

print(f"[OK] Data Card sauvegardée: {datacard_path}")
print("\n" + "="*60)
print("GÉNÉRATION DE LA DATA CARD TERMINÉE")
print("="*60)
print(f"Fichiers générés:")
print(f"  - {datacard_path}")
print(f"  - {heatmap_path}")
print(f"  - {heatmap_annotated_path}")
print(f"  - {dist_vital_path}")
print(f"  - {dist_labo_path}")
print(f"  - {boxplot_path}")
print(f"  - {missing_path}")
print(f"  - {target_dist_path}")