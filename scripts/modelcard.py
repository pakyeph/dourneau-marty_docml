"""
Model Card Generator - Analyse de Données Hospitalières
========================================================
Ce script entraîne les modèles de machine learning et génère automatiquement
les Model Cards avec visualisations des performances.

Auteur: Paul-Henri DOURNEAU & Dorian MARTY
Date: 30/01/2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (classification_report, mean_squared_error, r2_score, 
                             confusion_matrix, roc_curve, auc, precision_recall_curve,
                             mean_absolute_error)
import joblib
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = r'c:\Users\Ph\Documents\.EPSI\Documentations\hospital_deterioration_hourly_panel.csv'
OUTPUT_DIR = r'c:\Users\Ph\Documents\.EPSI\Documentations'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150

print(f"[INFO] Chargement des données depuis {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f"[INFO] Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")

# =============================================================================
# PRÉPARATION DES DONNÉES
# =============================================================================

print("[INFO] Nettoyage et normalisation des données...")
df_clean = df.copy()

# 1. Gestion des valeurs manquantes
# Stratégie: Moyenne pour numériques, mode pour catégorielles
for col in df_clean.columns:
    if df_clean[col].dtype in ['float64', 'int64']:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    else:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# 2. Encodage des variables catégorielles
categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    encoders[col] = le

# 3. Définition des features (excluant les targets et identifiants)
excluded_cols = ['deterioration_event', 'los_hours', 'patient_id', 
                 'deterioration_hour', 'deterioration_next_12h', 
                 'deterioration_within_12h_from_admission']
feature_cols = [c for c in df_clean.columns if c not in excluded_cols]

# 4. Normalisation
scaler = StandardScaler()
df_clean[feature_cols] = scaler.fit_transform(df_clean[feature_cols])

# Sauvegarde des données nettoyées
cleaned_data_path = os.path.join(OUTPUT_DIR, 'hospital_data_cleaned_normalized.csv')
df_clean.to_csv(cleaned_data_path, index=False)
print(f"[OK] Données nettoyées sauvegardées: {cleaned_data_path}")

# Log des transformations
transformation_log = f"""# Log des Transformations de Données

**Date**: {datetime.now().strftime('%d/%m/%Y à %H:%M')}

## Étapes de Prétraitement

### 1. Gestion des Valeurs Manquantes
- **Stratégie numériques**: Imputation par la moyenne
- **Stratégie catégorielles**: Imputation par le mode (valeur la plus fréquente)
- **Justification**: Conserver le maximum de données sans introduire de biais significatif

### 2. Encodage des Variables Catégorielles
Variables encodées avec LabelEncoder:
{chr(10).join([f'- `{col}`: {len(encoders[col].classes_)} classes' for col in encoders])}

### 3. Normalisation (StandardScaler)
- **Méthode**: Centrage (moyenne=0) et réduction (écart-type=1)
- **Variables normalisées**: {len(feature_cols)} features
- **Justification**: Nécessaire pour la régression logistique (sensible à l'échelle)

### 4. Variables Exclues de l'Entraînement
{chr(10).join([f'- `{col}`' for col in excluded_cols])}

## Fichier de Sortie
- **Chemin**: `{cleaned_data_path}`
- **Taille**: {df_clean.shape[0]} lignes × {df_clean.shape[1]} colonnes
"""

log_path = os.path.join(OUTPUT_DIR, 'transformation_log.md')
with open(log_path, 'w', encoding='utf-8') as f:
    f.write(transformation_log)
print(f"[OK] Log de transformation sauvegardé: {log_path}")

# =============================================================================
# MODÈLE 1 : CLASSIFICATION (Détérioration)
# =============================================================================

print("\n" + "="*60)
print("MODÈLE 1 : CLASSIFICATION - Prédiction de Détérioration")
print("="*60)

X_cls = df_clean[feature_cols]
y_cls = df_clean['deterioration_event']

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

print(f"[INFO] Entraînement: {len(X_train_cls)} samples")
print(f"[INFO] Test: {len(X_test_cls)} samples")
print(f"[INFO] Répartition classes train: {dict(y_train_cls.value_counts())}")

# Entraînement
clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42)
clf.fit(X_train_cls, y_train_cls)

# Prédictions
y_pred_cls = clf.predict(X_test_cls)
y_proba_cls = clf.predict_proba(X_test_cls)[:, 1]

# Métriques
cls_report = classification_report(y_test_cls, y_pred_cls)
cls_report_dict = classification_report(y_test_cls, y_pred_cls, output_dict=True)

print("\n[RÉSULTATS] Classification Report:")
print(cls_report)

# Sauvegarder le modèle
model_cls_path = os.path.join(OUTPUT_DIR, 'model_classification.joblib')
joblib.dump(clf, model_cls_path)
print(f"[OK] Modèle sauvegardé: {model_cls_path}")

# --- VISUALISATIONS CLASSIFICATION ---

# 1. Matrice de Confusion
print("[INFO] Génération de la matrice de confusion...")
cm = confusion_matrix(y_test_cls, y_pred_cls)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Stable (0)', 'Détérioration (1)'],
            yticklabels=['Stable (0)', 'Détérioration (1)'],
            annot_kws={'size': 14})
ax.set_xlabel('Prédiction', fontsize=12)
ax.set_ylabel('Réalité', fontsize=12)
ax.set_title('Matrice de Confusion - Modèle de Classification', fontsize=14, fontweight='bold')

# Ajouter les taux
tn, fp, fn, tp = cm.ravel()
text = f"TN={tn:,} | FP={fp:,}\nFN={fn:,} | TP={tp:,}"
ax.text(1.5, -0.15, f"Précision Détérioration: {tp/(tp+fp):.1%}  |  Rappel Détérioration: {tp/(tp+fn):.1%}", 
        ha='center', fontsize=10, transform=ax.transAxes)

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Matrice de confusion sauvegardée: {cm_path}")

# 2. Courbe ROC
print("[INFO] Génération de la courbe ROC...")
fpr, tpr, thresholds = roc_curve(y_test_cls, y_proba_cls)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=11)
ax.set_ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=11)
ax.set_title('Courbe ROC - Prédiction de Détérioration', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, 'roc_curve.png')
plt.savefig(roc_path, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Courbe ROC sauvegardée: {roc_path}")

# 3. Feature Importance
print("[INFO] Génération du graphique Feature Importance...")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': np.abs(clf.coef_[0])
}).sort_values('importance', ascending=True)

# Top 15 features
top_features = feature_importance.tail(15)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_features)))
bars = ax.barh(top_features['feature'], top_features['importance'], color=colors)
ax.set_xlabel('Importance (|Coefficient|)', fontsize=11)
ax.set_title('Top 15 Variables les Plus Importantes\n(Régression Logistique)', fontsize=14, fontweight='bold')

# Ajouter les valeurs
for bar, val in zip(bars, top_features['importance']):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
plt.savefig(fi_path, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Feature importance sauvegardée: {fi_path}")

# =============================================================================
# MODÈLE 2 : RÉGRESSION (Durée de Séjour)
# =============================================================================

print("\n" + "="*60)
print("MODÈLE 2 : RÉGRESSION - Estimation Durée de Séjour")
print("="*60)

X_reg = df_clean[feature_cols]
y_reg = df_clean['los_hours']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"[INFO] Entraînement: {len(X_train_reg)} samples")
print(f"[INFO] Test: {len(X_test_reg)} samples")

# Entraînement
reg = LinearRegression()
reg.fit(X_train_reg, y_train_reg)

# Prédictions
y_pred_reg = reg.predict(X_test_reg)

# Métriques
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"\n[RÉSULTATS] Régression:")
print(f"  - RMSE: {rmse:.2f} heures")
print(f"  - MAE: {mae:.2f} heures")
print(f"  - R²: {r2:.4f}")

# Sauvegarder le modèle
model_reg_path = os.path.join(OUTPUT_DIR, 'model_regression.joblib')
joblib.dump(reg, model_reg_path)
print(f"[OK] Modèle sauvegardé: {model_reg_path}")

# --- VISUALISATIONS RÉGRESSION ---

# 1. Prédictions vs Réalité
print("[INFO] Génération du graphique Prédictions vs Réalité...")
fig, ax = plt.subplots(figsize=(8, 8))

# Échantillonner pour lisibilité
sample_size = min(5000, len(y_test_reg))
indices = np.random.choice(len(y_test_reg), sample_size, replace=False)
y_test_sample = np.array(y_test_reg)[indices]
y_pred_sample = y_pred_reg[indices]

ax.scatter(y_test_sample, y_pred_sample, alpha=0.3, c='steelblue', s=10)
ax.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 
        'r--', lw=2, label='Prédiction Parfaite')
ax.set_xlabel('Durée Réelle (heures)', fontsize=11)
ax.set_ylabel('Durée Prédite (heures)', fontsize=11)
ax.set_title('Prédictions vs Valeurs Réelles\n(Durée de Séjour)', fontsize=14, fontweight='bold')
ax.legend()

# Ajouter métriques
textstr = f'R² = {r2:.3f}\nRMSE = {rmse:.1f}h\nMAE = {mae:.1f}h'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
pred_vs_real_path = os.path.join(OUTPUT_DIR, 'predictions_vs_reality.png')
plt.savefig(pred_vs_real_path, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Graphique Prédictions vs Réalité sauvegardé: {pred_vs_real_path}")

# 2. Distribution des Résidus
print("[INFO] Génération du graphique des résidus...")
residuals = y_test_reg - y_pred_reg

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogramme des résidus
ax1 = axes[0]
sns.histplot(residuals, kde=True, ax=ax1, color='purple', alpha=0.7)
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Résidu nul')
ax1.axvline(residuals.mean(), color='green', linestyle='-.', linewidth=2, 
            label=f'Moyenne: {residuals.mean():.2f}')
ax1.set_xlabel('Résidu (heures)', fontsize=11)
ax1.set_ylabel('Fréquence', fontsize=11)
ax1.set_title('Distribution des Résidus', fontsize=12, fontweight='bold')
ax1.legend()

# Résidus vs Prédictions
ax2 = axes[1]
ax2.scatter(y_pred_reg, residuals, alpha=0.3, c='steelblue', s=10)
ax2.axhline(0, color='red', linestyle='--', linewidth=2)
ax2.axhline(residuals.std()*2, color='orange', linestyle=':', linewidth=1, label='±2σ')
ax2.axhline(-residuals.std()*2, color='orange', linestyle=':', linewidth=1)
ax2.set_xlabel('Valeur Prédite (heures)', fontsize=11)
ax2.set_ylabel('Résidu (heures)', fontsize=11)
ax2.set_title('Résidus vs Prédictions', fontsize=12, fontweight='bold')
ax2.legend()

plt.tight_layout()
residuals_path = os.path.join(OUTPUT_DIR, 'residuals_analysis.png')
plt.savefig(residuals_path, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Analyse des résidus sauvegardée: {residuals_path}")

# 3. Coefficients de Régression
print("[INFO] Génération des coefficients de régression...")
reg_coef = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': reg.coef_
}).sort_values('coefficient', key=abs, ascending=True)

top_reg_coef = reg_coef.tail(15)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['green' if c > 0 else 'red' for c in top_reg_coef['coefficient']]
bars = ax.barh(top_reg_coef['feature'], top_reg_coef['coefficient'], color=colors, alpha=0.7)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient', fontsize=11)
ax.set_title('Top 15 Coefficients de Régression\n(Vert = ↑ durée, Rouge = ↓ durée)', 
             fontsize=14, fontweight='bold')

plt.tight_layout()
reg_coef_path = os.path.join(OUTPUT_DIR, 'regression_coefficients.png')
plt.savefig(reg_coef_path, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[OK] Coefficients de régression sauvegardés: {reg_coef_path}")

# =============================================================================
# GÉNÉRATION DES MODEL CARDS
# =============================================================================

print("\n[INFO] Génération des Model Cards...")

# --- MODEL CARD 1: CLASSIFICATION ---
modelcard_classification = f"""# Model Card : Prédiction de Détérioration

> **Document auto-généré** - Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y à %H:%M')}

- **Auteur**: Paul-Henri DOURNEAU & Dorian MARTY
- **Date de création**: 09/01/2026
- **Version**: 2.0

---

## 📋 Résumé du Modèle

| Propriété | Valeur |
|-----------|--------|
| **Type** | Classification Binaire |
| **Algorithme** | Régression Logistique |
| **Objectif** | Alerter le personnel soignant en cas de risque de détérioration |
| **Cible** | `deterioration_event` (0 = Stable, 1 = Détérioration) |

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

| Métrique | Valeur |
|----------|--------|
| **Dataset source** | Hospital Deterioration (Version Nettoyée) |
| **Taille totale** | {len(X_cls):,} observations |
| **Split train/test** | 80% / 20% |
| **Taille entraînement** | {len(X_train_cls):,} observations |
| **Taille test** | {len(X_test_cls):,} observations |

### Features Utilisées ({len(feature_cols)} variables)

Variables physiologiques et biologiques, **excluant** :
- `patient_id` (identifiant)
- Variables cibles (éviter fuite de données)

Lien vers le détail des features : [Data_Card.md](./Data_Card.md)

---

## ⚙️ Hyperparamètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `max_iter` | 1000 | Assurer la convergence |
| `solver` | lbfgs | Par défaut, performant |
| `C` | 1.0 | Régularisation standard |
| `class_weight` | balanced | Compenser le déséquilibre de classes |
| `random_state` | 42 | Reproductibilité |

---

## 📈 Performance et Analyse

### Métriques Globales

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | {cls_report_dict['accuracy']:.1%} |
| **AUC-ROC** | {roc_auc:.3f} |

### Rapport de Classification

```
{cls_report}
```

### Matrice de Confusion

![Matrice de Confusion](./confusion_matrix.png)

**Interprétation** :
- **TN (Vrais Négatifs)** : {tn:,} patients stables correctement identifiés
- **TP (Vrais Positifs)** : {tp:,} détériorations correctement détectées
- **FP (Faux Positifs)** : {fp:,} fausses alertes
- **FN (Faux Négatifs)** : {fn:,} détériorations manquées ⚠️

> 🔴 **Point Critique** : En milieu médical, les **Faux Négatifs** sont plus graves que les Faux Positifs.
> Un FN signifie qu'on rate une urgence potentielle.

### Courbe ROC

![Courbe ROC](./roc_curve.png)

Un AUC de **{roc_auc:.3f}** indique une capacité discriminante {"excellente" if roc_auc > 0.9 else "bonne" if roc_auc > 0.8 else "modérée"}.

---

## 🔍 Importance des Variables

![Feature Importance](./feature_importance.png)

### Top 5 Variables Prédictives

| Rang | Variable | Importance |
|------|----------|------------|
"""

# Ajouter le top 5
for i, (_, row) in enumerate(feature_importance.tail(5).iloc[::-1].iterrows(), 1):
    modelcard_classification += f"| {i} | `{row['feature']}` | {row['importance']:.3f} |\n"

modelcard_classification += f"""
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
"""

mc_cls_path = os.path.join(OUTPUT_DIR, 'Model_Card_Classification.md')
with open(mc_cls_path, 'w', encoding='utf-8') as f:
    f.write(modelcard_classification)
print(f"[OK] Model Card Classification sauvegardée: {mc_cls_path}")

# --- MODEL CARD 2: RÉGRESSION ---
modelcard_regression = f"""# Model Card : Estimation Durée de Séjour

> **Document auto-généré** - Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y à %H:%M')}

- **Auteur**: Paul-Henri DOURNEAU & Dorian MARTY
- **Date de création**: 09/01/2026
- **Version**: 2.0

---

## 📋 Résumé du Modèle

| Propriété | Valeur |
|-----------|--------|
| **Type** | Régression |
| **Algorithme** | Régression Linéaire Multiple |
| **Objectif** | Estimer la durée totale d'hospitalisation |
| **Cible** | `los_hours` (heures) |

### Cas d'Usage Hospitalier

Ce modèle permet d'**anticiper la gestion des lits** en prédisant la durée totale
d'hospitalisation d'un patient en fonction de son état clinique actuel.

**Applications** :
- Planification des sorties
- Optimisation de l'occupation des lits
- Estimation des ressources nécessaires

---

## 📊 Données d'Entraînement

| Métrique | Valeur |
|----------|--------|
| **Dataset source** | Hospital Deterioration (Version Nettoyée) |
| **Taille totale** | {len(X_reg):,} observations |
| **Split train/test** | 80% / 20% |
| **Durée moyenne (test)** | {y_test_reg.mean():.1f} heures |
| **Écart-type durée (test)** | {y_test_reg.std():.1f} heures |

Lien vers le détail des features : [Data_Card.md](./Data_Card.md)

---

## ⚙️ Paramètres du Modèle

| Paramètre | Valeur |
|-----------|--------|
| **Algorithme** | LinearRegression (sklearn) |
| **Régularisation** | Aucune |
| **Intercept** | {reg.intercept_:.4f} |

---

## 📈 Performance et Analyse

### Métriques de Régression

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **RMSE** | {rmse:.2f} heures | Erreur moyenne quadratique |
| **MAE** | {mae:.2f} heures | Erreur moyenne absolue |
| **R²** | {r2:.4f} | Variance expliquée ({r2*100:.1f}%) |

### Prédictions vs Valeurs Réelles

![Prédictions vs Réalité](./predictions_vs_reality.png)

**Interprétation** :
- Un R² de {r2:.2f} signifie que le modèle explique **{r2*100:.1f}%** de la variance
- {"C'est un score modeste, typique sur des données médicales complexes" if r2 < 0.5 else "C'est un résultat encourageant" if r2 < 0.7 else "C'est un excellent résultat"}
- L'erreur moyenne est de **{mae:.1f} heures** (environ {mae/24:.1f} jours)

### Analyse des Résidus

![Analyse des Résidus](./residuals_analysis.png)

**Observations** :
- Distribution des résidus {"centrée autour de zéro ✓" if abs(residuals.mean()) < 1 else "légèrement biaisée"}
- {"Pas d'hétéroscédasticité visible" if residuals.std() < 20 else "Variance augmente avec les valeurs prédites"}

---

## 🔍 Coefficients du Modèle

![Coefficients de Régression](./regression_coefficients.png)

### Interprétation des Coefficients

- **Coefficient positif** (vert) : Augmente la durée de séjour prédite
- **Coefficient négatif** (rouge) : Diminue la durée de séjour prédite

### Top 5 Variables Influentes

| Rang | Variable | Coefficient | Effet |
|------|----------|-------------|-------|
"""

# Ajouter le top 5
top5_reg = reg_coef.iloc[reg_coef['coefficient'].abs().nlargest(5).index]
for i, (_, row) in enumerate(top5_reg.iterrows(), 1):
    effet = "↑ Durée" if row['coefficient'] > 0 else "↓ Durée"
    modelcard_regression += f"| {i} | `{row['feature']}` | {row['coefficient']:.3f} | {effet} |\n"

modelcard_regression += f"""
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
"""

mc_reg_path = os.path.join(OUTPUT_DIR, 'Model_Card_Regression.md')
with open(mc_reg_path, 'w', encoding='utf-8') as f:
    f.write(modelcard_regression)
print(f"[OK] Model Card Régression sauvegardée: {mc_reg_path}")

# Fichier combiné pour compatibilité
combined_path = os.path.join(OUTPUT_DIR, 'Model_Cards.md')
with open(combined_path, 'w', encoding='utf-8') as f:
    f.write(f"# Model Cards - Vue Combinée\n\n")
    f.write(f"Ce fichier combine les deux Model Cards pour référence.\n")
    f.write(f"Pour les versions détaillées, voir:\n")
    f.write(f"- [Model_Card_Classification.md](./Model_Card_Classification.md)\n")
    f.write(f"- [Model_Card_Regression.md](./Model_Card_Regression.md)\n\n")
    f.write("---\n\n")
    f.write(modelcard_classification)
    f.write("\n\n---\n\n")
    f.write(modelcard_regression)

print("\n" + "="*60)
print("GÉNÉRATION DES MODEL CARDS TERMINÉE")
print("="*60)
print(f"Fichiers générés:")
print(f"  - {mc_cls_path}")
print(f"  - {mc_reg_path}")
print(f"  - {combined_path}")
print(f"  - {cm_path}")
print(f"  - {roc_path}")
print(f"  - {fi_path}")
print(f"  - {pred_vs_real_path}")
print(f"  - {residuals_path}")
print(f"  - {reg_coef_path}")
print(f"  - {model_cls_path}")
print(f"  - {model_reg_path}")
print(f"  - {log_path}")