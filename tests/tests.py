import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour éviter le blocage
import matplotlib.pyplot as plt
import os

# Chemin vers les données hospitalières
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "hospital_deterioration_hourly_panel.csv")

# Chargement des données
print(f"Chargement des données depuis: {data_path}")
df = pd.read_csv(data_path)

# Sélection des features numériques pour le test
# feature1 = heart_rate, feature2 = respiratory_rate, target = sepsis_risk_score
X = df[['heart_rate', 'respiratory_rate']].values
y = df['sepsis_risk_score']

# Échantillonnage pour accélérer l'exécution (10000 observations)
np.random.seed(42)
sample_size = min(10000, len(X))
indices = np.random.choice(len(X), sample_size, replace=False)
X = X[indices]
y = y.iloc[indices]

print(f"Nombre d'observations: {len(X)}")

# Normalisation des features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Arbre de décision par défaut
model = DecisionTreeRegressor(random_state=42)
model.fit(X_scaled, y)

# Test sur les données originales (sans bruit)
y_pred = model.predict(X_scaled)
rmse_clean = np.sqrt(mean_squared_error(y, y_pred))
print(f"RMSE sur données propres: {rmse_clean:.4f}")

def evaluate_noise_impact(model, X_base, y_true, feature_idx, noise_level):
    """
    Crée une copie de X, bruite une seule feature, et retourne la variation de RMSE en %.
    
    Arguments:
        model: Le modèle entraîné
        X_base: Les données originales (non normalisées)
        y_true: Les valeurs cibles
        feature_idx: L'index de la feature à bruiter (0 ou 1)
        noise_level: Le niveau de bruit en % de l'écart-type
    
    Retourne:
        La variation en % de la RMSE par rapport aux données propres
    """
    X_noisy = X_base.copy()
    
    # Calcul de l'écart-type de la feature
    std_dev = X_base[:, feature_idx].std()
    
    # Génération du bruit gaussien
    noise = np.random.normal(loc=0, scale=std_dev * noise_level / 100, size=X_noisy.shape[0])
    X_noisy[:, feature_idx] += noise
    
    # Normalisation des données bruitées
    X_noisy_scaled = scaler.transform(X_noisy)
    
    # Prédiction et calcul de la RMSE
    preds_noisy = model.predict(X_noisy_scaled)
    rmse_noisy = np.sqrt(mean_squared_error(y_true, preds_noisy))
    
    # Calcul de la variation en %
    variation = 100 * (rmse_noisy - rmse_clean) / rmse_clean
    return variation


# Niveaux d'intensité de bruit à tester
intensities = np.array([1, 3, 5, 10, 15, 20])

print("\nTest de robustesse en cours...")
print("=" * 50)

# Évaluation de l'impact du bruit sur chaque feature
mse_f1 = [evaluate_noise_impact(model, X, y, 0, n) for n in intensities]
mse_f2 = [evaluate_noise_impact(model, X, y, 1, n) for n in intensities]

# Affichage des résultats
print("\nRésultats du test de robustesse:")
print("-" * 50)
print(f"{'Niveau de bruit':<20} {'Heart Rate (%)':<20} {'Respiratory Rate (%)':<20}")
print("-" * 50)
for i, intensity in enumerate(intensities):
    print(f"{intensity}%{'':<18} {mse_f1[i]:>+.2f}%{'':<14} {mse_f2[i]:>+.2f}%")

print("\n" + "=" * 50)
print("Interprétation:")
print("-" * 50)

# Analyse de la sensibilité
avg_sensitivity_f1 = np.mean(mse_f1)
avg_sensitivity_f2 = np.mean(mse_f2)

if avg_sensitivity_f1 > avg_sensitivity_f2:
    print(f"- Le modèle est PLUS SENSIBLE aux perturbations sur 'Heart Rate'")
    print(f"  (variation moyenne: {avg_sensitivity_f1:.2f}% vs {avg_sensitivity_f2:.2f}%)")
else:
    print(f"- Le modèle est PLUS SENSIBLE aux perturbations sur 'Respiratory Rate'")
    print(f"  (variation moyenne: {avg_sensitivity_f2:.2f}% vs {avg_sensitivity_f1:.2f}%)")

print(f"\n- À 20% de bruit:")
print(f"  → Heart Rate cause une variation de {mse_f1[-1]:+.2f}% de RMSE")
print(f"  → Respiratory Rate cause une variation de {mse_f2[-1]:+.2f}% de RMSE")

# Génération du graphique
plt.figure(figsize=(10, 6))
plt.plot(intensities, mse_f1, 'o-', label="Bruit sur Heart Rate (%)", linewidth=2, markersize=8)
plt.plot(intensities, mse_f2, 's-', label="Bruit sur Respiratory Rate (%)", linewidth=2, markersize=8)

plt.title("Test de Robustesse : Impact du bruit sur les prédictions", fontsize=14, fontweight='bold')
plt.xlabel("Niveau de bruit (en % de l'écart-type)", fontsize=12)
plt.ylabel("Variation de la RMSE (%)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Sauvegarde du graphique
output_path = os.path.join(script_dir, "robustness_test_results.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nGraphique sauvegardé: {output_path}")

# plt.show()  # Désactivé pour éviter le blocage en mode non-interactif

print("\n✓ Test de robustesse terminé!")