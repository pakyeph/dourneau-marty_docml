"""
Script Maître - Génération Complète de Documentation
=====================================================
Ce script exécute tous les générateurs de documentation dans l'ordre
pour produire l'ensemble des Data Cards, Model Cards et documentation technique.

Auteur: Paul-Henri DOURNEAU & Dorian MARTY
Date: 30/01/2026

Usage:
    python generate_all_docs.py
"""

import subprocess
import sys
import os
from datetime import datetime

# Configuration
SCRIPTS_DIR = r'c:\Users\Ph\Documents\.EPSI\Documentations'
PYTHON_EXECUTABLE = sys.executable

# Liste des scripts à exécuter dans l'ordre
SCRIPTS = [
    ('datacard.py', 'Génération de la Data Card et visualisations de données'),
    ('modelcard.py', 'Entraînement des modèles et génération des Model Cards'),
    ('technicalcard.py', 'Génération de la Documentation Technique')
]

def run_script(script_name, description):
    """Exécute un script Python et capture le résultat."""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    
    print(f"\n{'='*60}")
    print(f"📋 {description}")
    print(f"📄 Script: {script_name}")
    print('='*60)
    
    if not os.path.exists(script_path):
        print(f"[ERREUR] Script non trouvé: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [PYTHON_EXECUTABLE, script_path],
            cwd=SCRIPTS_DIR,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes max
        )
        
        if result.stdout:
            print(result.stdout)
        
        if result.returncode != 0:
            print(f"[ERREUR] Le script a échoué avec le code {result.returncode}")
            if result.stderr:
                print(f"Stderr: {result.stderr}")
            return False
            
        return True
        
    except subprocess.TimeoutExpired:
        print(f"[ERREUR] Timeout dépassé pour {script_name}")
        return False
    except Exception as e:
        print(f"[ERREUR] Exception: {e}")
        return False

def main():
    """Point d'entrée principal."""
    start_time = datetime.now()
    
    print("\n" + "="*60)
    print("🚀 GÉNÉRATION COMPLÈTE DE LA DOCUMENTATION")
    print(f"   Date: {start_time.strftime('%d/%m/%Y à %H:%M')}")
    print("="*60)
    
    results = []
    
    for script_name, description in SCRIPTS:
        success = run_script(script_name, description)
        results.append((script_name, success))
    
    # Résumé
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DE L'EXÉCUTION")
    print("="*60)
    
    all_success = True
    for script_name, success in results:
        status = "✅ SUCCÈS" if success else "❌ ÉCHEC"
        print(f"  {status} - {script_name}")
        if not success:
            all_success = False
    
    print(f"\n⏱️  Durée totale: {duration:.1f} secondes")
    
    if all_success:
        print("\n" + "="*60)
        print("✅ DOCUMENTATION GÉNÉRÉE AVEC SUCCÈS!")
        print("="*60)
        print("\nFichiers générés:")
        print(f"  📁 {SCRIPTS_DIR}")
        print("  ├── 📑 Data_Card.md")
        print("  ├── 📑 Model_Card_Classification.md")
        print("  ├── 📑 Model_Card_Regression.md")
        print("  ├── 📑 Documentation_Technique.md")
        print("  ├── 📑 transformation_log.md")
        print("  ├── 📊 heatmap_correlation.png")
        print("  ├── 📊 heatmap_correlation_annotated.png")
        print("  ├── 📊 distributions_signes_vitaux.png")
        print("  ├── 📊 distributions_analyses_labo.png")
        print("  ├── 📊 boxplots_outliers.png")
        print("  ├── 📊 valeurs_manquantes.png")
        print("  ├── 📊 distribution_cibles.png")
        print("  ├── 📊 confusion_matrix.png")
        print("  ├── 📊 roc_curve.png")
        print("  ├── 📊 feature_importance.png")
        print("  ├── 📊 predictions_vs_reality.png")
        print("  ├── 📊 residuals_analysis.png")
        print("  ├── 📊 regression_coefficients.png")
        print("  ├── 🧠 model_classification.joblib")
        print("  └── 🧠 model_regression.joblib")
    else:
        print("\n⚠️  CERTAINS SCRIPTS ONT ÉCHOUÉ!")
        print("    Vérifiez les erreurs ci-dessus.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
