"""
MODEL B - CROP RECOMMENDATION SYSTEM
=====================================
Author: Montassar Nawara
Dataset: Crop Recommendation Dataset
Task: Multiclass Classification (21+ crop types)
Algorithms: Random Forest, LightGBM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             f1_score, precision_recall_fscore_support)
import warnings
warnings.filterwarnings('ignore')

import pickle
import json
from datetime import datetime

# Configuration
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("MODEL B - CROP RECOMMENDATION SYSTEM")
print("="*70)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1/8] Chargement des données...")

data_path = r"C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\data\Crop Recommendation Dataset\Crop_recommendation.csv"

try:
    df = pd.read_csv(data_path)
    print(f"✓ Dataset chargé avec succès: {df.shape[0]} lignes, {df.shape[1]} colonnes")
except Exception as e:
    print(f"✗ Erreur de chargement: {e}")
    exit(1)

print("\n--- Aperçu des données ---")
print(df.head(10))
print("\n--- Info du dataset ---")
print(df.info())
print("\n--- Statistiques descriptives ---")
print(df.describe())

# ============================================================================
# 2. ANALYSE EXPLORATOIRE (EDA)
# ============================================================================
print("\n[2/8] Analyse exploratoire des données...")

print("\n--- Valeurs manquantes ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "Aucune valeur manquante ✓")

print("\n--- Distribution des cultures (Top 10) ---")
print(df['label'].value_counts().head(10))
print(f"\nNombre total de types de cultures: {df['label'].nunique()}")

# Visualisation
plt.figure(figsize=(14, 8))
df['label'].value_counts().plot(kind='bar', color='forestgreen', edgecolor='black')
plt.title('Distribution des Types de Cultures', fontsize=14, fontweight='bold')
plt.xlabel('Type de Culture', fontsize=12)
plt.ylabel('Nombre d\'échantillons', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_b_crop_distribution.png', dpi=300)
print("✓ Graphique sauvegardé: model_b_crop_distribution.png")
plt.close()

# ============================================================================
# 3. PREPROCESSING
# ============================================================================
print("\n[3/8] Preprocessing des données...")

# Features et target
feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[feature_columns].copy()
y = df['label'].copy()

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Feature Engineering
print("\n--- Feature Engineering ---")
X['N_P_ratio'] = X['N'] / (X['P'] + 1)
X['N_K_ratio'] = X['N'] / (X['K'] + 1)
X['NPK_sum'] = X['N'] + X['P'] + X['K']
print("✓ Features créées: N_P_ratio, N_K_ratio, NPK_sum")

# Encoder les labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\n--- Mapping des cultures ({len(label_encoder.classes_)} classes) ---")
for idx, crop in enumerate(label_encoder.classes_[:10]):
    print(f"{crop} → {idx}")
if len(label_encoder.classes_) > 10:
    print(f"... et {len(label_encoder.classes_) - 10} autres")

# Domain validation des outliers
print("\n--- Validation des valeurs (domain limits) ---")
n_invalid = 0
n_invalid += (X['N'] < 0).sum() + (X['N'] > 140).sum()
n_invalid += (X['P'] < 5).sum() + (X['P'] > 145).sum()
n_invalid += (X['K'] < 5).sum() + (X['K'] > 205).sum()
n_invalid += (X['temperature'] < -10).sum() + (X['temperature'] > 50).sum()
n_invalid += (X['humidity'] < 0).sum() + (X['humidity'] > 100).sum()
n_invalid += (X['ph'] < 3).sum() + (X['ph'] > 10).sum()
n_invalid += (X['rainfall'] < 0).sum() + (X['rainfall'] > 3000).sum()
print(f"Valeurs hors limites: {n_invalid} ({n_invalid/len(X)*100:.2f}%)")

# Train/Val/Test split (70/15/15)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"\n--- Split des données ---")
print(f"Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("✓ Normalisation appliquée (StandardScaler)")

# ============================================================================
# 4. ENTRAÎNEMENT DES MODÈLES
# ============================================================================
print("\n[4/8] Entraînement des modèles...")

models = {}
results = {}

# -----------------------------
# 4.1 Random Forest Classifier
# -----------------------------
print("\n--- Random Forest Classifier ---")
rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    criterion='gini',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train_scaled, y_train)

# Prédictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)

# Métriques
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
f1_macro_rf = f1_score(y_test, y_pred_rf, average='macro')
print(f"Accuracy: {acc_rf:.4f}")
print(f"F1-Score (weighted): {f1_rf:.4f}")
print(f"F1-Score (macro): {f1_macro_rf:.4f}")

models['Random Forest'] = rf_model
results['Random Forest'] = {
    'accuracy': acc_rf,
    'f1_weighted': f1_rf,
    'f1_macro': f1_macro_rf,
    'predictions': y_pred_rf,
    'probabilities': y_pred_proba_rf
}

# -----------------------------
# 4.2 LightGBM Classifier (si installé)
# -----------------------------
try:
    import lightgbm as lgb
    
    print("\n--- LightGBM Classifier ---")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=12,
        learning_rate=0.08,
        num_leaves=50,
        random_state=42,
        verbose=-1
    )
    
    lgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # Prédictions
    y_pred_lgb = lgb_model.predict(X_test_scaled)
    y_pred_proba_lgb = lgb_model.predict_proba(X_test_scaled)
    
    # Métriques
    acc_lgb = accuracy_score(y_test, y_pred_lgb)
    f1_lgb = f1_score(y_test, y_pred_lgb, average='weighted')
    f1_macro_lgb = f1_score(y_test, y_pred_lgb, average='macro')
    print(f"Accuracy: {acc_lgb:.4f}")
    print(f"F1-Score (weighted): {f1_lgb:.4f}")
    print(f"F1-Score (macro): {f1_macro_lgb:.4f}")
    
    models['LightGBM'] = lgb_model
    results['LightGBM'] = {
        'accuracy': acc_lgb,
        'f1_weighted': f1_lgb,
        'f1_macro': f1_macro_lgb,
        'predictions': y_pred_lgb,
        'probabilities': y_pred_proba_lgb
    }
except ImportError:
    print("\n⚠️  LightGBM non installé. Seulement Random Forest sera utilisé.")
    print("   Pour installer: pip install lightgbm")

# ============================================================================
# 5. SÉLECTION DU MEILLEUR MODÈLE
# ============================================================================
print("\n[5/8] Sélection du meilleur modèle...")

best_model_name = max(results, key=lambda k: results[k]['f1_weighted'])
best_model = models[best_model_name]
best_predictions = results[best_model_name]['predictions']

print(f"\n🏆 Meilleur modèle: {best_model_name}")
print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"   F1-Score (weighted): {results[best_model_name]['f1_weighted']:.4f}")
print(f"   F1-Score (macro): {results[best_model_name]['f1_macro']:.4f}")

# ============================================================================
# 6. ÉVALUATION DÉTAILLÉE
# ============================================================================
print("\n[6/8] Évaluation détaillée du meilleur modèle...")

# Classification Report
print("\n--- Classification Report (Top 10 cultures) ---")
report = classification_report(y_test, best_predictions, 
                               target_names=label_encoder.classes_,
                               digits=4)
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, best_predictions)
print("\n--- Confusion Matrix (shape) ---")
print(f"Shape: {cm.shape}")

# Visualisation de la Confusion Matrix
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_b_confusion_matrix.png', dpi=300)
print("✓ Confusion Matrix sauvegardée")
plt.close()

# Feature Importance
print("\n--- Feature Importance ---")
if hasattr(best_model, 'feature_importances_'):
    importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance_df)
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], 
             color='darkgreen', edgecolor='black')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_b_feature_importance.png', dpi=300)
    print("✓ Feature Importance sauvegardée")
    plt.close()

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(y_test, best_predictions)
per_class_metrics = pd.DataFrame({
    'Crop': label_encoder.classes_,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
}).sort_values('F1-Score', ascending=False)
print("\n--- Per-Crop Metrics (Top 10) ---")
print(per_class_metrics.head(10))

# Visualisation per-crop performance
plt.figure(figsize=(14, 8))
per_class_metrics_sorted = per_class_metrics.sort_values('F1-Score', ascending=True)
plt.barh(per_class_metrics_sorted['Crop'], per_class_metrics_sorted['F1-Score'], 
         color='seagreen', edgecolor='black')
plt.xlabel('F1-Score', fontsize=12)
plt.ylabel('Crop Type', fontsize=12)
plt.title(f'Per-Crop F1-Score Performance - {best_model_name}', fontsize=14, fontweight='bold')
plt.xlim([0, 1.05])
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_b_per_crop_performance.png', dpi=300)
print("✓ Per-Crop Performance sauvegardée")
plt.close()

# ============================================================================
# 7. CROSS-VALIDATION
# ============================================================================
print("\n[7/8] Cross-Validation (5-Fold)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, 
                            cv=cv, scoring='f1_weighted', n_jobs=-1)

print(f"CV F1-Scores: {cv_scores}")
print(f"Mean CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# 8. SAUVEGARDE DES RÉSULTATS
# ============================================================================
print("\n[8/8] Sauvegarde des modèles et résultats...")

import os
results_dir = r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results'
models_dir = r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\models'

# Sauvegarder le meilleur modèle
model_path = os.path.join(models_dir, f'model_b_{best_model_name.lower().replace(" ", "_")}.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"✓ Modèle sauvegardé: {model_path}")

# Sauvegarder le scaler
scaler_path = os.path.join(models_dir, 'model_b_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler sauvegardé: {scaler_path}")

# Sauvegarder le label encoder
encoder_path = os.path.join(models_dir, 'model_b_label_encoder.pkl')
with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"✓ Label Encoder sauvegardé: {encoder_path}")

# Sauvegarder les métriques
metrics_summary = {
    'model_name': best_model_name,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': 'Crop Recommendation Dataset',
    'n_samples': len(X),
    'n_features': len(X.columns),
    'n_crops': len(label_encoder.classes_),
    'test_accuracy': float(results[best_model_name]['accuracy']),
    'test_f1_weighted': float(results[best_model_name]['f1_weighted']),
    'test_f1_macro': float(results[best_model_name]['f1_macro']),
    'cv_mean_f1': float(cv_scores.mean()),
    'cv_std_f1': float(cv_scores.std()),
    'per_crop_metrics': per_class_metrics.to_dict('records'),
    'confusion_matrix': cm.tolist(),
    'all_models_comparison': {
        name: {
            'accuracy': float(results[name]['accuracy']),
            'f1_weighted': float(results[name]['f1_weighted']),
            'f1_macro': float(results[name]['f1_macro'])
        } for name in results
    }
}

if hasattr(best_model, 'feature_importances_'):
    metrics_summary['feature_importance'] = feature_importance_df.to_dict('records')

metrics_path = os.path.join(results_dir, 'model_b_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_summary, f, indent=4)
print(f"✓ Métriques sauvegardées: {metrics_path}")

# Comparaison des modèles
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[name]['accuracy'] for name in results],
    'F1-Weighted': [results[name]['f1_weighted'] for name in results],
    'F1-Macro': [results[name]['f1_macro'] for name in results]
}).sort_values('F1-Weighted', ascending=False)

comparison_path = os.path.join(results_dir, 'model_b_comparison.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"✓ Comparaison sauvegardée: {comparison_path}")

# Sauvegarder per-crop metrics
per_crop_path = os.path.join(results_dir, 'model_b_per_crop_metrics.csv')
per_class_metrics.to_csv(per_crop_path, index=False)
print(f"✓ Per-Crop Metrics sauvegardées: {per_crop_path}")

print("\n" + "="*70)
print("✅ MODEL B - ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
print("="*70)
print(f"\n📊 RÉSULTATS FINAUX:")
print(f"   Meilleur modèle: {best_model_name}")
print(f"   Test Accuracy: {results[best_model_name]['accuracy']:.4f} (Objectif: 0.90-0.95)")
print(f"   Test F1-Score (weighted): {results[best_model_name]['f1_weighted']:.4f} (Objectif: 0.89-0.93)")
print(f"   Test F1-Score (macro): {results[best_model_name]['f1_macro']:.4f}")
print(f"   CV F1-Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

if results[best_model_name]['accuracy'] >= 0.90:
    print("\n   🎯 OBJECTIF ATTEINT! ✅")
else:
    print(f"\n   ⚠️  Proche de l'objectif (écart: {0.90 - results[best_model_name]['accuracy']:.4f})")

print(f"\n📁 Fichiers générés:")
print(f"   - Modèle: {model_path}")
print(f"   - Métriques: {metrics_path}")
print(f"   - Graphiques: {results_dir}/")
print("\n" + "="*70)
