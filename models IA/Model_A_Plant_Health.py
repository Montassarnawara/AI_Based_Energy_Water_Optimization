"""
MODEL A - PLANT GROWTH AND HEALTH CLASSIFICATION
=================================================
Author: Montassar Nawara
Dataset: Advanced IoT Agriculture 2024 (30,000 records)
Task: Multiclass Classification (6 classes: SA, SB, SC, TA, TB, TC)
Algorithms: XGBoost, Random Forest, MLP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Pour sauvegarder les modèles
import pickle
import json
from datetime import datetime

# Configuration
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("MODEL A - PLANT GROWTH AND HEALTH CLASSIFICATION")
print("="*70)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1/8] Chargement des données...")

data_path = r"C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\data\Advanced IoT Agriculture 2024\Advanced_IoT_Dataset.csv"

try:
    df = pd.read_csv(data_path)
    print(f"✓ Dataset chargé avec succès: {df.shape[0]} lignes, {df.shape[1]} colonnes")
except Exception as e:
    print(f"✗ Erreur de chargement: {e}")
    exit(1)

# Afficher les premières lignes
print("\n--- Aperçu des données ---")
print(df.head())
print("\n--- Info du dataset ---")
print(df.info())
print("\n--- Statistiques descriptives ---")
print(df.describe())

# ============================================================================
# 2. ANALYSE EXPLORATOIRE (EDA)
# ============================================================================
print("\n[2/8] Analyse exploratoire des données...")

# Vérifier les valeurs manquantes
print("\n--- Valeurs manquantes ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "Aucune valeur manquante ✓")

# Distribution des classes
print("\n--- Distribution des classes ---")
print(df['Class'].value_counts())
print("\nProportions (%)")
print(df['Class'].value_counts(normalize=True) * 100)

# Visualisation de la distribution des classes
plt.figure(figsize=(10, 6))
df['Class'].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Distribution des Classes (Plant Growth Stages)', fontsize=14, fontweight='bold')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Nombre d\'échantillons', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_a_class_distribution.png', dpi=300)
print("✓ Graphique sauvegardé: model_a_class_distribution.png")
plt.close()

# ============================================================================
# 3. PREPROCESSING
# ============================================================================
print("\n[3/8] Preprocessing des données...")

# Séparer features et target
# Les noms de colonnes du CSV contiennent des espaces, on prend toutes les colonnes numériques
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
feature_columns = numeric_columns  # Toutes les colonnes numériques sauf Class

X = df[feature_columns].copy()
y = df['Class'].copy()

print(f"\n--- Colonnes utilisées comme features ---")
for i, col in enumerate(feature_columns, 1):
    print(f"{i}. {col}")

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Encoder les labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\n--- Mapping des classes ---")
for idx, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} → {idx}")

# Détecter et traiter les outliers (IQR method)
print("\n--- Détection des outliers (IQR) ---")
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
outlier_condition = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
n_outliers = outlier_condition.sum()
print(f"Nombre d'outliers détectés: {n_outliers} ({n_outliers/len(X)*100:.2f}%)")

# Optionnel: supprimer les outliers extrêmes
if n_outliers > 0 and n_outliers < len(X) * 0.1:  # Seulement si < 10%
    X_clean = X[~outlier_condition]
    y_clean = y_encoded[~outlier_condition]
    print(f"✓ Outliers supprimés. Nouveau shape: {X_clean.shape}")
else:
    X_clean = X
    y_clean = y_encoded
    print("✓ Outliers conservés (trop nombreux ou trop peu)")

# Train/Val/Test split (70/15/15)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_clean, y_clean, test_size=0.15, random_state=42, stratify=y_clean
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
)

print(f"\n--- Split des données ---")
print(f"Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_clean)*100:.1f}%)")
print(f"Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X_clean)*100:.1f}%)")
print(f"Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X_clean)*100:.1f}%)")

# Normalisation (StandardScaler)
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
# 4.1 XGBoost Classifier
# -----------------------------
print("\n--- XGBoost Classifier ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    verbosity=0
)

# Entraînement avec early stopping
xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=False
)

# Prédictions
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_proba_xgb = xgb_model.predict_proba(X_test_scaled)

# Métriques
acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')
print(f"Accuracy: {acc_xgb:.4f}")
print(f"F1-Score (weighted): {f1_xgb:.4f}")

models['XGBoost'] = xgb_model
results['XGBoost'] = {
    'accuracy': acc_xgb,
    'f1_score': f1_xgb,
    'predictions': y_pred_xgb,
    'probabilities': y_pred_proba_xgb
}

# -----------------------------
# 4.2 Random Forest Classifier
# -----------------------------
print("\n--- Random Forest Classifier ---")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

# Prédictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)

# Métriques
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
print(f"Accuracy: {acc_rf:.4f}")
print(f"F1-Score (weighted): {f1_rf:.4f}")

models['Random Forest'] = rf_model
results['Random Forest'] = {
    'accuracy': acc_rf,
    'f1_score': f1_rf,
    'predictions': y_pred_rf,
    'probabilities': y_pred_proba_rf
}

# -----------------------------
# 4.3 Decision Tree (baseline)
# -----------------------------
print("\n--- Decision Tree Classifier (Baseline) ---")
dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=10,
    random_state=42
)

dt_model.fit(X_train_scaled, y_train)

# Prédictions
y_pred_dt = dt_model.predict(X_test_scaled)

# Métriques
acc_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
print(f"Accuracy: {acc_dt:.4f}")
print(f"F1-Score (weighted): {f1_dt:.4f}")

models['Decision Tree'] = dt_model
results['Decision Tree'] = {
    'accuracy': acc_dt,
    'f1_score': f1_dt,
    'predictions': y_pred_dt
}

# ============================================================================
# 5. SÉLECTION DU MEILLEUR MODÈLE
# ============================================================================
print("\n[5/8] Sélection du meilleur modèle...")

best_model_name = max(results, key=lambda k: results[k]['f1_score'])
best_model = models[best_model_name]
best_predictions = results[best_model_name]['predictions']

print(f"\n🏆 Meilleur modèle: {best_model_name}")
print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"   F1-Score: {results[best_model_name]['f1_score']:.4f}")

# ============================================================================
# 6. ÉVALUATION DÉTAILLÉE
# ============================================================================
print("\n[6/8] Évaluation détaillée du meilleur modèle...")

# Classification Report
print("\n--- Classification Report ---")
report = classification_report(y_test, best_predictions, 
                               target_names=label_encoder.classes_,
                               digits=4)
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, best_predictions)
print("\n--- Confusion Matrix ---")
print(cm)

# Visualisation de la Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_a_confusion_matrix.png', dpi=300)
print("✓ Confusion Matrix sauvegardée")
plt.close()

# Feature Importance (pour XGBoost et Random Forest)
if best_model_name in ['XGBoost', 'Random Forest']:
    print("\n--- Feature Importance ---")
    if best_model_name == 'XGBoost':
        importance = best_model.feature_importances_
    else:
        importance = best_model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance_df)
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='teal', edgecolor='black')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_a_feature_importance.png', dpi=300)
    print("✓ Feature Importance sauvegardée")
    plt.close()

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(y_test, best_predictions)
per_class_metrics = pd.DataFrame({
    'Class': label_encoder.classes_,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})
print("\n--- Per-Class Metrics ---")
print(per_class_metrics)

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

# Créer le dossier results s'il n'existe pas
import os
results_dir = r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results'
models_dir = r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\models'

for directory in [results_dir, models_dir]:
    os.makedirs(directory, exist_ok=True)

# Sauvegarder le meilleur modèle
model_path = os.path.join(models_dir, f'model_a_{best_model_name.lower().replace(" ", "_")}.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"✓ Modèle sauvegardé: {model_path}")

# Sauvegarder le scaler
scaler_path = os.path.join(models_dir, 'model_a_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler sauvegardé: {scaler_path}")

# Sauvegarder le label encoder
encoder_path = os.path.join(models_dir, 'model_a_label_encoder.pkl')
with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"✓ Label Encoder sauvegardé: {encoder_path}")

# Sauvegarder les métriques dans un JSON
metrics_summary = {
    'model_name': best_model_name,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': 'Advanced IoT Agriculture 2024',
    'n_samples': len(X_clean),
    'n_features': len(feature_columns),
    'n_classes': len(label_encoder.classes_),
    'test_accuracy': float(results[best_model_name]['accuracy']),
    'test_f1_score': float(results[best_model_name]['f1_score']),
    'cv_mean_f1': float(cv_scores.mean()),
    'cv_std_f1': float(cv_scores.std()),
    'per_class_metrics': per_class_metrics.to_dict('records'),
    'confusion_matrix': cm.tolist(),
    'all_models_comparison': {
        name: {
            'accuracy': float(results[name]['accuracy']),
            'f1_score': float(results[name]['f1_score'])
        } for name in results
    }
}

if best_model_name in ['XGBoost', 'Random Forest']:
    metrics_summary['feature_importance'] = feature_importance_df.to_dict('records')

metrics_path = os.path.join(results_dir, 'model_a_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_summary, f, indent=4)
print(f"✓ Métriques sauvegardées: {metrics_path}")

# Sauvegarder le tableau de comparaison des modèles
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[name]['accuracy'] for name in results],
    'F1-Score': [results[name]['f1_score'] for name in results]
}).sort_values('F1-Score', ascending=False)

comparison_path = os.path.join(results_dir, 'model_a_comparison.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"✓ Comparaison sauvegardée: {comparison_path}")

print("\n" + "="*70)
print("✅ MODEL A - ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
print("="*70)
print(f"\n📊 RÉSULTATS FINAUX:")
print(f"   Meilleur modèle: {best_model_name}")
print(f"   Test Accuracy: {results[best_model_name]['accuracy']:.4f} (Objectif: 0.85-0.92)")
print(f"   Test F1-Score: {results[best_model_name]['f1_score']:.4f} (Objectif: 0.88-0.91)")
print(f"   CV F1-Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

if results[best_model_name]['f1_score'] >= 0.88:
    print("\n   🎯 OBJECTIF ATTEINT! ✅")
else:
    print(f"\n   ⚠️  Proche de l'objectif (écart: {0.88 - results[best_model_name]['f1_score']:.4f})")

print(f"\n📁 Fichiers générés:")
print(f"   - Modèle: {model_path}")
print(f"   - Métriques: {metrics_path}")
print(f"   - Graphiques: {results_dir}/")
print("\n" + "="*70)
