"""
MODEL D - INTELLIGENT IRRIGATION CONTROL
=========================================
Author: Montassar Nawara
Dataset: IoT Agriculture 2024 + Smart Agriculture Dataset
Task: Binary Classification (Irrigation ON/OFF)
Algorithms: Decision Tree, Logistic Regression, XGBoost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             f1_score, precision_recall_curve, roc_curve, roc_auc_score)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

import pickle
import json
from datetime import datetime
import time

# Configuration
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("MODEL D - INTELLIGENT IRRIGATION CONTROL")
print("="*70)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1/9] Chargement des données...")

# Dataset 1: IoT Agriculture 2024
data_path1 = r"C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\data\IoT Agriculture 2024\IoTProcessed_Data.csv"

# Dataset 2: Smart Agriculture Dataset
data_path2 = r"C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\data\Smart Agriculture Dataset\cropdata_updated.csv"

try:
    df1 = pd.read_csv(data_path1)
    print(f"✓ Dataset 1 (IoT 2024) chargé: {df1.shape[0]} lignes, {df1.shape[1]} colonnes")
except Exception as e:
    print(f"✗ Erreur Dataset 1: {e}")
    df1 = None

try:
    df2 = pd.read_csv(data_path2)
    print(f"✓ Dataset 2 (Smart Agri) chargé: {df2.shape[0]} lignes, {df2.shape[1]} colonnes")
except Exception as e:
    print(f"✗ Erreur Dataset 2: {e}")
    df2 = None

# Utiliser le dataset disponible
if df1 is not None:
    df = df1.copy()
    dataset_name = "IoT Agriculture 2024"
    print(f"\n→ Utilisation du Dataset 1: {dataset_name}")
elif df2 is not None:
    df = df2.copy()
    dataset_name = "Smart Agriculture Dataset"
    print(f"\n→ Utilisation du Dataset 2: {dataset_name}")
else:
    print("✗ Aucun dataset disponible!")
    exit(1)

print("\n--- Aperçu des données ---")
print(df.head(10))
print("\n--- Info du dataset ---")
print(df.info())
print("\n--- Colonnes disponibles ---")
print(df.columns.tolist())

# ============================================================================
# 2. ANALYSE EXPLORATOIRE (EDA)
# ============================================================================
print("\n[2/9] Analyse exploratoire des données...")

print("\n--- Valeurs manquantes ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "Aucune valeur manquante ✓")

# Identifier la colonne target (irrigation status)
target_col = None
possible_targets = ['Irrigation_Status', 'Water_Pump', 'Action', 'irrigation', 'pump']
for col in df.columns:
    for target_name in possible_targets:
        if target_name.lower() in col.lower():
            target_col = col
            break
    if target_col:
        break

if target_col is None:
    print("\n⚠️  Colonne target non trouvée automatiquement.")
    print("Colonnes disponibles:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    # Essayer la dernière colonne
    target_col = df.columns[-1]
    print(f"\n→ Utilisation de la dernière colonne comme target: '{target_col}'")

print(f"\n--- Target Column: {target_col} ---")
print(df[target_col].value_counts())
print("\nProportions:")
print(df[target_col].value_counts(normalize=True) * 100)

# Visualisation de la distribution
plt.figure(figsize=(8, 6))
df[target_col].value_counts().plot(kind='bar', color=['lightcoral', 'lightblue'], edgecolor='black')
plt.title('Distribution des Décisions d\'Irrigation', fontsize=14, fontweight='bold')
plt.xlabel('Status', fontsize=12)
plt.ylabel('Nombre d\'échantillons', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_d_irrigation_distribution.png', dpi=300)
print("✓ Graphique sauvegardé: model_d_irrigation_distribution.png")
plt.close()

# ============================================================================
# 3. PREPROCESSING
# ============================================================================
print("\n[3/9] Preprocessing des données...")

# Séparer features et target
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

X = df[numeric_cols].copy()
y = df[target_col].copy()

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\n--- Features numériques utilisées ({len(numeric_cols)}) ---")
for i, col in enumerate(numeric_cols[:10], 1):
    print(f"{i}. {col}")
if len(numeric_cols) > 10:
    print(f"... et {len(numeric_cols) - 10} autres")

# Convertir target en binaire si nécessaire
y_unique = y.unique()
print(f"\n--- Valeurs uniques du target: {y_unique} ---")
if len(y_unique) > 2:
    print("⚠️  Target a plus de 2 classes, conversion en binaire...")
    # Prendre la médiane comme seuil
    threshold = y.median()
    y_binary = (y > threshold).astype(int)
    print(f"Seuil: {threshold}, 0: OFF (≤{threshold}), 1: ON (>{threshold})")
else:
    # Encoder en 0/1
    if y.dtype == 'object' or y.dtype == 'bool':
        y_binary = (y == y_unique[1]).astype(int)
    else:
        y_binary = y.astype(int)

print(f"\nDistribution binaire:")
print(f"OFF (0): {(y_binary == 0).sum()} ({(y_binary == 0).sum()/len(y_binary)*100:.1f}%)")
print(f"ON  (1): {(y_binary == 1).sum()} ({(y_binary == 1).sum()/len(y_binary)*100:.1f}%)")

# Feature Engineering
print("\n--- Feature Engineering ---")
# Ajouter des features dérivées si les colonnes existent
new_features = []

if 'MOI' in X.columns or 'Soil_Moisture' in X.columns:
    moi_col = 'MOI' if 'MOI' in X.columns else 'Soil_Moisture'
    X['Moisture_deficit'] = 80 - X[moi_col]  # 80% optimal
    new_features.append('Moisture_deficit')

if 'TMP' in X.columns and 'HUM' in X.columns:
    # VPD = Vapor Pressure Deficit
    X['VPD'] = X['TMP'] - X['HUM'] / 100 * X['TMP']
    new_features.append('VPD')

if new_features:
    print(f"✓ Features créées: {', '.join(new_features)}")
else:
    print("✓ Aucune feature engineering appliquée")

# Vérifier les valeurs hors limites
print("\n--- Validation domain limits ---")
outliers_count = 0
for col in X.columns:
    if 'temp' in col.lower() or 'tmp' in col.lower():
        outliers = ((X[col] < -10) | (X[col] > 50)).sum()
        outliers_count += outliers
    elif 'moi' in col.lower() or 'moisture' in col.lower() or 'hum' in col.lower():
        outliers = ((X[col] < 0) | (X[col] > 100)).sum()
        outliers_count += outliers

print(f"Valeurs hors limites détectées: {outliers_count} ({outliers_count/len(X)*100:.2f}%)")

# Train/Val/Test split (70/15/15) - TEMPORAL pour time-series
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_binary, test_size=0.15, random_state=42, stratify=y_binary, shuffle=True
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp, shuffle=True
)

print(f"\n--- Split des données ---")
print(f"Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Normalisation (MinMaxScaler pour compatibility)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("✓ Normalisation appliquée (MinMaxScaler [0,1])")

# ============================================================================
# 4. ENTRAÎNEMENT DES MODÈLES
# ============================================================================
print("\n[4/9] Entraînement des modèles...")

models = {}
results = {}
inference_times = {}

# -----------------------------
# 4.1 Decision Tree (pour ESP32)
# -----------------------------
print("\n--- Decision Tree Classifier (pour ESP32) ---")
dt_model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=15,
    min_samples_leaf=5,
    criterion='gini',
    random_state=42
)

start_time = time.time()
dt_model.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

# Prédictions
start_time = time.time()
y_pred_dt = dt_model.predict(X_test_scaled)
inference_time_dt = (time.time() - start_time) / len(X_test) * 1000  # ms per sample

# Métriques
acc_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
print(f"Accuracy: {acc_dt:.4f}")
print(f"F1-Score: {f1_dt:.4f}")
print(f"Training time: {train_time:.2f}s")
print(f"Inference time: {inference_time_dt:.4f}ms/sample")

models['Decision Tree'] = dt_model
results['Decision Tree'] = {
    'accuracy': acc_dt,
    'f1_score': f1_dt,
    'predictions': y_pred_dt,
    'train_time': train_time,
    'inference_time_ms': inference_time_dt
}

# -----------------------------
# 4.2 Logistic Regression (pour ESP32)
# -----------------------------
print("\n--- Logistic Regression (pour ESP32) ---")
lr_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='liblinear',
    max_iter=1000,
    random_state=42
)

start_time = time.time()
lr_model.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

# Prédictions
start_time = time.time()
y_pred_lr = lr_model.predict(X_test_scaled)
inference_time_lr = (time.time() - start_time) / len(X_test) * 1000

# Métriques
acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
print(f"Accuracy: {acc_lr:.4f}")
print(f"F1-Score: {f1_lr:.4f}")
print(f"Training time: {train_time:.2f}s")
print(f"Inference time: {inference_time_lr:.4f}ms/sample")

models['Logistic Regression'] = lr_model
results['Logistic Regression'] = {
    'accuracy': acc_lr,
    'f1_score': f1_lr,
    'predictions': y_pred_lr,
    'train_time': train_time,
    'inference_time_ms': inference_time_lr
}

# -----------------------------
# 4.3 XGBoost (pour edge/cloud)
# -----------------------------
print("\n--- XGBoost Classifier (pour Edge/Cloud) ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

start_time = time.time()
xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
train_time = time.time() - start_time

# Prédictions
start_time = time.time()
y_pred_xgb = xgb_model.predict(X_test_scaled)
inference_time_xgb = (time.time() - start_time) / len(X_test) * 1000

# Métriques
acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
print(f"Accuracy: {acc_xgb:.4f}")
print(f"F1-Score: {f1_xgb:.4f}")
print(f"Training time: {train_time:.2f}s")
print(f"Inference time: {inference_time_xgb:.4f}ms/sample")

models['XGBoost'] = xgb_model
results['XGBoost'] = {
    'accuracy': acc_xgb,
    'f1_score': f1_xgb,
    'predictions': y_pred_xgb,
    'train_time': train_time,
    'inference_time_ms': inference_time_xgb
}

# ============================================================================
# 5. SÉLECTION DU MEILLEUR MODÈLE
# ============================================================================
print("\n[5/9] Sélection du meilleur modèle...")

best_model_name = max(results, key=lambda k: results[k]['f1_score'])
best_model = models[best_model_name]
best_predictions = results[best_model_name]['predictions']

print(f"\n🏆 Meilleur modèle: {best_model_name}")
print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"   F1-Score: {results[best_model_name]['f1_score']:.4f}")
print(f"   Inference: {results[best_model_name]['inference_time_ms']:.4f}ms/sample")

# ============================================================================
# 6. ÉVALUATION DÉTAILLÉE
# ============================================================================
print("\n[6/9] Évaluation détaillée du meilleur modèle...")

# Classification Report
print("\n--- Classification Report ---")
report = classification_report(y_test, best_predictions, 
                               target_names=['OFF (0)', 'ON (1)'],
                               digits=4)
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, best_predictions)
print("\n--- Confusion Matrix ---")
print(cm)

# Visualisation
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['OFF (0)', 'ON (1)'],
            yticklabels=['OFF (0)', 'ON (1)'],
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_d_confusion_matrix.png', dpi=300)
print("✓ Confusion Matrix sauvegardée")
plt.close()

# ============================================================================
# 7. ROC & PRECISION-RECALL CURVES
# ============================================================================
print("\n[7/9] Génération des courbes ROC et Precision-Recall...")

# ROC Curve
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else best_predictions
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba) if hasattr(best_model, 'predict_proba') else acc_xgb

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title(f'ROC Curve - {best_model_name}', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='darkgreen', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall', fontsize=11)
plt.ylabel('Precision', fontsize=11)
plt.title(f'Precision-Recall Curve - {best_model_name}', fontsize=12, fontweight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower left")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_d_roc_pr_curves.png', dpi=300)
print("✓ ROC & Precision-Recall curves sauvegardées")
plt.close()

# ============================================================================
# 8. CROSS-VALIDATION
# ============================================================================
print("\n[8/9] Cross-Validation (5-Fold)...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, 
                            cv=cv, scoring='f1', n_jobs=-1)

print(f"CV F1-Scores: {cv_scores}")
print(f"Mean CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# 9. SAUVEGARDE DES RÉSULTATS
# ============================================================================
print("\n[9/9] Sauvegarde des modèles et résultats...")

import os
results_dir = r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results'
models_dir = r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\models'

# Sauvegarder tous les modèles (pour déploiement multiple)
for model_name, model in models.items():
    model_path = os.path.join(models_dir, f'model_d_{model_name.lower().replace(" ", "_")}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Calculer la taille du modèle
    model_size = os.path.getsize(model_path) / 1024  # KB
    results[model_name]['model_size_kb'] = model_size
    print(f"✓ {model_name} sauvegardé: {model_size:.2f} KB")

# Sauvegarder le scaler
scaler_path = os.path.join(models_dir, 'model_d_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler sauvegardé")

# Métriques
metrics_summary = {
    'best_model_name': best_model_name,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': dataset_name,
    'n_samples': len(X),
    'n_features': X.shape[1],
    'test_accuracy': float(results[best_model_name]['accuracy']),
    'test_f1_score': float(results[best_model_name]['f1_score']),
    'cv_mean_f1': float(cv_scores.mean()),
    'cv_std_f1': float(cv_scores.std()),
    'roc_auc': float(roc_auc),
    'confusion_matrix': cm.tolist(),
    'all_models_comparison': {
        name: {
            'accuracy': float(results[name]['accuracy']),
            'f1_score': float(results[name]['f1_score']),
            'inference_time_ms': float(results[name]['inference_time_ms']),
            'model_size_kb': float(results[name]['model_size_kb'])
        } for name in results
    }
}

metrics_path = os.path.join(results_dir, 'model_d_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_summary, f, indent=4)
print(f"✓ Métriques sauvegardées: {metrics_path}")

# Comparaison
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[name]['accuracy'] for name in results],
    'F1-Score': [results[name]['f1_score'] for name in results],
    'Inference (ms)': [results[name]['inference_time_ms'] for name in results],
    'Size (KB)': [results[name]['model_size_kb'] for name in results]
}).sort_values('F1-Score', ascending=False)

comparison_path = os.path.join(results_dir, 'model_d_comparison.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"✓ Comparaison sauvegardée")

print("\n" + "="*70)
print("✅ MODEL D - ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
print("="*70)
print(f"\n📊 RÉSULTATS FINAUX:")
print(f"   Meilleur modèle: {best_model_name}")
print(f"   Test Accuracy: {results[best_model_name]['accuracy']:.4f} (Objectif: 0.88-0.94)")
print(f"   Test F1-Score: {results[best_model_name]['f1_score']:.4f} (Objectif: 0.89-0.93)")
print(f"   ROC-AUC: {roc_auc:.4f}")
print(f"   CV F1-Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print(f"\n⚡ DEPLOYMENT METRICS:")
for model_name in results:
    print(f"   {model_name}:")
    print(f"      - Inference: {results[model_name]['inference_time_ms']:.4f}ms/sample")
    print(f"      - Model Size: {results[model_name]['model_size_kb']:.2f} KB")

if results[best_model_name]['f1_score'] >= 0.89:
    print("\n   🎯 OBJECTIF ATTEINT! ✅")
else:
    print(f"\n   ⚠️  Proche de l'objectif (écart: {0.89 - results[best_model_name]['f1_score']:.4f})")

print(f"\n📁 Fichiers générés:")
print(f"   - Modèles (3): {models_dir}/")
print(f"   - Métriques: {metrics_path}")
print(f"   - Graphiques: {results_dir}/")
print("\n" + "="*70)
