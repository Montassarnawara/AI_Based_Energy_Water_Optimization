"""
MODEL C - CROP YIELD PREDICTION
================================
Author: Montassar Nawara
Dataset: Agriculture Crop Yield + Smart Farming Sensor Data
Task: Regression (predict tonnes/hectare)
Algorithms: CatBoost, LightGBM, Random Forest
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             mean_absolute_percentage_error)
import lightgbm as lgb
import catboost as cb
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
print("MODEL C - CROP YIELD PREDICTION")
print("="*70)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1/9] Chargement des données...")

# Dataset 1: Agriculture Crop Yield
data_path1 = r"C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\data\Agriculture Crop Yield\crop_yield.csv"

# Dataset 2: Smart Farming Sensor Data for Yield Prediction
data_path2 = r"C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\data\Smart Farming Sensor Data for Yield Prediction\Smart_Farming_Crop_Yield_2024.csv"

try:
    df1 = pd.read_csv(data_path1)
    print(f"✓ Dataset 1 (Crop Yield) chargé: {df1.shape[0]} lignes, {df1.shape[1]} colonnes")
except Exception as e:
    print(f"✗ Erreur Dataset 1: {e}")
    df1 = None

try:
    df2 = pd.read_csv(data_path2)
    print(f"✓ Dataset 2 (Smart Farming) chargé: {df2.shape[0]} lignes, {df2.shape[1]} colonnes")
except Exception as e:
    print(f"✗ Erreur Dataset 2: {e}")
    df2 = None

# Utiliser le meilleur dataset disponible (prioriser le plus grand)
if df1 is not None:
    df = df1.copy()
    dataset_name = "Agriculture Crop Yield"
    print(f"\n→ Utilisation du Dataset 1: {dataset_name} (1M échantillons)")
elif df2 is not None:
    df = df2.copy()
    dataset_name = "Smart Farming Sensor Data 2024"
    print(f"\n→ Utilisation du Dataset 2: {dataset_name}")
else:
    print("✗ Aucun dataset disponible!")
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
print("\n[2/9] Analyse exploratoire des données...")

print("\n--- Valeurs manquantes ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "Aucune valeur manquante ✓")

# Identifier la colonne target (yield)
target_col = None
possible_targets = ['Yield', 'yield', 'Crop_Yield', 'Production', 'production', 'tonnes_per_hectare']
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
    # Essayer la dernière colonne numérique
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_col = numeric_cols[-1]
    print(f"\n→ Utilisation de la dernière colonne numérique comme target: '{target_col}'")

print(f"\n--- Target Column: {target_col} ---")
print(df[target_col].describe())

# Visualisation de la distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df[target_col], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Yield (tonnes/ha)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Distribution du Rendement', fontsize=13, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df[target_col], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightgreen', color='darkgreen'),
            medianprops=dict(color='red', linewidth=2))
plt.ylabel('Yield (tonnes/ha)', fontsize=11)
plt.title('Boxplot du Rendement', fontsize=13, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_c_yield_distribution.png', dpi=300)
print("✓ Graphique de distribution sauvegardé")
plt.close()

# ============================================================================
# 3. PREPROCESSING
# ============================================================================
print("\n[3/9] Preprocessing des données...")

# Séparer features et target
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

if target_col in numeric_cols:
    numeric_cols.remove(target_col)
if target_col in categorical_cols:
    categorical_cols.remove(target_col)

print(f"\n--- Features numériques ({len(numeric_cols)}) ---")
for i, col in enumerate(numeric_cols[:10], 1):
    print(f"{i}. {col}")
if len(numeric_cols) > 10:
    print(f"... et {len(numeric_cols) - 10} autres")

print(f"\n--- Features catégorielles ({len(categorical_cols)}) ---")
for i, col in enumerate(categorical_cols[:5], 1):
    print(f"{i}. {col}: {df[col].nunique()} valeurs uniques")
if len(categorical_cols) > 5:
    print(f"... et {len(categorical_cols) - 5} autres")

# Encoder les features catégorielles
label_encoders = {}
df_processed = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le
    print(f"✓ Encodé: {col}")

# Features finales
feature_cols = numeric_cols + categorical_cols
X = df_processed[feature_cols].copy()
y = df_processed[target_col].copy()

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
print(f"Target mean: {y.mean():.2f} ± {y.std():.2f}")

# Feature Engineering
print("\n--- Feature Engineering ---")
new_features = []

# Créer des features d'interaction si possible
if 'N' in X.columns and 'P' in X.columns and 'K' in X.columns:
    X['NPK_sum'] = X['N'] + X['P'] + X['K']
    X['NPK_product'] = X['N'] * X['P'] * X['K']
    X['N_P_ratio'] = X['N'] / (X['P'] + 1)
    X['N_K_ratio'] = X['N'] / (X['K'] + 1)
    new_features.extend(['NPK_sum', 'NPK_product', 'N_P_ratio', 'N_K_ratio'])

if 'temperature' in X.columns and 'humidity' in X.columns:
    X['temp_humidity_interaction'] = X['temperature'] * X['humidity'] / 100
    new_features.append('temp_humidity_interaction')

if 'rainfall' in X.columns and 'temperature' in X.columns:
    X['rain_temp_ratio'] = X['rainfall'] / (X['temperature'] + 1)
    new_features.append('rain_temp_ratio')

if new_features:
    print(f"✓ Features créées: {', '.join(new_features)}")
    feature_cols.extend(new_features)
else:
    print("✓ Aucune feature engineering appliquée")

# Détection des outliers (IQR method)
print("\n--- Détection des outliers ---")
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = ((y < lower_bound) | (y > upper_bound)).sum()
print(f"Outliers détectés: {outliers} ({outliers/len(y)*100:.2f}%)")
print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

# Option: Retirer les outliers extrêmes
if outliers / len(y) < 0.05:  # Moins de 5%
    mask = (y >= lower_bound) & (y <= upper_bound)
    X = X[mask]
    y = y[mask]
    print(f"✓ Outliers supprimés, nouveau shape: {X.shape}")

# Train/Val/Test split (70/15/15)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=True
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, shuffle=True
)

print(f"\n--- Split des données ---")
print(f"Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Normalisation (StandardScaler pour régression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("✓ Normalisation appliquée (StandardScaler)")

# ============================================================================
# 4. ENTRAÎNEMENT DES MODÈLES
# ============================================================================
print("\n[4/9] Entraînement des modèles...")

models = {}
results = {}

# -----------------------------
# 4.1 Random Forest Regressor
# -----------------------------
print("\n--- Random Forest Regressor ---")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
rf_model.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

# Prédictions
y_pred_rf = rf_model.predict(X_test_scaled)

# Métriques
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf) * 100

print(f"RMSE: {rmse_rf:.4f}")
print(f"MAE: {mae_rf:.4f}")
print(f"R²: {r2_rf:.4f}")
print(f"MAPE: {mape_rf:.2f}%")
print(f"Training time: {train_time:.2f}s")

models['Random Forest'] = rf_model
results['Random Forest'] = {
    'rmse': rmse_rf,
    'mae': mae_rf,
    'r2': r2_rf,
    'mape': mape_rf,
    'predictions': y_pred_rf,
    'train_time': train_time
}

# -----------------------------
# 4.2 LightGBM Regressor
# -----------------------------
print("\n--- LightGBM Regressor ---")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=8,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

start_time = time.time()
lgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)
train_time = time.time() - start_time

# Prédictions
y_pred_lgb = lgb_model.predict(X_test_scaled)

# Métriques
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mse_lgb)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)
mape_lgb = mean_absolute_percentage_error(y_test, y_pred_lgb) * 100

print(f"RMSE: {rmse_lgb:.4f}")
print(f"MAE: {mae_lgb:.4f}")
print(f"R²: {r2_lgb:.4f}")
print(f"MAPE: {mape_lgb:.2f}%")
print(f"Training time: {train_time:.2f}s")

models['LightGBM'] = lgb_model
results['LightGBM'] = {
    'rmse': rmse_lgb,
    'mae': mae_lgb,
    'r2': r2_lgb,
    'mape': mape_lgb,
    'predictions': y_pred_lgb,
    'train_time': train_time
}

# -----------------------------
# 4.3 CatBoost Regressor
# -----------------------------
print("\n--- CatBoost Regressor ---")
cat_features = [i for i, col in enumerate(feature_cols) if col in categorical_cols]
print(f"Categorical features indices: {cat_features}")

cb_model = cb.CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0,
    early_stopping_rounds=50
)

start_time = time.time()
cb_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_features,
    verbose=False
)
train_time = time.time() - start_time

# Prédictions
y_pred_cb = cb_model.predict(X_test)

# Métriques
mse_cb = mean_squared_error(y_test, y_pred_cb)
rmse_cb = np.sqrt(mse_cb)
mae_cb = mean_absolute_error(y_test, y_pred_cb)
r2_cb = r2_score(y_test, y_pred_cb)
mape_cb = mean_absolute_percentage_error(y_test, y_pred_cb) * 100

print(f"RMSE: {rmse_cb:.4f}")
print(f"MAE: {mae_cb:.4f}")
print(f"R²: {r2_cb:.4f}")
print(f"MAPE: {mape_cb:.2f}%")
print(f"Training time: {train_time:.2f}s")

models['CatBoost'] = cb_model
results['CatBoost'] = {
    'rmse': rmse_cb,
    'mae': mae_cb,
    'r2': r2_cb,
    'mape': mape_cb,
    'predictions': y_pred_cb,
    'train_time': train_time
}

# ============================================================================
# 5. SÉLECTION DU MEILLEUR MODÈLE
# ============================================================================
print("\n[5/9] Sélection du meilleur modèle...")

# Meilleur modèle = R² le plus élevé
best_model_name = max(results, key=lambda k: results[k]['r2'])
best_model = models[best_model_name]
best_predictions = results[best_model_name]['predictions']

print(f"\n🏆 Meilleur modèle: {best_model_name}")
print(f"   RMSE: {results[best_model_name]['rmse']:.4f}")
print(f"   MAE: {results[best_model_name]['mae']:.4f}")
print(f"   R²: {results[best_model_name]['r2']:.4f}")
print(f"   MAPE: {results[best_model_name]['mape']:.2f}%")

# ============================================================================
# 6. VISUALISATIONS
# ============================================================================
print("\n[6/9] Génération des visualisations...")

# Actual vs Predicted
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, best_predictions, alpha=0.5, color='blue', edgecolor='black', s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect prediction')
plt.xlabel('Actual Yield (tonnes/ha)', fontsize=11)
plt.ylabel('Predicted Yield (tonnes/ha)', fontsize=11)
plt.title(f'Actual vs Predicted - {best_model_name}', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# Residuals plot
residuals = y_test - best_predictions
plt.subplot(1, 2, 2)
plt.scatter(best_predictions, residuals, alpha=0.5, color='green', edgecolor='black', s=30)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Yield (tonnes/ha)', fontsize=11)
plt.ylabel('Residuals', fontsize=11)
plt.title(f'Residuals Plot - {best_model_name}', fontsize=13, fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_c_predictions.png', dpi=300)
print("✓ Graphique Actual vs Predicted sauvegardé")
plt.close()

# Feature Importance
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='teal', edgecolor='black')
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top 15 Feature Importances - {best_model_name}', 
              fontsize=13, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_c_feature_importance.png', dpi=300)
    print("✓ Graphique Feature Importance sauvegardé")
    plt.close()

# ============================================================================
# 7. COMPARAISON DES MODÈLES
# ============================================================================
print("\n[7/9] Comparaison des modèles...")

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[name]['rmse'] for name in results],
    'MAE': [results[name]['mae'] for name in results],
    'R²': [results[name]['r2'] for name in results],
    'MAPE (%)': [results[name]['mape'] for name in results]
}).sort_values('R²', ascending=False)

print("\n", comparison_df.to_string(index=False))

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# RMSE
axes[0, 0].bar(comparison_df['Model'], comparison_df['RMSE'], 
               color=['gold', 'silver', 'coral'], edgecolor='black')
axes[0, 0].set_title('RMSE Comparison', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('RMSE', fontsize=11)
axes[0, 0].grid(axis='y', alpha=0.3)

# MAE
axes[0, 1].bar(comparison_df['Model'], comparison_df['MAE'], 
               color=['gold', 'silver', 'coral'], edgecolor='black')
axes[0, 1].set_title('MAE Comparison', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('MAE', fontsize=11)
axes[0, 1].grid(axis='y', alpha=0.3)

# R²
axes[1, 0].bar(comparison_df['Model'], comparison_df['R²'], 
               color=['gold', 'silver', 'coral'], edgecolor='black')
axes[1, 0].set_title('R² Score Comparison', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('R²', fontsize=11)
axes[1, 0].grid(axis='y', alpha=0.3)

# MAPE
axes[1, 1].bar(comparison_df['Model'], comparison_df['MAPE (%)'], 
               color=['gold', 'silver', 'coral'], edgecolor='black')
axes[1, 1].set_title('MAPE Comparison', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('MAPE (%)', fontsize=11)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_c_comparison.png', dpi=300)
print("✓ Graphique de comparaison sauvegardé")
plt.close()

# ============================================================================
# 8. CROSS-VALIDATION
# ============================================================================
print("\n[8/9] Cross-Validation (5-Fold)...")

cv = KFold(n_splits=5, shuffle=True, random_state=42)

if best_model_name == 'CatBoost':
    # CatBoost needs original data
    cv_scores = cross_val_score(best_model, X_train, y_train, 
                                cv=cv, scoring='r2', n_jobs=-1)
else:
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, 
                                cv=cv, scoring='r2', n_jobs=-1)

print(f"CV R² Scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# 9. SAUVEGARDE DES RÉSULTATS
# ============================================================================
print("\n[9/9] Sauvegarde des modèles et résultats...")

import os
results_dir = r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results'
models_dir = r'C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\models'

# Sauvegarder le meilleur modèle
model_path = os.path.join(models_dir, f'model_c_{best_model_name.lower().replace(" ", "_")}.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
model_size = os.path.getsize(model_path) / 1024  # KB
print(f"✓ Meilleur modèle sauvegardé: {model_size:.2f} KB")

# Sauvegarder le scaler
scaler_path = os.path.join(models_dir, 'model_c_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler sauvegardé")

# Sauvegarder les label encoders
if label_encoders:
    encoders_path = os.path.join(models_dir, 'model_c_label_encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"✓ Label Encoders sauvegardés")

# Métriques
metrics_summary = {
    'best_model_name': best_model_name,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': dataset_name,
    'n_samples': len(X),
    'n_features': X.shape[1],
    'feature_names': feature_cols,
    'test_rmse': float(results[best_model_name]['rmse']),
    'test_mae': float(results[best_model_name]['mae']),
    'test_r2': float(results[best_model_name]['r2']),
    'test_mape': float(results[best_model_name]['mape']),
    'cv_mean_r2': float(cv_scores.mean()),
    'cv_std_r2': float(cv_scores.std()),
    'all_models_comparison': {
        name: {
            'rmse': float(results[name]['rmse']),
            'mae': float(results[name]['mae']),
            'r2': float(results[name]['r2']),
            'mape': float(results[name]['mape'])
        } for name in results
    }
}

metrics_path = os.path.join(results_dir, 'model_c_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_summary, f, indent=4)
print(f"✓ Métriques sauvegardées: {metrics_path}")

# Comparaison
comparison_path = os.path.join(results_dir, 'model_c_comparison.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"✓ Comparaison sauvegardée")

print("\n" + "="*70)
print("✅ MODEL C - ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
print("="*70)
print(f"\n📊 RÉSULTATS FINAUX:")
print(f"   Meilleur modèle: {best_model_name}")
print(f"   Test RMSE: {results[best_model_name]['rmse']:.4f} tonnes/ha (Objectif: 0.8-1.5)")
print(f"   Test MAE: {results[best_model_name]['mae']:.4f} tonnes/ha")
print(f"   Test R²: {results[best_model_name]['r2']:.4f} (Objectif: 0.85-0.92)")
print(f"   Test MAPE: {results[best_model_name]['mape']:.2f}%")
print(f"   CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

if results[best_model_name]['r2'] >= 0.85:
    print("\n   🎯 OBJECTIF ATTEINT! ✅")
else:
    print(f"\n   ⚠️  Proche de l'objectif (écart R²: {0.85 - results[best_model_name]['r2']:.4f})")

print(f"\n📁 Fichiers générés:")
print(f"   - Modèle: {model_path}")
print(f"   - Métriques: {metrics_path}")
print(f"   - Graphiques: {results_dir}/")
print("\n" + "="*70)
