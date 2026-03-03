"""
10-Fold Cross-Validation for Model A (XGBoost - Plant Health Classification)
Dataset: Advanced IoT Agriculture 2024 (30,000 samples, 6 classes)
Purpose: Validate 100% accuracy claim for IEEE IBI 2026 paper
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("=" * 80)
print("MODEL A: 10-Fold Cross-Validation - XGBoost Plant Health Classification")
print("=" * 80)

data_path = r"C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\data\Advanced IoT Agriculture 2024\Advanced_IoT_Dataset.csv"
df = pd.read_csv(data_path)

print(f"\nDataset loaded: {len(df)} samples")
print(f"Classes: {df['Class'].unique()}")
print(f"Class distribution:\n{df['Class'].value_counts()}")

# Prepare features and target
X = df.drop(columns=['Class', 'Random'])
y = df['Class']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nFeatures: {X.columns.tolist()}")
print(f"Number of features: {X.shape[1]}")

# 10-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_accuracies = []
fold_confusion_matrices = []
fold_reports = []

print("\n" + "=" * 80)
print("Starting 10-Fold Cross-Validation...")
print("=" * 80)

for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y_encoded), start=1):
    print(f"\n--- Fold {fold_idx}/10 ---")
    
    # Split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model (same hyperparameters as original Model A)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)
    
    print(f"Fold {fold_idx} Accuracy: {accuracy * 100:.4f}%")
    print(f"Train samples: {len(train_index)}, Test samples: {len(test_index)}")
    
    # Store confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fold_confusion_matrices.append(cm)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
    fold_reports.append(report)

# Calculate statistics
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)
min_accuracy = np.min(fold_accuracies)
max_accuracy = np.max(fold_accuracies)

print("\n" + "=" * 80)
print("10-FOLD CROSS-VALIDATION RESULTS")
print("=" * 80)
print(f"\nMean Accuracy: {mean_accuracy * 100:.4f}%")
print(f"Standard Deviation: {std_accuracy * 100:.4f}%")
print(f"Min Accuracy: {min_accuracy * 100:.4f}%")
print(f"Max Accuracy: {max_accuracy * 100:.4f}%")
print(f"\nAccuracy per fold:")
for i, acc in enumerate(fold_accuracies, start=1):
    print(f"  Fold {i:2d}: {acc * 100:.4f}%")

# Save results to CSV
results_df = pd.DataFrame({
    'Fold': list(range(1, 11)),
    'Accuracy': [f"{acc * 100:.4f}%" for acc in fold_accuracies]
})
results_df.loc[len(results_df)] = ['Mean ± Std', f"{mean_accuracy * 100:.4f}% ± {std_accuracy * 100:.4f}%"]
results_csv_path = r"C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_a_10fold_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\nResults saved to: {results_csv_path}")

# Calculate average confusion matrix
avg_confusion_matrix = np.mean(fold_confusion_matrices, axis=0).astype(int)

# Plot average confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(avg_confusion_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_,
            cbar_kws={'label': 'Count'})
plt.title(f'Model A: Average Confusion Matrix (10-Fold CV)\nMean Accuracy: {mean_accuracy * 100:.2f}% ± {std_accuracy * 100:.2f}%', 
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('True Class', fontsize=12)
plt.tight_layout()
cm_path = r"C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_a_10fold_confusion_matrix.png"
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to: {cm_path}")
plt.close()

# Plot accuracy distribution across folds
plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), [acc * 100 for acc in fold_accuracies], color='steelblue', alpha=0.8, edgecolor='black')
plt.axhline(y=mean_accuracy * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_accuracy * 100:.2f}%')
plt.axhline(y=(mean_accuracy - std_accuracy) * 100, color='orange', linestyle=':', linewidth=1.5, label=f'Mean - Std')
plt.axhline(y=(mean_accuracy + std_accuracy) * 100, color='orange', linestyle=':', linewidth=1.5, label=f'Mean + Std')
plt.xlabel('Fold Number', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title(f'Model A: Accuracy Distribution Across 10 Folds\nMean: {mean_accuracy * 100:.2f}% ± {std_accuracy * 100:.2f}%', 
          fontsize=14, fontweight='bold')
plt.xticks(range(1, 11))
plt.ylim(99.0, 100.5)
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
acc_dist_path = r"C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_a_10fold_accuracy_distribution.png"
plt.savefig(acc_dist_path, dpi=300, bbox_inches='tight')
print(f"Accuracy distribution saved to: {acc_dist_path}")
plt.close()

# Print detailed classification report for last fold
print("\n" + "=" * 80)
print("Classification Report (Fold 10 - Representative Sample)")
print("=" * 80)
print(fold_reports[-1])

print("\n" + "=" * 80)
print("VALIDATION COMPLETE!")
print("=" * 80)
print(f"\nFor IEEE paper, report as:")
print(f"  Mean Accuracy: {mean_accuracy * 100:.2f}% ± {std_accuracy * 100:.2f}% (10-fold cross-validation)")
print("=" * 80)
