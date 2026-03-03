"""
10-Fold Cross-Validation for Model D (Decision Tree - Binary Irrigation Control)
Dataset: IoT Agriculture 2024 (37,923 samples, 2 classes: OFF/ON)
Purpose: Validate 100% accuracy claim for IEEE IBI 2026 paper
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("=" * 80)
print("MODEL D: 10-Fold Cross-Validation - Decision Tree Binary Irrigation Control")
print("=" * 80)

data_path = r"C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\data\IoT Agriculture 2024\IoTProcessed_Data.csv"
df = pd.read_csv(data_path)

print(f"\nDataset loaded: {len(df)} samples")
print(f"Columns: {df.columns.tolist()}")

# Prepare features and target
# Target: Watering_plant_pump (OFF=0, ON=1)
X = df[['tempreature', 'humidity', 'water_level', 'N', 'P', 'K']]
y = df['Watering_plant_pump_ON'].astype(int)

print(f"\nFeatures: {X.columns.tolist()}")
print(f"Number of features: {X.shape[1]}")
print(f"Target classes: {y.unique()}")
print(f"Class distribution:\n{y.value_counts()}")

# 10-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_accuracies = []
fold_confusion_matrices = []
fold_reports = []

print("\n" + "=" * 80)
print("Starting 10-Fold Cross-Validation...")
print("=" * 80)

for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
    print(f"\n--- Fold {fold_idx}/10 ---")
    
    # Split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Decision Tree model (same hyperparameters as original Model D)
    model = DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
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
    report = classification_report(y_test, y_pred, target_names=['OFF', 'ON'], zero_division=0)
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
results_csv_path = r"C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_d_10fold_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\nResults saved to: {results_csv_path}")

# Calculate average confusion matrix
avg_confusion_matrix = np.mean(fold_confusion_matrices, axis=0).astype(int)

# Plot average confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(avg_confusion_matrix, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['OFF', 'ON'], 
            yticklabels=['OFF', 'ON'],
            cbar_kws={'label': 'Count'})
plt.title(f'Model D: Average Confusion Matrix (10-Fold CV)\nMean Accuracy: {mean_accuracy * 100:.2f}% ± {std_accuracy * 100:.2f}%', 
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('True Class', fontsize=12)
plt.tight_layout()
cm_path = r"C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_d_10fold_confusion_matrix.png"
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to: {cm_path}")
plt.close()

# Plot accuracy distribution across folds
plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), [acc * 100 for acc in fold_accuracies], color='forestgreen', alpha=0.8, edgecolor='black')
plt.axhline(y=mean_accuracy * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_accuracy * 100:.2f}%')
plt.axhline(y=(mean_accuracy - std_accuracy) * 100, color='orange', linestyle=':', linewidth=1.5, label=f'Mean - Std')
plt.axhline(y=(mean_accuracy + std_accuracy) * 100, color='orange', linestyle=':', linewidth=1.5, label=f'Mean + Std')
plt.xlabel('Fold Number', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title(f'Model D: Accuracy Distribution Across 10 Folds\nMean: {mean_accuracy * 100:.2f}% ± {std_accuracy * 100:.2f}%', 
          fontsize=14, fontweight='bold')
plt.xticks(range(1, 11))
plt.ylim(99.0, 100.5)
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
acc_dist_path = r"C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\models IA\results\model_d_10fold_accuracy_distribution.png"
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
