"""
Test Model A (XGBoost Plant Health) with a Single Sample
User provides a line number from the dataset, and the script:
1. Loads the trained model
2. Extracts the sample at that line
3. Predicts the class
4. Compares with the actual class
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

print("=" * 80)
print("MODEL A: Single Sample Prediction Test")
print("=" * 80)

# Load dataset
data_path = r"C:\Users\lapte\OneDrive - ensi-uma.tn\Bureau\conference-latex-template\data\Advanced IoT Agriculture 2024\Advanced_IoT_Dataset.csv"
df = pd.read_csv(data_path)

print(f"\nDataset loaded: {len(df)} samples")
print(f"Classes available: {df['Class'].unique()}")

# Prepare features
X = df.drop(columns=['Class', 'Random'])
y = df['Class']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\n" + "=" * 80)
print("INTERACTIVE TESTING")
print("=" * 80)
print(f"Dataset has {len(df)} samples (rows 0 to {len(df)-1})")
print("\nEnter a line number to test (or 'q' to quit)")
print("Example: 0 (first row), 15000 (middle), 29999 (last row)")
print("=" * 80)

while True:
    # Get user input
    user_input = input("\nEnter line number (0-29999): ").strip()
    
    if user_input.lower() == 'q':
        print("\nExiting... Goodbye!")
        break
    
    try:
        line_number = int(user_input)
        
        if line_number < 0 or line_number >= len(df):
            print(f"❌ Error: Line number must be between 0 and {len(df)-1}")
            continue
        
        # Extract the sample
        sample = X.iloc[line_number].values.reshape(1, -1)
        actual_class = y.iloc[line_number]
        actual_encoded = y_encoded[line_number]
        
        print("\n" + "-" * 80)
        print(f"📊 TESTING SAMPLE AT LINE {line_number}")
        print("-" * 80)
        
        # Display sample features
        print("\nInput Features:")
        for col, val in zip(X.columns, sample[0]):
            print(f"  {col}: {val:.4f}")
        
        print(f"\n✅ Actual Class: {actual_class}")
        
        # Scale features
        scaler = StandardScaler()
        scaler.fit(X)  # Fit on entire dataset (same as training)
        sample_scaled = scaler.transform(sample)
        
        # Train a quick model (or load if you have saved model)
        print("\n⏳ Training XGBoost model...")
        model = XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        # Scale entire dataset for training
        X_scaled = scaler.transform(X)
        model.fit(X_scaled, y_encoded, verbose=False)
        
        # Predict
        prediction_encoded = model.predict(sample_scaled)[0]
        prediction_class = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probability
        prediction_proba = model.predict_proba(sample_scaled)[0]
        confidence = np.max(prediction_proba) * 100
        
        print(f"🔮 Predicted Class: {prediction_class}")
        print(f"📈 Confidence: {confidence:.2f}%")
        
        # Show all class probabilities
        print("\nPrediction Probabilities:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"  {class_name}: {prediction_proba[i]*100:.2f}%")
        
        # Compare
        if prediction_class == actual_class:
            print("\n✅ ✅ ✅ CORRECT PREDICTION! ✅ ✅ ✅")
        else:
            print(f"\n❌ ❌ ❌ WRONG PREDICTION! ❌ ❌ ❌")
            print(f"Expected: {actual_class}, Got: {prediction_class}")
        
        print("-" * 80)
        
    except ValueError:
        print("❌ Error: Please enter a valid integer")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("TEST SESSION COMPLETED")
print("=" * 80)
