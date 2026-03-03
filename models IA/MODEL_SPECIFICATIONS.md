# 🤖 SPÉCIFICATIONS COMPLÈTES DES MODÈLES IA - AGRICULTURE DURABLE

**Auteur:** Montassar Nawara  
**Date:** Novembre 2025  
**Projet:** AI-Based Energy and Water Optimization for Sustainable Agriculture

---

## 📋 TABLE DES MATIÈRES

1. [Model Family A - Plant Growth and Health](#model-family-a)
2. [Model Family B - Crop Recommendation](#model-family-b)
3. [Model Family C - Yield Prediction](#model-family-c)
4. [Model Family D - Intelligent Irrigation Control](#model-family-d)
5. [Model Family E - Decision Fusion](#model-family-e)
6. [Pipeline d'implémentation](#pipeline)

---

## 🌱 MODEL FAMILY A - Plant Growth and Health Classification

### **Objectif**
Classifier les stades de croissance des plantes et détecter les conditions de stress basées sur des paramètres morphologiques et physiologiques.

### **Dataset Principal**
- **Nom:** Advanced IoT Agriculture 2024
- **Fichier:** `data/Advanced IoT Agriculture 2024/Advanced_IoT_Dataset.csv`
- **Taille:** 30,000 enregistrements
- **Type de tâche:** Classification multiclasse (6 classes)

### **Features d'Entrée (Input)**
```python
input_features = [
    'ACHP',    # Average chlorophyll in plant (float)
    'PHR',     # Plant height rate (float)
    'AWWGV',   # Average wet weight of growth vegetative (float)
    'ALAP',    # Average leaf area of plant (float)
    'ANPL',    # Average number of plant leaves (float)
    'ARD',     # Average root diameter (float)
    'ADWR',    # Average dry weight of root (float)
    'PDMVG',   # Percentage of dry matter for vegetative growth (float)
    'ARL',     # Average root length (float)
    'AWWR',    # Average wet weight of root (float)
    'ADWV',    # Average dry weight of vegetative plants (float)
    'PDMRG',   # Percentage of dry matter for root growth (float)
]
# Total: 12 features numériques
```

### **Target Output**
```python
target = 'Class'  # 6 classes: SA, SB, SC, TA, TB, TC
# SA, SB, SC: Plantes cultivées en serre IoT
# TA, TB, TC: Plantes cultivées en serre traditionnelle
```

### **Algorithmes à Implémenter**
1. **XGBoost Classifier**
   - Hyperparamètres suggérés:
     - `n_estimators`: 100-500
     - `max_depth`: 5-10
     - `learning_rate`: 0.01-0.1
     - `subsample`: 0.8
     - `colsample_bytree`: 0.8

2. **Random Forest Classifier**
   - Hyperparamètres suggérés:
     - `n_estimators`: 200-500
     - `max_depth`: 10-30
     - `min_samples_split`: 2-5
     - `min_samples_leaf`: 1-2

3. **Multi-Layer Perceptron (MLP)**
   - Architecture suggérée:
     - Input layer: 12 neurons
     - Hidden layers: [64, 32] neurons
     - Output layer: 6 neurons (softmax)
     - Activation: ReLU
     - Dropout: 0.2-0.3

### **Métriques d'Évaluation**
- **Accuracy** (objectif: 85-92%)
- **F1-Score** par classe (objectif: 0.88-0.91)
- **Confusion Matrix** 6x6
- **Precision/Recall** par classe
- **ROC-AUC** (One-vs-Rest)

### **Preprocessing Requis**
```python
preprocessing_steps = {
    'missing_values': 'Supprimer lignes avec NaN (très rare)',
    'outliers': 'IQR method avec seuil 1.5',
    'normalization': 'StandardScaler (z-score)',
    'train_test_split': '70/15/15 (train/val/test)',
    'cross_validation': '5-fold stratified'
}
```

### **Feature Importance attendue**
- Top 3: ACHP (chlorophyll), ALAP (leaf area), PHR (height rate)

### **Graphiques à Générer**
1. Confusion Matrix heatmap
2. Feature Importance bar chart
3. ROC curves (One-vs-Rest)
4. Training/Validation loss curves (pour MLP)
5. Distribution des classes

---

## 🌾 MODEL FAMILY B - Crop Recommendation System

### **Objectif**
Recommander le type de culture optimal basé sur les nutriments du sol (N, P, K), conditions environnementales et pH.

### **Datasets Principaux**
- **Nom 1:** Crop Recommendation Dataset
- **Fichier 1:** `data/Crop Recommendation Dataset/Crop_recommendation.csv`
- **Nom 2:** Smart Agricultural Production Optimizing Engine
- **Fichier 2:** `data/Smart_Agricultural Production Optimizing Engine/Crop_recommendation.csv`
- **Type de tâche:** Classification multiclasse (21+ cultures)

### **Features d'Entrée (Input)**
```python
input_features = [
    'N',              # Nitrogen content (kg/ha)
    'P',              # Phosphorus content (kg/ha)
    'K',              # Potassium content (kg/ha)
    'temperature',    # Temperature (°C)
    'humidity',       # Relative humidity (%)
    'ph',             # Soil pH value
    'rainfall',       # Rainfall (mm)
]
# Total: 7 features numériques
```

### **Target Output**
```python
target = 'label'  # 21 crop types
crops = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
    'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
    'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
    'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'
]
```

### **Algorithmes à Implémenter**
1. **Random Forest Classifier**
   - Hyperparamètres suggérés:
     - `n_estimators`: 300-500
     - `max_depth`: 15-25
     - `criterion`: 'gini' ou 'entropy'

2. **LightGBM Classifier**
   - Hyperparamètres suggérés:
     - `n_estimators`: 200-500
     - `max_depth`: 8-15
     - `learning_rate`: 0.05-0.1
     - `num_leaves`: 31-63

### **Métriques d'Évaluation**
- **Multiclass Accuracy** (objectif: 90-95%)
- **F1-Score macro** (objectif: 0.89-0.93)
- **Precision/Recall** par culture
- **Confusion Matrix** 21x21
- **Classification Report** complet

### **Preprocessing Requis**
```python
preprocessing_steps = {
    'missing_values': 'Imputation avec médiane',
    'outliers': 'Domain validation (N: 0-140, P: 5-145, K: 5-205)',
    'normalization': 'MinMaxScaler [0,1] pour Random Forest',
    'feature_engineering': [
        'N_P_ratio = N / (P + 1)',
        'N_K_ratio = N / (K + 1)',
        'NPK_sum = N + P + K'
    ],
    'train_test_split': '70/15/15 stratified',
    'class_balance': 'Check distribution, SMOTE si nécessaire'
}
```

### **Feature Importance attendue**
- Top 3: Rainfall, N (Nitrogen), K (Potassium)

### **Graphiques à Générer**
1. Confusion Matrix heatmap (21x21)
2. Feature Importance bar chart
3. Per-crop Precision/Recall bar chart
4. NPK ternary plot colored by crop type
5. Rainfall vs Temperature scatter colored by recommended crop

---

## 📈 MODEL FAMILY C - Crop Yield Prediction

### **Objectif**
Prédire le rendement des cultures (kg/ha ou tonnes/ha) basé sur les données historiques, conditions actuelles, irrigation et fertilisation.

### **Datasets Principaux**
- **Nom 1:** Agriculture Crop Yield
- **Fichier 1:** `data/Agriculture Crop Yield/crop_yield.csv`
- **Taille:** 1,000,000 échantillons
- **Nom 2:** Smart Farming Sensor Data for Yield Prediction
- **Fichier 2:** `data/Smart Farming Sensor Data for Yield Prediction/Smart_Farming_Crop_Yield_2024.csv`
- **Taille:** 500 farms
- **Type de tâche:** Régression

### **Features d'Entrée (Input)**
```python
# Dataset 1: Agriculture Crop Yield
input_features_1 = [
    'Region',              # Categorical: North, East, South, West
    'Soil_Type',           # Categorical: Clay, Sandy, Loam, Silt, Peaty, Chalky
    'Crop',                # Categorical: Wheat, Rice, Maize, Barley, Soybean
    'Rainfall_mm',         # Numeric: Annual rainfall
    'Temperature_Celsius', # Numeric: Average temperature
    'Fertilizer_kg_per_hectare', # Numeric: Fertilizer application
    'Irrigation_Schedule', # Categorical: Weekly, Biweekly, Monthly
    'Days_to_Harvest',     # Numeric: Growth period
]

# Dataset 2: Smart Farming (avec features IoT)
input_features_2 = [
    'Farm_ID', 'Crop_Type', 'Soil_Moisture', 'Temperature',
    'Humidity', 'Rainfall', 'NDVI', 'Fertilizer_Applied',
    'Pesticide_Used', 'Days_Since_Planting', 'Weather_Condition',
    'Irrigation_Type', 'Field_Size_hectares'
]
```

### **Target Output**
```python
target = 'Yield_tons_per_hectare'  # Régression continue
# Range attendu: 1.0 - 10.0 tonnes/hectare
```

### **Algorithmes à Implémenter**
1. **CatBoost Regressor**
   - Hyperparamètres suggérés:
     - `iterations`: 500-1000
     - `depth`: 6-10
     - `learning_rate`: 0.03-0.1
     - `loss_function`: 'RMSE'
     - Gère nativement les variables catégorielles

2. **LightGBM Regressor**
   - Hyperparamètres suggérés:
     - `n_estimators`: 500-1000
     - `max_depth`: 8-15
     - `learning_rate`: 0.05
     - `num_leaves`: 31

3. **Deep Neural Network (DNN)**
   - Architecture suggérée:
     - Input layer: nombre de features après encoding
     - Hidden layers: [128, 64, 32] neurons
     - Output layer: 1 neuron (linear)
     - Activation: ReLU
     - Dropout: 0.3

### **Métriques d'Évaluation**
- **RMSE** (Root Mean Squared Error) - objectif: 0.8-1.5 tonnes/ha
- **MAE** (Mean Absolute Error) - objectif: 0.5-1.0 tonnes/ha
- **R² Score** - objectif: 0.85-0.92
- **MAPE** (Mean Absolute Percentage Error) - objectif: <15%

### **Preprocessing Requis**
```python
preprocessing_steps = {
    'missing_values': 'Forward-fill pour time-series, médiane pour autres',
    'categorical_encoding': 'LabelEncoder pour CatBoost, OneHot pour autres',
    'outliers': 'Remove yields > 15 tonnes/ha (irréaliste)',
    'normalization': 'StandardScaler pour features numériques',
    'feature_engineering': [
        'GDD = sum((Tmax + Tmin)/2 - Tbase)',  # Growing Degree Days
        'Water_stress = (Rainfall - Evapotranspiration)',
        'Fertilizer_efficiency = Yield / Fertilizer_kg'
    ],
    'train_test_split': '70/15/15 temporal split'
}
```

### **Feature Importance attendue**
- Top 5: NDVI, Rainfall, Fertilizer_Applied, Days_to_Harvest, Soil_Type

### **Graphiques à Générer**
1. Predicted vs Actual yield scatter plot
2. Residuals plot
3. Feature Importance bar chart
4. Yield distribution histogram
5. SHAP values plot (pour interprétabilité)
6. Learning curves (train/val loss)

---

## 💧 MODEL FAMILY D - Intelligent Irrigation Control

### **Objectif**
Décision binaire en temps réel: activer l'irrigation (1) ou non (0) basée sur les capteurs IoT (moisture, température, humidité).

### **Datasets Principaux**
- **Nom 1:** IoT Agriculture 2024
- **Fichier 1:** `data/IoT Agriculture 2024/IoTProcessed_Data.csv`
- **Taille:** 37,923 enregistrements
- **Nom 2:** Smart Agriculture Dataset
- **Fichier 2:** `data/Smart Agriculture Dataset/cropdata_updated.csv`
- **Taille:** 16,411 enregistrements
- **Type de tâche:** Classification binaire (Control Decision)

### **Features d'Entrée (Input)**
```python
# Dataset 1: IoT Agriculture 2024
input_features_1 = [
    'TMP',    # Temperature (°C)
    'HUM',    # Humidity (%)
    'MOI',    # Soil Moisture (%)
    'NIT',    # Nitrogen (ppm)
    'PHOS',   # Phosphorus (ppm)
    'POT',    # Potassium (ppm)
    'PH',     # Soil pH
    'RAIN',   # Rainfall (mm)
]

# Dataset 2: Smart Agriculture (plus features)
input_features_2 = [
    'Soil_Moisture', 'Ambient_Temperature', 'Ambient_Humidity',
    'Soil_Temperature', 'Light_Intensity', 'CO2_Level',
    'Soil_pH', 'Soil_EC', 'NPK_N', 'NPK_P', 'NPK_K'
]
```

### **Target Output**
```python
target = 'Irrigation_Status'  # Binary: 0 (OFF) or 1 (ON)
# Alternative target names: 'Water_Pump', 'Fan_Status', 'Action'
```

### **Algorithmes à Implémenter**
1. **Decision Tree Classifier** ⭐ (pour embedded deployment)
   - Hyperparamètres suggérés:
     - `max_depth`: 5-10 (petit pour ESP32)
     - `min_samples_split`: 10-20
     - `criterion`: 'gini'
     - Objectif: Modèle <100 KB

2. **Logistic Regression** ⭐ (pour embedded deployment)
   - Hyperparamètres suggérés:
     - `penalty`: 'l2'
     - `C`: 0.1-1.0
     - `solver`: 'liblinear'
     - Objectif: Modèle <50 KB

3. **XGBoost Classifier** (pour edge/cloud)
   - Hyperparamètres suggérés:
     - `n_estimators`: 100-200
     - `max_depth`: 3-6
     - `learning_rate`: 0.1

### **Métriques d'Évaluation**
- **Accuracy** (objectif: 88-94%)
- **Precision** (important: éviter irrigation inutile)
- **Recall** (important: ne pas manquer irrigation nécessaire)
- **F1-Score** (objectif: 0.89-0.93)
- **Confusion Matrix**
- **Inference Latency** (objectif: <50ms sur ESP32, <10ms sur RPi)
- **Model Size** (objectif: <100 KB pour Decision Tree)

### **Preprocessing Requis**
```python
preprocessing_steps = {
    'missing_values': 'Forward-fill (time-series)',
    'outliers': 'Domain limits (Temp: -10 to 50°C, Moisture: 0-100%)',
    'normalization': 'MinMaxScaler [0,1]',
    'feature_engineering': [
        'Moisture_deficit = 80 - MOI',  # 80% est optimal
        'VPD = calculate_vpd(TMP, HUM)',  # Vapor Pressure Deficit
        'hour_sin = sin(2π * hour / 24)',
        'hour_cos = cos(2π * hour / 24)'
    ],
    'class_balance': 'Check ratio ON/OFF, SMOTE si déséquilibre >70/30',
    'train_test_split': '70/15/15 temporal split'
}
```

### **Feature Importance attendue**
- Top 3: MOI (Soil Moisture), TMP (Temperature), HUM (Humidity)

### **Graphiques à Générer**
1. Confusion Matrix
2. Precision-Recall curve
3. ROC curve with AUC
4. Feature Importance
5. Decision Tree visualization (si Decision Tree)
6. Threshold analysis plot
7. Time-series plot: Moisture + Irrigation decisions

---

## 🔀 MODEL FAMILY E - Decision Fusion & Optimization

### **Objectif**
Stacking meta-learner qui agrège les prédictions des Models A, B, C, D + variables d'état système pour générer des commandes optimales d'actuateurs.

### **Architecture**
Meta-learner de type **Stacking** avec **Model Predictive Control (MPC)**.

### **Features d'Entrée (Input)**
```python
input_features = [
    # Sorties des modèles A-D
    'health_score',           # Model A output (probabilité classe saine)
    'crop_suitability',       # Model B output (confidence score)
    'predicted_yield',        # Model C output (tonnes/ha)
    'irrigation_recommendation', # Model D output (0/1)
    
    # Variables d'état système
    'current_soil_moisture',  # Capteur en temps réel (%)
    'water_budget_remaining', # Budget eau restant (L)
    'energy_budget_remaining',# Budget énergie restant (kWh)
    'timestamp_hour',         # Heure du jour (0-23)
    'timestamp_day',          # Jour de la semaine (0-6)
    
    # Contexte agronomique
    'crop_growth_stage',      # Stade de croissance (encoded)
    'days_since_planting',    # Jours depuis plantation
]
# Total: ~12 features
```

### **Target Output**
```python
# Commandes pour actuateurs (multi-output)
outputs = {
    'water_pump_volume': 'Régression (0-50 L/m²)',
    'nutrient_pump_dose': 'Régression (0-100 ml)',
    'led_light_intensity': 'Régression (0-100%)',
    'fan_speed': 'Régression (0-100%)',
    'heater_target_temp': 'Régression (15-35°C)',
}
```

### **Architecture du Meta-Learner**
```python
meta_learner = {
    'type': 'Multi-Layer Perceptron (MLP)',
    'architecture': {
        'input_layer': 12,  # features
        'hidden_layer_1': 128,  # neurons, activation=ReLU
        'dropout_1': 0.2,
        'hidden_layer_2': 64,   # neurons, activation=ReLU
        'dropout_2': 0.2,
        'output_layer': 5,   # multi-output (5 actuators)
    },
    'loss_function': 'Custom multi-objective loss',
    'optimizer': 'Adam',
    'learning_rate': 0.001,
}
```

### **Model Predictive Control (MPC)**
```python
# Fonction objectif à minimiser
def objective_function(u):
    """
    u: vecteur de commandes d'actuateurs
    Returns: coût total
    """
    alpha = 1.0  # Poids eau
    beta = 1.0   # Poids énergie
    gamma = 2.0  # Poids rendement (plus important)
    
    water_cost = alpha * calculate_water_consumption(u)
    energy_cost = beta * calculate_energy_consumption(u)
    yield_penalty = -gamma * estimate_yield_impact(u)
    
    return water_cost + energy_cost + yield_penalty

# Contraintes
constraints = {
    'soil_moisture': (20, 100),  # %
    'daily_water': (0, 50),      # L/m²
    'temperature': (15, 35),     # °C
    'humidity': (40, 90),        # %
    'pH': (5.5, 7.5),
}

# Règles de sécurité (hard constraints)
safety_rules = [
    'max_pump_runtime <= 30 minutes',
    'heater_on AND cooler_on == False',  # Interlock
    'soil_moisture > 20%',  # Minimum critique
    'water_budget >= 0',
    'energy_budget >= 0',
]
```

### **Pipeline d'Entraînement**
```python
training_pipeline = {
    'step_1': 'Entraîner Models A, B, C, D séparément',
    'step_2': 'Générer prédictions sur validation set',
    'step_3': 'Créer meta-dataset (predictions + state + optimal_actions)',
    'step_4': 'Entraîner meta-learner MLP sur meta-dataset',
    'step_5': 'Intégrer MPC pour post-processing',
    'step_6': 'Validation end-to-end sur test set',
}
```

### **Métriques d'Évaluation**
- **System Accuracy** (objectif: 91-95%)
- **Water Consumption Reduction** (objectif: 20-30%)
- **Energy Consumption Reduction** (objectif: 10-25%)
- **Yield Maintenance** (objectif: ≥95% of baseline)
- **False Positive Irrigation Rate** (objectif: 30-40% reduction)
- **Inference Latency Total** (objectif: <200ms sur RPi, <80ms sur cloud)
- **Constraint Violation Rate** (objectif: <1%)

### **Preprocessing & Feature Engineering**
```python
preprocessing = {
    'model_outputs_normalization': 'MinMaxScaler [0,1]',
    'temporal_encoding': 'Cyclical (sin/cos)',
    'state_variables': 'StandardScaler',
    'ensemble_method': 'Stacking (MLP meta-learner)',
}
```

### **Graphiques à Générer**
1. Multi-objective optimization convergence plot
2. Water vs Energy consumption scatter (avec Pareto front)
3. System accuracy over time
4. Actuator commands time-series
5. Constraint satisfaction heatmap
6. Model ablation study (contribution de chaque modèle A-D)

---

## 🛠️ PIPELINE D'IMPLÉMENTATION COMPLÈTE

### **Phase 1: Préparation des Données**
```python
tasks = [
    '1. Charger tous les datasets depuis data/',
    '2. EDA (Exploratory Data Analysis) pour chaque dataset',
    '3. Nettoyage: missing values, outliers, duplicates',
    '4. Feature engineering par modèle',
    '5. Train/Val/Test split (70/15/15)',
    '6. Sauvegarder datasets préprocessés',
]
```

### **Phase 2: Entraînement des Modèles A-D**
```python
for model in ['A', 'B', 'C', 'D']:
    steps = [
        'Charger dataset préprocessé',
        'Grid Search / Bayesian Optimization hyperparameters',
        '5-Fold Cross-Validation',
        'Entraîner sur train set',
        'Évaluer sur validation set',
        'Fine-tuning',
        'Test final sur test set',
        'Sauvegarder modèle + metrics + plots',
    ]
```

### **Phase 3: Entraînement du Model E (Fusion)**
```python
steps = [
    'Charger Models A, B, C, D pré-entraînés',
    'Générer prédictions sur validation data',
    'Créer meta-dataset',
    'Entraîner MLP meta-learner',
    'Implémenter MPC optimization',
    'Validation end-to-end',
    'Générer graphiques système complet',
]
```

### **Phase 4: Génération des Résultats pour Paper**
```python
outputs = [
    'Tables: Performance metrics pour chaque modèle',
    'Confusion matrices (Models A, B, D)',
    'Feature importance charts',
    'Predicted vs Actual plots (Model C)',
    'System-level metrics: water/energy savings',
    'Inference latency benchmarks (ESP32, RPi, Cloud)',
    'Model size comparison table',
    'ROC curves, Precision-Recall curves',
]
```

### **Phase 5: Export pour Paper LaTeX**
```python
exports = {
    'figures': 'PNG 300dpi pour LaTeX',
    'tables': 'CSV → LaTeX tabular format',
    'metrics': 'JSON summary file',
    'model_sizes': 'KB/MB pour deployment analysis',
}
```

---

## 📊 STRUCTURE DES FICHIERS À CRÉER

```
models IA/
├── MODEL_SPECIFICATIONS.md (ce fichier)
├── notebooks/
│   ├── 01_EDA_all_datasets.ipynb
│   ├── 02_Model_A_Plant_Health.ipynb
│   ├── 03_Model_B_Crop_Recommendation.ipynb
│   ├── 04_Model_C_Yield_Prediction.ipynb
│   ├── 05_Model_D_Irrigation_Control.ipynb
│   ├── 06_Model_E_Fusion_MPC.ipynb
│   └── 07_Results_for_Paper.ipynb
├── models/
│   ├── model_a_xgboost.pkl
│   ├── model_a_rf.pkl
│   ├── model_a_mlp.h5
│   ├── model_b_rf.pkl
│   ├── model_b_lgbm.pkl
│   ├── model_c_catboost.pkl
│   ├── model_c_lgbm.pkl
│   ├── model_c_dnn.h5
│   ├── model_d_decision_tree.pkl
│   ├── model_d_logistic.pkl
│   └── model_e_fusion_mlp.h5
├── preprocessed_data/
│   ├── model_a_data.csv
│   ├── model_b_data.csv
│   ├── model_c_data.csv
│   ├── model_d_data.csv
│   └── meta_dataset.csv
├── results/
│   ├── figures/
│   │   ├── model_a_confusion_matrix.png
│   │   ├── model_b_crop_performance.png
│   │   ├── model_c_yield_prediction.png
│   │   ├── model_d_irrigation_roc.png
│   │   └── model_e_system_performance.png
│   ├── tables/
│   │   ├── model_performance_summary.csv
│   │   ├── feature_importance.csv
│   │   └── deployment_metrics.csv
│   └── metrics_summary.json
└── src/
    ├── preprocessing.py
    ├── model_a.py
    ├── model_b.py
    ├── model_c.py
    ├── model_d.py
    ├── model_e.py
    ├── evaluation.py
    └── utils.py
```

---

## 🎯 MÉTRIQUES CIBLES GLOBALES (pour le Paper)

| Modèle | Métrique Principale | Objectif | Status |
|--------|---------------------|----------|--------|
| Model A | F1-Score | 0.88-0.91 | À entraîner |
| Model B | Accuracy | 90-95% | À entraîner |
| Model C | R² Score | 0.85-0.92 | À entraîner |
| Model D | F1-Score | 0.89-0.93 | À entraîner |
| Model E | System Accuracy | 91-95% | À entraîner |
| **Système** | **Water Reduction** | **20-30%** | **À valider** |
| **Système** | **Energy Reduction** | **10-25%** | **À valider** |

---

## 💻 ENVIRONNEMENT DE DÉVELOPPEMENT

### **Librairies Python Requises**
```bash
# Core ML
pip install numpy pandas scikit-learn

# Gradient Boosting
pip install xgboost lightgbm catboost

# Deep Learning
pip install tensorflow  # ou pytorch

# Visualization
pip install matplotlib seaborn plotly

# Utilities
pip install jupyter notebook tqdm

# Model Interpretation
pip install shap

# Optimization (pour MPC)
pip install scipy cvxpy
```

### **Versions Recommandées**
```
Python: 3.8+
scikit-learn: 1.0+
xgboost: 1.6+
lightgbm: 3.3+
catboost: 1.0+
tensorflow: 2.8+
```

---

## 📝 NOTES IMPORTANTES

1. **Temporal Splits**: Pour les données time-series (IoT), toujours utiliser temporal split pour éviter data leakage.

2. **Class Imbalance**: Vérifier la distribution des classes, appliquer SMOTE si nécessaire.

3. **Embedded Deployment**: Model D doit être optimisé pour ESP32 (<100 KB). Utiliser TensorFlow Lite ou ONNX.

4. **Hyperparameter Tuning**: Utiliser Grid Search ou Bayesian Optimization (Optuna).

5. **Reproductibilité**: Fixer les random seeds partout (`np.random.seed(42)`).

6. **Validation**: Toujours faire 5-Fold Cross-Validation pour robustesse.

7. **Feature Engineering**: Les features dérivées (VPD, GDD, ratios NPK) améliorent souvent les performances de 5-10%.

8. **Model E**: Nécessite que Models A-D soient déjà entraînés et sauvegardés.

---

## ✅ CHECKLIST AVANT INTÉGRATION AU PAPER

- [ ] Tous les modèles entraînés avec CV
- [ ] Métriques documentées dans `metrics_summary.json`
- [ ] Figures haute résolution (300dpi PNG)
- [ ] Tables exportées en format LaTeX
- [ ] Feature importance calculée pour tous les modèles
- [ ] Confusion matrices générées
- [ ] Model sizes mesurés (KB/MB)
- [ ] Inference latency benchmarké (ESP32, RPi, Cloud)
- [ ] Water/Energy savings calculés (simulation)
- [ ] Comparaison avec baseline (rule-based system)

---

**Bon courage pour l'implémentation! 🚀**

Si tu as des questions sur un modèle spécifique, n'hésite pas!
