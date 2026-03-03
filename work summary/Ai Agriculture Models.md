4. AI Model Design and Implementation
4.1 Overview of the Proposed Multi-Model Architecture
The proposed system relies on a multi-model Artificial Intelligence (AI) architecture integrating heterogeneous datasets from smart agriculture environments. The main objective is to optimize both resource consumption (water and energy) and agricultural productivity through predictive modeling and intelligent control. The architecture is composed of four main families of AI models operating either in parallel or in cascade depending on the task: (A) Plant Growth and Health Models, (B) Crop Recommendation Models, (C) Yield Prediction Models, and (D) Intelligent Irrigation and Climate Control Models. An additional (E) Sustainable Multi-Agent AI layer can be added to integrate global decision-making across agents (farmer, environment, and market).
4.2 Family A — Plant Growth and Health Models
Datasets used:
• Advanced IoT Agriculture 2024
• Greenhouse Plant Growth Metrics
Objective:
Predict the physiological health and growth class of plants based on environmental and morphological variables. This enables early detection of plant stress, nutrient deficiency, or abnormal growth patterns.
Task type:
Multiclass classification or regression.
Input features:
ACHP, PHR, ALAP, AWWGV, ARD, ADWR, PDMVG, ARL, AWWR, ADWV, PDMRG
Target variable:
Class (SA–TC)
Recommended models:
• Random Forest
• XGBoost / LightGBM
• Multilayer Perceptron (MLP)
Remarks:
MLP networks capture non-linear dependencies between morphological variables (e.g., root–leaf correlation). Ensemble models like XGBoost enhance robustness against noise typical in IoT data.
4.3 Family B — Crop Recommendation Models
Datasets used:
• Crop Recommendation Dataset
• Smart Agricultural Production Optimizing Engine
Objective:
Recommend the most suitable crop type (e.g., rice, maize, wheat) according to soil and climatic conditions, maximizing productivity while minimizing resource consumption.
Task type:
Multiclass classification.
Input features:
N, P, K, temperature, humidity, pH, rainfall
Target variable:
label (crop type)
Recommended models:
• Random Forest
• Gradient Boosting / LightGBM
• TabNet or 1D CNN (for embedded deep learning)
Remarks:
Both datasets share similar feature structures, allowing for data fusion and transfer learning across domains (open field and greenhouse).
4.4 Family C — Yield Prediction Models
Datasets used:
• Agriculture Crop Yield
• Smart Farming Sensor Data for Yield Prediction
Objective:
Predict crop yield (in tons or kg per hectare) based on environmental and agronomic factors such as rainfall, temperature, fertilizer usage, and soil conditions.
Task type:
Supervised regression.
Input features:
Soil_Type, Rainfall_mm, Temperature_Celsius, Fertilizer_Used, Irrigation_Used, Weather_Condition, Days_to_Harvest
Target variable:
Yield_tons_per_hectare (or equivalent)
Recommended models:
• CatBoost / LightGBM
• Deep Neural Networks for regression (DNNs)
Remarks:
Feature engineering may include new variables such as vegetation indices (NDVI) or disease presence, improving generalization across crop types.
4.5 Family D — Intelligent Irrigation and Climate Control Models
Datasets used:
• IoT Agriculture 2024
• Smart Agriculture Dataset
Objective:
Control irrigation systems and climatic actuators (fans, pumps) automatically using IoT sensor inputs. The system aims to maintain optimal growing conditions while minimizing water and energy consumption.
Task type:
Binary classification / control optimization.
Input features:
• Dataset 6: temperature, humidity, water_level, N, P, K, actuators
• Dataset 7: MOI, temp, humidity, soil_type, Seedling_Stage
Target variable:
result (or ON/OFF action label)
Recommended models:
• Decision Tree / Logistic Regression for embedded deployment
• Lightweight Neural Networks for real-time IoT integration (TinyML)
Remarks:
This model can be embedded into microcontrollers such as ESP32 for on-device decision making. Threshold adaptation based on environmental learning ensures real-time optimization.
4.6 Family E — Sustainable Multi-Agent AI (Optional Advanced Layer)
Dataset used:
• AI for Sustainable Agriculture
Objective:
Integrate distributed decision-making using multiple intelligent agents—representing the farmer, the climate system, and the market. This level of reasoning enables global optimization of sustainability objectives (profit, water usage, environmental impact).
Task type:
Reinforcement Learning / Multi-Agent Collaboration.
Agents:
• Farmer Agent – Crop management decisions
• Climate Agent – Weather and irrigation regulation
• Market Agent – Pricing and demand adaptation
• Sustainability Agent – Long-term resource optimization
Recommended algorithms:
• Deep Q-Learning (DQN)
• Proximal Policy Optimization (PPO)
• Hybrid Fuzzy Logic + Ontology reasoning
4.7 Integration and Data Flow
The multi-model system can be deployed in two complementary configurations:
Parallel processing: Each family operates independently on its specific dataset and outputs insights (growth status, crop recommendation, yield prediction, irrigation decision).

Cascaded architecture: The outputs of one model serve as inputs for another, for example:
oModel B (crop recommendation) feeds into Model C (yield prediction).

oModel A (plant health) influences Model D (irrigation control).
Each model’s results are combined within a decision fusion module, which aggregates predictions through weighted averaging or rule-based logic. The system is designed to be easily connected to IoT platforms or microcontrollers for real-time adaptive control.
4.8 Summary Table
Dataset ID	Dataset Name	Model	Task Type	Objective	Recommended Algorithm
1	Advanced IoT Agriculture 2024	A	Classification	Plant growth health	XGBoost / MLP
2	Agriculture Crop Yield	C	Regression	Yield prediction	LightGBM / DNN
3	AI for Sustainable Agriculture	E	Multi-Agent	Global sustainability	DQN / PPO
4	Crop Recommendation	B	Classification	Crop suitability	Random Forest
5	Greenhouse Plant Growth	A	Classification	Growth and stress	MLP
6	IoT Agriculture 2024	D	Control	Irrigation management	Decision Tree
7	Smart Agriculture Dataset	D	Classification	Climate control	Logistic Regression
8	Smart Farming Sensor Data	C	Regression	Sensor-based yield	CatBoost
9	Smart Prod Optimizing Engine	B	Classification	Smart recommendation	TabNet
