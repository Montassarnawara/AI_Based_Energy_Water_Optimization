# 📊 RÉCAPITULATIF COMPLET DES DATASETS AGRICOLES

Ce document présente une description détaillée de tous les datasets agricoles disponibles, incluant leurs colonnes et attributs.

---

## 1. 🌱 Advanced IoT Agriculture 2024

**Fichier de données :** `Advanced_IoT_Dataset.csv`  
**Nombre d'entrées :** 30,000 enregistrements  
**Nombre de colonnes :** 14

### Description du Dataset
Ce dataset a été collecté dans le cadre d'une recherche de thèse de master menée par l'étudiant Mohammed Ismail Lifta (2023-2024) au Département d'Informatique, Collège d'Informatique et de Mathématiques - Université de Tikrit, Irak. Les données ont été collectées à partir du laboratoire d'agriculture sur des plantes cultivées dans une serre IoT et une serre traditionnelle. L'étude a été supervisée par le Professeur Assistant Wisam Dawood Abdullah, administrateur de Cisco Networking Academy / Université de Tikrit.

### Colonnes et Attributs

1. **Random** (object)
   - Identifiant pour chaque enregistrement
   - Valeurs : R1, R2, R3 représentant différents échantillons aléatoires

2. **ACHP - Average of chlorophyll in the plant** (float)
   - Teneur moyenne en chlorophylle dans la plante
   - La chlorophylle est vitale pour la photosynthèse
   - Indique la santé et l'efficacité de la plante à convertir l'énergie lumineuse en énergie chimique

3. **PHR - Plant height rate** (float)
   - Taux de croissance de la hauteur de la plante
   - Mesure essentielle pour comprendre la dynamique de croissance verticale de la plante au fil du temps

4. **AWWGV - Average wet weight of the growth vegetative** (float)
   - Poids humide moyen de la croissance végétative
   - Indicateur de la teneur en eau et de la biomasse globale de la croissance végétative de la plante

5. **ALAP - Average leaf area of the plant** (float)
   - Surface foliaire moyenne de la plante
   - Facteur critique pour la photosynthèse, détermine la surface disponible pour l'absorption de la lumière

6. **ANPL - Average number of plant leaves** (float)
   - Nombre moyen de feuilles par plante
   - Corrèle avec la capacité de la plante à effectuer la photosynthèse et sa santé globale

7. **ARD - Average root diameter** (float)
   - Diamètre moyen des racines de la plante
   - Affecte la capacité de la plante à absorber l'eau et les nutriments du sol

8. **ADWR - Average dry weight of the root** (float)
   - Poids sec moyen des racines de la plante
   - Mesure de la biomasse de la plante après élimination de la teneur en eau
   - Indicateur de la capacité structurelle et de stockage de la racine

9. **PDMVG - Percentage of dry matter for vegetative growth** (float)
   - Pourcentage de matière sèche dans la croissance végétative
   - Indique la proportion de la biomasse de la plante qui n'est pas de l'eau
   - Crucial pour comprendre son état structurel et nutritionnel

10. **ARL - Average root length** (float)
    - Longueur moyenne des racines de la plante
    - Influence la capacité de la plante à explorer et absorber les nutriments et l'eau du sol

11. **AWWR - Average wet weight of the root** (float)
    - Poids humide moyen des racines de la plante
    - Comprend la teneur en eau des racines
    - Indique leur biomasse globale et leur capacité de rétention d'eau

12. **ADWV - Average dry weight of vegetative plants** (float)
    - Poids sec moyen des parties végétatives de la plante
    - Reflète la biomasse structurelle de la plante sans teneur en eau

13. **PDMRG - Percentage of dry matter for root growth** (float)
    - Pourcentage de matière sèche dans la croissance des racines
    - Montre la proportion de la biomasse racinaire qui n'est pas de l'eau
    - Important pour évaluer la santé et la fonction des racines

14. **Class** (object)
    - Classe ou catégorie à laquelle appartient l'enregistrement de la plante
    - Valeurs : SA, SB, SC, TA, TB, TC
    - Représente différents groupes ou conditions sous lesquels les plantes ont été étudiées

**Licence :** CC BY-ND  
**Contact :** Prof. Wisam Dawood Abdullah (wisamdawood@tu.edu.iq)

---

## 2. 🌾 Agriculture Crop Yield

**Fichier de données :** `crop_yield.csv`  
**Nombre d'entrées :** 1,000,000 échantillons

### Description du Dataset
Ce dataset contient des données agricoles visant à prédire le rendement des cultures (en tonnes par hectare) en fonction de divers facteurs. Le dataset peut être utilisé pour des tâches de régression en apprentissage automatique, en particulier pour prédire la productivité des cultures.

### Colonnes et Attributs

1. **Region** (catégoriel)
   - Région géographique où la culture est cultivée
   - Valeurs : North (Nord), East (Est), South (Sud), West (Ouest)

2. **Soil_Type** (catégoriel)
   - Type de sol dans lequel la culture est plantée
   - Valeurs : Clay (Argile), Sandy (Sableux), Loam (Limon), Silt (Limon fin), Peaty (Tourbeux), Chalky (Calcaire)

3. **Crop** (catégoriel)
   - Type de culture cultivée
   - Valeurs : Wheat (Blé), Rice (Riz), Maize (Maïs), Barley (Orge), Soybean (Soja), Cotton (Coton)

4. **Rainfall_mm** (numérique)
   - Quantité de précipitations reçues en millimètres pendant la période de croissance de la culture

5. **Temperature_Celsius** (numérique)
   - Température moyenne pendant la période de croissance de la culture, mesurée en degrés Celsius

6. **Fertilizer_Used** (booléen)
   - Indique si de l'engrais a été appliqué
   - Valeurs : True (Oui), False (Non)

7. **Irrigation_Used** (booléen)
   - Indique si l'irrigation a été utilisée pendant la période de croissance de la culture
   - Valeurs : True (Oui), False (Non)

8. **Weather_Condition** (catégoriel)
   - Condition météorologique prédominante pendant la saison de croissance
   - Valeurs : Sunny (Ensoleillé), Rainy (Pluvieux), Cloudy (Nuageux)

9. **Days_to_Harvest** (numérique)
   - Nombre de jours pris pour que la culture soit récoltée après la plantation

10. **Yield_tons_per_hectare** (numérique) - **VARIABLE CIBLE**
    - Rendement total de la culture produit, mesuré en tonnes par hectare

---

## 3. 🤖 AI for Sustainable Agriculture Dataset

**Fichiers de données :** 
- `farmer_advisor_dataset.csv`
- `market_researcher_dataset.csv`

### Description du Dataset
Ce dataset est conçu pour soutenir le développement d'un système d'IA multi-agents visant à optimiser les pratiques agricoles tout en favorisant la durabilité. Il intègre des données provenant d'agriculteurs, de stations météorologiques et de tendances du marché pour permettre une prise de décision basée sur l'IA pour une agriculture efficace en ressources et rentable.

### Composition du Dataset

#### 1. **Données des Agriculteurs (Farmer Data)**
- Caractéristiques du terrain
- Préférences de cultures
- Contraintes financières
- Enregistrements de rendement passés

#### 2. **Données Météorologiques et du Sol (Weather & Soil Data)**
- Précipitations
- Température
- Humidité
- Humidité du sol
- Autres variables liées au climat affectant la croissance des cultures

#### 3. **Tendances du Marché (Market Trends)**
- Prix des cultures régionales
- Prévisions de la demande
- Modèles commerciaux pour aider les agriculteurs à maximiser les profits

#### 4. **Métriques de Durabilité (Sustainability Metrics)**
- Utilisation de l'eau
- Application de pesticides
- Empreinte carbone
- Indicateurs de santé du sol pour des recommandations agricoles respectueuses de l'environnement

**Objectif :** Construire des solutions d'IA intelligentes qui aident à réduire l'impact environnemental, optimiser les ressources agricoles et améliorer la prise de décision des agriculteurs.

---

## 4. 🎯 Crop Recommendation Dataset

**Fichier de données :** `Crop_recommendation.csv`

### Description du Dataset
L'agriculture de précision est une tendance actuelle. Elle aide les agriculteurs à prendre des décisions éclairées sur la stratégie agricole. Ce dataset permet aux utilisateurs de construire un modèle prédictif pour recommander les cultures les plus appropriées à cultiver dans une ferme particulière en fonction de divers paramètres.

Ce dataset a été construit en augmentant les datasets de précipitations, de climat et de données d'engrais disponibles pour l'Inde.

### Colonnes et Attributs

1. **N** (numérique)
   - Ratio de la teneur en azote (Nitrogen) dans le sol
   - Nutriment essentiel pour la croissance des plantes

2. **P** (numérique)
   - Ratio de la teneur en phosphore (Phosphorous) dans le sol
   - Important pour le développement des racines et la floraison

3. **K** (numérique)
   - Ratio de la teneur en potassium (Potassium) dans le sol
   - Aide à la régulation de l'eau et à la résistance aux maladies

4. **temperature** (numérique)
   - Température en degrés Celsius
   - Les températures du sol pour la bioactivité varient de 50 à 75°F

5. **humidity** (numérique)
   - Humidité relative en pourcentage (%)

6. **ph** (numérique)
   - Valeur du pH du sol
   - Échelle utilisée pour identifier la nature acide ou basique
   - pH < 7 : Nature acide
   - pH = 7 : Neutre
   - pH > 7 : Nature basique

7. **rainfall** (numérique)
   - Précipitations en millimètres (mm)

8. **label** (catégoriel) - **VARIABLE CIBLE**
   - Types de cultures
   - Valeurs : Rice (Riz), Maize (Maïs), Chickpea (Pois chiche), Kidney beans (Haricots rouges), Pigeonpeas (Pois d'Angole), Mothbeans, Mungbean (Haricot mungo), Blackgram, Lentil (Lentille), Pomegranate (Grenade), Banana (Banane), Mango (Mangue), Grapes (Raisin), Watermelon (Pastèque), Muskmelon (Melon), Apple (Pomme), Orange, Papaya (Papaye), Coconut (Noix de coco), Cotton (Coton), Jute, Coffee (Café)

---

## 5. 🏡 Greenhouse Plant Growth

**Fichier de données :** `Greenhouse Plant Growth Metrics.csv`  
**Nombre d'entrées :** 30,000 enregistrements  
**Nombre de colonnes :** 14

### Description du Dataset
Le dataset Advanced IoT Agriculture capture des mesures physiologiques et morphologiques détaillées de plantes cultivées dans deux types de serres (équipée IoT vs traditionnelle) au laboratoire d'agriculture de l'Université de Tikrit. Compilé par Mohammed Ismail Lifta (2023-2024) sous la supervision du Prof. Wisam Dawood Abdullah.

### Informations sur la Collecte de Données

**Lieu et Période :** Laboratoire d'Agriculture, Collège d'Informatique et de Mathématiques, Université de Tikrit, Irak (2023-2024)

**Types de Serres :**
- **Serre IoT :** Plantes surveillées via des capteurs capturant en temps réel la chlorophylle, l'humidité et les données de croissance
- **Serre Traditionnelle :** Métriques enregistrées manuellement suivant des protocoles d'échantillonnage standard

**Échantillonnage :** Lots randomisés (identifiants Random R1-R3) pour assurer une couverture représentative à travers les stades et conditions des plantes

### Colonnes et Attributs

1. **Random** (String)
   - ID du lot d'échantillon (R1, R2, R3)

2. **ACHP** (Float)
   - Teneur moyenne en chlorophylle (pigment photosynthétique)

3. **PHR** (Float)
   - Taux de croissance de la hauteur de la plante

4. **AWWGV** (Float)
   - Poids humide moyen de la croissance végétative

5. **ALAP** (Float)
   - Surface foliaire moyenne par plante

6. **ANPL** (Float)
   - Nombre moyen de feuilles par plante

7. **ARD** (Float)
   - Diamètre moyen des racines

8. **ADWR** (Float)
   - Poids sec moyen des racines

9. **PDMVG** (Float)
   - % de matière sèche dans la croissance végétative

10. **ARL** (Float)
    - Longueur moyenne des racines

11. **AWWR** (Float)
    - Poids humide moyen des racines

12. **ADWV** (Float)
    - Poids sec moyen des parties végétatives

13. **PDMRG** (Float)
    - % de matière sèche dans la croissance des racines

14. **Class** (Catégoriel)
    - Étiquette du groupe expérimental (SA, SB, SC, TA, TB, TC)

### Cas d'Utilisation Potentiels
- **Recherche environnementale et agricole :** Analyser comment les interventions IoT pilotées par capteurs impactent la santé des plantes par rapport aux méthodes traditionnelles
- **Apprentissage automatique :** Construire des modèles de classification pour prédire la classe de traitement à partir de caractéristiques physiologiques
- **Études de physiologie végétale :** Corréler la teneur en chlorophylle avec la surface foliaire, l'architecture racinaire et les pourcentages de matière sèche

**Licence :** CC BY-ND  
**Contact :** Prof. Wisam Dawood Abdullah (wisamdawood@tu.edu.iq)

---

## 6. 📡 IoT Agriculture 2024

**Fichier de données :** `IoTProcessed_Data.csv`  
**Nombre d'entrées :** 37,923 lignes  
**Nombre de colonnes :** 13

### Description du Dataset
Recherche de thèse de master menée par l'étudiant Mohammed Ismail Lifta (2023-2024) au Département d'Informatique, Université de Tikrit, Irak. Les données ont été collectées à partir d'une serre intelligemment équipée. L'étude a été supervisée par le Professeur Assistant Wissam Dawood Abdullah, Directeur de la Cisco Networking Academy à l'Université de Tikrit.

L'étude comprenait la construction d'une serre intelligente équipée de technologies avancées pour surveiller et contrôler les conditions environnementales. Elle comprenait une application qui relie les données à Google Sheets pour la surveillance et le contrôle à distance, fournissant une plateforme efficace pour une gestion efficace de la serre.

### Colonnes et Attributs

1. **date** (datetime64)
   - Date et heure auxquelles les mesures ont été enregistrées

2. **temperature** (int64)
   - Température enregistrée en degrés Celsius

3. **humidity** (int64)
   - Pourcentage d'humidité dans l'environnement

4. **water_level** (int64)
   - Niveau d'eau en pourcentage

5. **N** (int64)
   - Niveau d'azote dans le sol
   - Échelle de 0 à 255

6. **P** (int64)
   - Niveau de phosphore dans le sol
   - Échelle de 0 à 255

7. **K** (int64)
   - Niveau de potassium dans le sol
   - Échelle de 0 à 255

8. **Fan_actuator_OFF** (float64)
   - Indicateur pour l'actionneur de ventilateur s'il est éteint (0 ou 1)

9. **Fan_actuator_ON** (float64)
   - Indicateur pour l'actionneur de ventilateur s'il est allumé (0 ou 1)

10. **Watering_plant_pump_OFF** (float64)
    - Indicateur pour la pompe d'arrosage des plantes si elle est éteinte (0 ou 1)

11. **Watering_plant_pump_ON** (float64)
    - Indicateur pour la pompe d'arrosage des plantes si elle est allumée (0 ou 1)

12. **Water_pump_actuator_OFF** (float64)
    - Indicateur pour l'actionneur de la pompe à eau s'il est éteint (0 ou 1)

13. **Water_pump_actuator_ON** (float64)
    - Indicateur pour l'actionneur de la pompe à eau s'il est allumé (0 ou 1)

### Détails Additionnels
- Les données ont été nettoyées en supprimant les lignes dupliquées et les valeurs manquantes
- Les colonnes catégorielles ont été encodées en utilisant la technique One-Hot Encoding pour faciliter l'utilisation des données en apprentissage automatique
- Le fichier est prêt pour l'analyse et la modélisation en utilisant des outils d'apprentissage automatique

**Licence :** CC BY-ND  
**Contact :** Prof. Wisam Dawood Abdullah (wisamdawood@tu.edu.iq)

---

## 7. 💧 Smart Agriculture Dataset

**Fichier de données :** `cropdata_updated.csv`  
**Nombre d'entrées :** 16,411 enregistrements

### Description du Dataset
Ce dataset contient 16,411 enregistrements de données liées aux cultures, se concentrant sur les conditions environnementales clés et leur effet sur les stades de croissance des cultures. Le dataset inclut des informations telles que le type de sol, le stade de semis, l'indice d'humidité (MOI), la température et l'humidité. Chaque ligne représente un échantillon de culture unique avec ses facteurs environnementaux correspondants et un résultat de sortie, qui peut être utilisé pour des modèles d'apprentissage automatique pour prédire la santé des cultures ou les besoins en irrigation.

### Colonnes et Attributs

1. **crop ID** (catégoriel)
   - Identifiant unique pour la culture

2. **soil_type** (catégoriel)
   - Type de sol pour la culture
   - Exemples : Black Soil (Sol noir), Red Soil (Sol rouge)

3. **Seedling Stage** (catégoriel)
   - Stade de croissance de la culture
   - Exemple : Germination

4. **MOI - Moisture Index** (integer)
   - Indice d'humidité
   - Teneur en humidité du sol au moment de la collecte des données

5. **temp - Temperature** (integer)
   - Température ambiante en degrés Celsius

6. **humidity** (float)
   - Humidité relative en pourcentage

7. **result** (binaire) - **VARIABLE CIBLE**
   - Indique si la culture nécessite une irrigation ou non
   - Valeurs : 1 = oui, 0 = non

### Cas d'Utilisation Potentiels
- **Gestion de l'irrigation :** Ce dataset peut être utilisé pour entraîner des modèles qui prédisent le besoin d'irrigation en fonction des conditions environnementales
- **Recherche agricole :** Les chercheurs peuvent utiliser ce dataset pour étudier l'influence des conditions de sol et de climat sur les stades de croissance des cultures
- **Applications d'agriculture intelligente :** Peut être intégré dans des systèmes IoT pour automatiser les décisions d'irrigation basées sur des données en temps réel

---

## 8. 🚜 Smart Farming Sensor Data for Yield Prediction

**Fichier de données :** `Smart_Farming_Crop_Yield_2024.csv`  
**Nombre de fermes :** 500 fermes

### Description du Dataset
Ce dataset simule des opérations agricoles intelligentes du monde réel alimentées par des capteurs IoT et des données satellitaires. Il capture les variables environnementales et opérationnelles qui affectent le rendement des cultures dans 500 fermes situées dans des régions comme l'Inde, les États-Unis et l'Afrique.

Conçu pour refléter les systèmes agritech modernes, les données sont idéales pour :
- Modélisation prédictive utilisant ML/AI
- Analyse de séries temporelles
- Optimisation basée sur les capteurs
- Visualisations de données environnementales
- Analyse de la santé des cultures

### Colonnes et Attributs

1. **farm_id** (String)
   - ID unique pour chaque ferme intelligente
   - Format : FARM0001, FARM0002, etc.

2. **region** (catégoriel)
   - Région géographique
   - Exemples : North India (Inde du Nord), South USA (Sud des États-Unis)

3. **crop_type** (catégoriel)
   - Culture cultivée
   - Valeurs : Wheat (Blé), Rice (Riz), Maize (Maïs), Cotton (Coton), Soybean (Soja)

4. **soil_moisture_%** (float)
   - Teneur en humidité du sol en pourcentage

5. **soil_pH** (float)
   - Niveau de pH du sol
   - Plage typique : 5.5 à 7.5

6. **temperature_C** (float)
   - Température moyenne pendant le cycle de culture (en °C)

7. **rainfall_mm** (float)
   - Précipitations totales reçues en millimètres

8. **humidity_%** (float)
   - Niveau d'humidité moyen en pourcentage

9. **sunlight_hours** (float)
   - Heures d'ensoleillement moyennes reçues par jour

10. **irrigation_type** (catégoriel)
    - Type d'irrigation
    - Valeurs : Drip (Goutte-à-goutte), Sprinkler (Aspersion), Manual (Manuel), None (Aucune)

11. **fertilizer_type** (catégoriel)
    - Engrais utilisé
    - Valeurs : Organic (Organique), Inorganic (Inorganique), Mixed (Mixte)

12. **pesticide_usage_ml** (float)
    - Utilisation quotidienne de pesticides en millilitres

13. **sowing_date** (date)
    - Date à laquelle la culture a été semée

14. **harvest_date** (date)
    - Date à laquelle la culture a été récoltée

15. **total_days** (integer)
    - Durée de croissance de la culture (harvest - sowing)

16. **yield_kg_per_hectare** (float) - **🌾 VARIABLE CIBLE**
    - Rendement de la culture en kilogrammes par hectare

17. **sensor_id** (String)
    - ID du capteur IoT rapportant les données

18. **timestamp** (datetime)
    - Horodatage aléatoire dans le cycle lorsque l'instantané de données a été enregistré

19. **latitude** (float)
    - Latitude de l'emplacement de la ferme
    - Plage : 10.0 à 35.0

20. **longitude** (float)
    - Longitude de l'emplacement de la ferme
    - Plage : 70.0 à 90.0

21. **NDVI_index** (float)
    - Indice de Végétation par Différence Normalisée (Normalized Difference Vegetation Index)
    - Plage : 0.3 à 0.9
    - Indicateur de la santé de la végétation

22. **crop_disease_status** (catégoriel)
    - État de maladie de la culture
    - Valeurs : None (Aucune), Mild (Légère), Moderate (Modérée), Severe (Sévère)

### Idéal Pour
- Modèles ML supervisés (régression, classification)
- Prédiction et optimisation du rendement
- Systèmes d'aide à la décision agricole
- Analyse de stratégie d'irrigation intelligente
- Visualisation de données d'efficacité agricole régionale

---

## 9. ⚙️ Smart Agricultural Production Optimizing Engine

**Fichier de données :** `Crop_recommendation.csv`

### Description du Dataset
Ce dataset fait partie d'un moteur d'optimisation de la production agricole intelligente. Il vise à améliorer l'efficacité en agriculture - produire plus avec moins. L'industrie sera transformée par la science des données et l'intelligence artificielle. Les agriculteurs auront les outils pour tirer le meilleur parti de chaque hectare.

Jusqu'à maintenant, des facteurs tels que le changement climatique, la croissance démographique et les préoccupations en matière de sécurité alimentaire ont propulsé l'industrie à rechercher des approches plus innovantes pour améliorer le rendement des cultures.

### Colonnes et Attributs

1. **N** (numérique)
   - Azote (Nitrogen)
   - Ratio de la teneur en azote dans le sol

2. **P** (numérique)
   - Phosphore (Phosphorous)
   - Ratio de la teneur en phosphore dans le sol

3. **K** (numérique)
   - Potassium (Potassium)
   - Ratio de la teneur en potassium dans le sol

4. **temperature** (numérique)
   - Température moyenne du sol
   - Les températures du sol pour la bioactivité varient de 50 à 75°F

5. **humidity** (numérique)
   - Humidité relative en pourcentage

6. **ph** (numérique)
   - Valeur du pH
   - Échelle utilisée pour identifier la nature acide ou basique
   - Nature acide : pH < 7
   - Neutre : pH = 7
   - Nature basique : pH > 7

7. **rainfall** (numérique)
   - Précipitations en millimètres

8. **label** (catégoriel) - **VARIABLE CIBLE**
   - Types de cultures
   - Valeurs : Rice (Riz), Maize (Maïs), Chickpea (Pois chiche), Kidney beans (Haricots rouges), Pigeonpeas (Pois d'Angole), Mothbeans, Mungbean (Haricot mungo), Blackgram, Lentil (Lentille), Pomegranate (Grenade), Banana (Banane), Mango (Mangue), Grapes (Raisin), Watermelon (Pastèque), Muskmelon (Melon), Apple (Pomme), Orange, Papaya (Papaye), Coconut (Noix de coco), Cotton (Coton), Jute, Coffee (Café)

### Objectif
Permettre aux agriculteurs d'obtenir des recommandations basées sur l'IA pour choisir les cultures les plus appropriées en fonction des conditions du sol, du climat et d'autres facteurs environnementaux, contribuant ainsi à une agriculture durable et efficace.

---

## 📝 Notes Importantes

### Utilisation des Datasets
- Tous ces datasets peuvent être utilisés pour des tâches d'apprentissage automatique incluant la classification, la régression, et l'analyse prédictive
- Idéaux pour la recherche en agriculture de précision, IoT agricole, et optimisation des rendements
- Peuvent être combinés pour créer des modèles plus robustes

### Licences
- La plupart des datasets de l'Université de Tikrit sont sous licence CC BY-ND
- Attribution appropriée requise lors de l'utilisation dans toute publication
- Ne pas modifier les datasets originaux sans autorisation

### Applications Potentielles
- Systèmes d'aide à la décision pour les agriculteurs
- Prédiction de rendements de cultures
- Recommandation de cultures basée sur les conditions environnementales
- Gestion intelligente de l'irrigation
- Optimisation de l'utilisation des engrais et pesticides
- Surveillance de la santé des plantes en temps réel
- Agriculture de précision et agriculture durable

---

**Document créé le :** 13 novembre 2025  
**Nombre total de datasets :** 9  
**Nombre total de fichiers de données :** 12+

---

*Ce document récapitulatif a été créé pour faciliter la compréhension et l'utilisation de l'ensemble des datasets agricoles disponibles dans la collection.*
