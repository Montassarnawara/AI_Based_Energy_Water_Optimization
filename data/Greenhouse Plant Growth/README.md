About Dataset
The Advanced IoT Agriculture Dataset captures detailed physiological and morphological measurements of plants grown under two greenhouse settings (IoT‑enabled vs. traditional) at Tikrit University’s Agriculture Lab. Compiled by Mohammed Ismail Lifta (2023–2024) under the supervision of Prof. Wisam Dawood Abdullah, it comprises 30,000 records spanning 14 variables that quantify chlorophyll levels, growth rates, biomass (wet/dry weight), root metrics, and more, alongside a final categorical Class label.

Description
1. Data Collection
Location & Period: Agriculture Lab, College of Computer Science & Mathematics, Tikrit University, Iraq (2023–2024).
Greenhouse Types:
IoT Greenhouse: Plants monitored via sensors capturing real‑time chlorophyll, moisture, and growth data.
Traditional Greenhouse: Manually recorded metrics following standard sampling protocols.
Sampling: Randomized batches (Random identifiers R1–R3) to ensure representative coverage across plant stages and conditions.
2. Dataset Structure
Column	Description	Type
Random	Sample batch ID (e.g., R1, R2, R3)	String
ACHP	Average chlorophyll content (photosynthetic pigment)	Float
PHR	Plant height growth rate	Float
AWWGV	Average wet weight of vegetative growth	Float
ALAP	Average leaf area per plant	Float
ANPL	Average number of leaves per plant	Float
ARD	Average root diameter	Float
ADWR	Average dry weight of roots	Float
PDMVG	% dry matter in vegetative growth	Float
ARL	Average root length	Float
AWWR	Average wet weight of roots	Float
ADWV	Average dry weight of vegetative parts	Float
PDMRG	% dry matter in root growth	Float
Class	Experimental group label (SA, SB, SC, TA, TB, TC)	Categorical
3. Potential Use Cases
Environmental & Agricultural Research: Analyze how sensor‑driven IoT interventions impact plant health vs. traditional methods.
Machine Learning: Build classification models to predict treatment class from physiological features or regression models forecasting biomass and growth rates.
Plant Physiology Studies: Correlate chlorophyll content with leaf area, root architecture, and dry‑matter percentages.
4. License & Citation
License: CC BY‑ND. Proper attribution required for any publication or derivative.
Original Source: Lifta, M. I. (2023–2024). Master’s Thesis, Department of Computer Science, Tikrit University—supervised by Prof. Wisam Dawood Abdullah.
OpenML Entry: [OpenML dataset ID 46871] (unavailable via direct fetch).
5. Contact
For questions or collaboration inquiries, please reach out to:

Prof. Wisam Dawood Abdullah
Email: wisamdawood@tu.edu.iq