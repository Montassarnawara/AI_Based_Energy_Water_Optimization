Actuators and Control Logic – Detailed Description
=================================================

⚙️ Types de contrôles recommandés
--------------------------------
- ON/OFF — simple et fiable (pompe à eau simple, petits chauffages).
- PWM / modulation analogique — pour LEDs, ventilateurs, pompes à vitesse variable.
- Position control (Stepper / Servo) — pour les aérations motorisées et vannes à ouverture précise.
- PID loops — pour les contrôleurs de température et le maintien précis de l’humidité du sol.
- MPC / constrained optimization (Model E) — pour la planification sur horizon (ex : 24h) avec contraintes d’eau/énergie.

🔐 Sécurité & règles opérationnelles
-----------------------------------
- Hard limits : min/max d’eau par jour, durée maximale de fonctionnement des pompes, intensité lumineuse maximale.
- Fail-safe : si un capteur critique tombe en panne → fallback vers règles simples (horaire fixe).
- Watchdogs : redémarrage automatique du microcontrôleur en cas de freeze.
- Interlocks : empêcher heater + fan d’être à 100% simultanément si dangereux.
- Logging & rollback : journalisation continue et possibilité de retour arrière.

📎 Exemple (à utiliser dans papier IEEE)
----------------------------------------
"The system controls five primary actuator types: irrigation pumps, nutrient dosing pumps, LED grow lights, environmental temperature controllers (heaters/coolers), and motorized valves/windows. Additional actuators include ventilation fans, motorized shading panels, precision valves, and a power management module for solar/battery integration. Each actuator supports either ON/OFF or continuous control (PWM/position/PID). The Meta-Optimizer (Model E) receives predicted health scores, yield forecasts, crop suitability, and local actuator recommendations from Models A–D, and computes constrained optimal commands (pump volume, valve positions, fan speed, light intensity) that minimize water and energy consumption while preserving crop yield. Safety measures—hard limits, sensor sanity checks, and fail-safe fallbacks—are enforced at the gateway to prevent unsafe operations."

📊 Tableau récapitulatif – Capteurs / Actionneurs → Entrée Model E → Type contrôle → Sécurité
--------------------------------------------------------------------------------------------

Capteur / Actionneur | Entrée Model E                       | Type de contrôle     | Sécurité
---------------------|--------------------------------------|-----------------------|-----------------------------------------------
Soil moisture sensor | Soil_moisture_level                  | PID / ON-OFF          | Hard limits + fail-safe
Temp & humidity      | Env_temperature, Env_humidity        | PID / PWM             | Watchdog + interlocks
Water pump           | Irrigation_need (from Models A–D)    | ON/OFF / PWM          | Max runtime + dry-run protection
Nutrient pump        | Nutrient_recommendation (Model B/E)  | ON/OFF / PWM          | Max dosing limit
LED lights           | Light_intensity_target               | PWM                   | Max intensity + heat interlock
Ventilation fan      | Ventilation_required                 | PWM                   | Heater/fan interlock
Motorized vents      | Airflow_target                       | Position control      | Position limits
Heater / Cooler      | Temperature_setpoint                 | PID                   | Overheat/overcool protection
Shading panels       | Radiation_target                     | Servo/Stepper         | Position limits
Solar power module   | Energy_budget                        | ON/OFF + feedback     | Battery safety

