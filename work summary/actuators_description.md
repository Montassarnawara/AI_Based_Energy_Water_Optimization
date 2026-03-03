# Actuators and Control System Description

## 1. Water Pump – Irrigation
- Purpose: irrigate crops with a precise volume of water.
- Control Type: ON/OFF or flow‑based modulation.
- Model E Inputs: pump_on, pump_volume (L), pump_duration (s).
- Safety: max runtime, water level check, leak detection.

## 2. Nutrient/Fertilizer Pump
- Purpose: inject nutrient solution.
- Control Type: pulse‑based dosing or timed ON.
- Model E Inputs: fert_dose (ml), fert_mode.
- Safety: EC/pH limits, daily maximum.

## 3. Grow Lights (LED)
- Purpose: regulate photoperiod and light intensity.
- Control Type: PWM intensity + schedule ON/OFF.
- Model E Inputs: light_level (0–100%), light_schedule.
- Safety: thermal limit, power consumption cap.

## 4. Temperature Controller (Heating/Cooling)
- Purpose: maintain ideal temperature.
- Control Type: PID or thermostat ON/OFF.
- Model E Inputs: target_T, heating_on, cooling_on, fan_speed.
- Safety: min/max temperature, sensor failure protection.

## 5. Motors (Windows, Valves, Mechanisms)
- Purpose: open/close vents and regulate flow.
- Control Type: servo/stepper positioning.
- Model E Inputs: valve_position, window_open_percent.
- Safety: end‑stop sensors, emergency stop.

## 6. Ventilation Fans
- Purpose: control humidity, temperature, CO2.
- Control Type: PWM speed control.
- Model E Inputs: fan_speed (0–100%).
- Safety: temperature‑linked limits.

## 7. Motorized Shade/Curtains
- Purpose: reduce excessive sun/heat.
- Control Type: position control.
- Model E Inputs: shade_position (0–100%).
- Safety: wind/temperature interaction rules.

## 8. Precision Valves (Solenoid/Proportional)
- Purpose: zone irrigation control.
- Model E Inputs: zone_valve[i]_open.
- Safety: over‑watering protection.

## 9. Heating/Cooling Units
- Purpose: climate maintenance.
- Control Type: ON/OFF or modulated.
- Model E Inputs: heater_on, cooler_on.
- Safety: temp bounds.

## 10. UV Sterilization Lamp (Optional)
- Purpose: disease control.
- Model E Inputs: uv_on.
- Safety: strict interlocks, human presence lockout.

## 11. Solar/Battery Power Manager
- Purpose: energy optimization.
- Model E Inputs: energy_budget_remaining, use_solar_first.
- Safety: battery protection.

## 12. Alarm/Notification Module
- Purpose: alert operator.
- Model E Inputs: alert_level, alert_message.
- Safety: redundancy.

## Control Logic Summary
Model E reads predictions from Models A–D and computes optimal actuator commands considering constraints: crop health, water/energy budgets, environmental conditions, and safety limits. After each action, sensors verify the effect, triggering corrections or safety shutdowns as needed.
