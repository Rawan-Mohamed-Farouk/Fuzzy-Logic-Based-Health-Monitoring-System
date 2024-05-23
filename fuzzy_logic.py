import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import skfuzzy.control as ctrl

# Define input variables and ranges (blood pressure, temperature)
blood_pressure = ctrl.Antecedent(np.arange(80, 181, 1), 'blood_pressure')
temperature = ctrl.Antecedent(np.arange(98, 107, 1), 'temperature')
health_condition = ctrl.Consequent(np.arange(0, 1.01, 0.1), 'health_condition')

# Define membership functions
blood_pressure['low'] = fuzz.trimf(blood_pressure.universe, [80, 95, 120])
blood_pressure['normal'] = fuzz.trimf(blood_pressure.universe, [90, 115, 140])
blood_pressure['high_BP_stage_1'] = fuzz.trimf(blood_pressure.universe, [100, 130, 160])
blood_pressure['high_BP_stage_2'] = fuzz.trimf(blood_pressure.universe, [110, 140, 170])
blood_pressure['emergency'] = fuzz.trimf(blood_pressure.universe, [130, 155, 180])

temperature['low'] = fuzz.trimf(temperature.universe, [98, 98, 100])
temperature['normal'] = fuzz.trimf(temperature.universe, [98, 100, 102])
temperature['high_1'] = fuzz.trimf(temperature.universe, [100, 102, 104])
temperature['high_2'] = fuzz.trimf(temperature.universe, [102, 104, 106])
temperature['emergency'] = fuzz.trimf(temperature.universe, [104, 106, 106])

health_condition['good'] = fuzz.trimf(health_condition.universe, [0, 0, 0.5])
health_condition['normal'] = fuzz.trimf(health_condition.universe, [0, 0.5, 1])
health_condition['worst'] = fuzz.trimf(health_condition.universe, [0.5, 1, 1])

# Define fuzzy rules
rule1 = ctrl.Rule(
    antecedent=(
        (blood_pressure['low'] & temperature['low']) |
        (blood_pressure['normal'] & temperature['normal']) |
        (blood_pressure['high_BP_stage_1'] & temperature['high_1'])
    ),
    consequent=health_condition['good'])

rule2 = ctrl.Rule(
    antecedent=(
        (age['young'] & sugar_level['normal']) |
        (age['middle age'] & sugar_level['elevated']) |
        (age['elder'] & sugar_level['high'])
    ),
    consequent=health_condition['normal'])

rule3 = ctrl.Rule(
    antecedent=(
        (blood_pressure['high_BP_stage_2'] & temperature['high_2']) |
        (age['elder'] & sugar_level['high']) |
        (temperature['emergency'] & sugar_level['elevated'])
    ),
    consequent=health_condition['worst'])

# Create control system and simulate
health_condition_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
health_condition_simulation = ctrl.ControlSystemSimulation(health_condition_ctrl)

# Input values (to be modified as per actual implementation)
health_condition_simulation.input['blood_pressure'] = 120
health_condition_simulation.input['temperature'] = 100

# Compute fuzzy logic
health_condition_simulation.compute()

# Defuzzified output
health_condition_output_defuzzified = health_condition_simulation.output['health_condition']
print("Defuzzified Output:", health_condition_output_defuzzified)

# Visualize membership functions (optional)
blood_pressure.view()
temperature.view()
health_condition.view()
plt.show()

