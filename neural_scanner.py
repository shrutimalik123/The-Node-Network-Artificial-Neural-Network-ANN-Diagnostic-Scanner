import math
import random

def neural_network_medical_game():
    # 1. Scenario: Deep Learning Oncology Scanner
    print("--- 🧠 THE NODE NETWORK: NEURAL NETWORK DIAGNOSTIC SIM 🧠 ---")
    print("Mission: Route biopsy sensor readings through hidden layers to spot malignancies.")
    print("Goal: Use forward propagation to solve a non-linear medical classification problem.")

    # 2. Raw Features: [Cell Density, Protein Marker Level]
    # Complex, non-linear boundary (e.g., if both are high or both are low, it's benign; if mixed, it's malignant)
    print("\n--- 🖥️ LAB REPORT TELEMETRY (TRAINING DATA) ---")
    biopsies = [
        {"features": [1.0, 1.0], "label": 0, "type": "Benign Tissue"},
        {"features": [9.0, 9.0], "label": 0, "type": "Benign Tissue"},
        {"features": [2.0, 8.0], "label": 1, "type": "Malignant Tumor"},
        {"features": [8.0, 2.0], "label": 1, "type": "Malignant Tumor"}
    ]
    for idx, b in enumerate(biopsies):
        print(f"Sample {idx+1}: Density = {b['features'][0]} | Protein = {b['features'][1]} -> [{b['type']}]")

    # 3. Interactive Inputs: Hidden Layer Node Weights
    print("\n--- STEP 1: CONFIGURE THE HIDDEN LAYER NEURONS ---")
    print("We are setting up a Hidden Layer with 2 Neurons to extract complex patterns.")
    try:
        print("\n[Neuron H1 Configuration]")
        w11 = float(input("Weight from Density to H1 (Recommended: 0.5): "))
        w21 = float(input("Weight from Protein to H1 (Recommended: 0.5): "))
        
        print("\n[Neuron H2 Configuration]")
        w12 = float(input("Weight from Density to H2 (Recommended: -0.5): "))
        w22 = float(input("Weight from Protein to H2 (Recommended: 0.5): "))
        
        print("\n[Output Neuron Configuration]")
        w_out1 = float(input("Weight from H1 to Output Node (Recommended: 1.0): "))
        w_out2 = float(input("Weight from H2 to Output Node (Recommended: -1.0): "))
    except ValueError:
        # Default weights that show a working mathematical pathway
        w11, w21, w12, w22 = 0.5, 0.5, -0.5, 0.5
        w_out1, w_out2 = 1.0, -1.0

    # 4. Incoming Critical Patient Case
    test_patient = [2.2, 7.8] # Mixed features -> High likelihood of being Malignant
    print(f"\n--- 🚨 DIAGNOSTIC BAY: URGENT BIOPSY ARRIVED ---")
    print(f"Patient Sample Profile -> Cell Density: {test_patient[0]} | Protein Level: {test_patient[1]}")

    # 5. Forward Propagation Math Engine
    print("\n--- 🔄 COMPUTING FORWARD PROPAGATION THROUGH THE NETWORK ---")
    
    # Hidden Node 1 Calculation (Linear combination + ReLU Activation)
    h1_input = (test_patient[0] * w11) + (test_patient[1] * w21)
    h1_output = max(0.0, h1_input) # ReLU: f(x) = max(0, x)
    
    # Hidden Node 2 Calculation (Linear combination + ReLU Activation)
    h2_input = (test_patient[0] * w12) + (test_patient[1] * w22)
    h2_output = max(0.0, h2_input) # ReLU: f(x) = max(0, x)
    
    print(f"Node H1 Output (ReLU Activated): {h1_output:.2f}")
    print(f"Node H2 Output (ReLU Activated): {h2_output:.2f}")

    # Output Node Calculation (Pass aggregated weights through a Sigmoid Function)
    output_input = (h1_output * w_out1) + (h2_output * w_out2)
    
    # Sigmoid Activation Function to scale output between 0 and 1
    malignancy_probability = 1.0 / (1.0 + math.exp(-output_input))
    
    print(f"\nFinal Network Output Raw Value: {output_input:.2f}")
    print(f"Calculated Malignancy Probability: {malignancy_probability:.2%}")

    # 6. Classification Output Matrix
    threshold = 0.50
    if malignancy_probability >= threshold:
        prediction_label = 1
        prediction_str = "🔴 MALIGNANT TUMOR DETECTED"
    else:
        prediction_label = 0
        prediction_str = "🟢 BENIGN TISSUE PROFILE"

    print(f"Scanner Verdict: {prediction_str}")

    # 7. Ground Truth Validation
    actual_label = 1 # The testing coordinate represents a malignant cell arrangement
    
    if prediction_label == actual_label:
        print("\n🏆 SUCCESS: Your Deep Learning configuration isolated the non-linear tumor signature!")
        print("The oncology unit has mapped an early medical intervention plan.")
    else:
        print("\n💥 DIAGNOSTIC FAILURE: False reading calculated. Adjust your neural synapse weights.")

if __name__ == "__main__":
    neural_network_medical_game()
