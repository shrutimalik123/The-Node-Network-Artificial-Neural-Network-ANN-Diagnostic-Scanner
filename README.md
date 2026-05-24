# 🧠 The Node Network: ANN Diagnostic Scanner

An interactive Deep Learning simulation designed to teach **Artificial Neural Networks (ANN)** and **Forward Propagation** from scratch. You play as a Biomedical Engineer calibrating an advanced oncology scanner, routing multi-dimensional tissue biopsy features through activated hidden layer neurons to isolate non-linear tumor signatures.

## 🎓 Learning Objectives

This project focuses on teaching:
* **Artificial Neural Networks (ANN):** Layered computational structures inspired by biological neural networks that map highly complex data patterns.
* **Hidden Layers:** Intermediary processing nodes that perform abstract feature extraction, allowing networks to solve non-linear puzzles that defeat simple linear models.
* **Activation Functions (ReLU & Sigmoid):** Mathematical gates that introduce non-linearity into equations, enabling networks to adapt to complex geometric shapes.
* **Forward Propagation:** The left-to-right calculation pipeline where data values stream through inputs, weights, and hidden synapses to compute a final predictive probability.

---

## ✨ Features

* **Biomedical Diagnostics Scenario:** Translates abstract neural matrix calculations into an intuitive, high-stakes medical scanner framework.
* **Mathematical Transparency:** Traces and displays live layer telemetry, detailing raw calculations before and after activation functions execute.
* **Manual Synapse Alignment:** Allows users to adjust the weights of specific hidden nodes to see exactly how individual connections reshape the network's final output.
* **Pure Python Core:** Built completely with core standard libraries—no bulky machine learning installations or black-box tracking layers required.

---

## 🚀 How to Run the Game

### 1. Prerequisites
You only need **Python 3** installed.

### 2. Setup and Execution
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/node-network-ann.git](https://github.com/YOUR_USERNAME/node-network-ann.git)
    cd node-network-ann
    ```
2.  **Save the Code:** Save the provided script as `neural_scanner.py`.
3.  **Run the Script:**
    ```bash
    python neural_scanner.py
    ```

### 3. Gameplay Instructions
1.  **Examine Lab Benchmarks:** Review the non-linear structure of your training records (where low-low and high-high values are benign, but mixed parameters indicate malignancy).
2.  **Calibrate the Synapses:** Configure the weights for Hidden Neuron 1, Hidden Neuron 2, and the final Output Node.
3.  **Track the Forward Stream:** Observe how the raw numbers change as they pass through the hidden layer's ReLU gates and step up to the final classification point.
4.  **Review the Diagnosis:** Verify whether your specialized network output accurately isolated the malignant test case profile.

---

## 🧠 Code Structure Highlights

### Rectified Linear Unit (ReLU) Activation
The hidden layer maps non-linear geometry by passing continuous data dot products through a strict `max(0, x)` filter, squashing negative signals while keeping positive vectors linear.

```python
# ReLU Implementation: f(x) = max(0, x)
h1_input = (test_patient[0] * w11) + (test_patient[1] * w21)
h1_output = max(0.0, h1_input)
