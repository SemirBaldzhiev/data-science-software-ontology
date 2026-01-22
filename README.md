# DSSO: Data Science Software Ontology

[![Ontology](https://img.shields.io/badge/Ontology-OWL2_DL-blue)](https://www.w3.org/TR/owl2-overview/)
[![Language](https://img.shields.io/badge/Language-Python_3.10+-yellow)](https://www.python.org/)
[![Library](https://img.shields.io/badge/Library-Owlready2-green)](https://owlready2.readthedocs.io/en/v0.49/)
[![Reasoner](https://img.shields.io/badge/Reasoner-HermiT-red)](http://www.hermit-reasoner.com/)

## 📌 Project Overview
**DSSO (Data Science Software Ontology)** is a semantic model designed to classify, query, and manage dependencies within the modern Data Science and Machine Learning ecosystem.

Unlike simple databases, DSSO uses **Description Logics (OWL 2 DL)** to model complex relationships between:
* **Software** (Libraries, Platforms, Notebooks)
* **Algorithms** (Deep Learning, Optimization, Clustering)
* **Hardware** (GPUs, TPUs)
* **Licenses** (Open Source vs. Proprietary)

The project leverages the **HermiT Reasoner** to automatically classify software based on its properties (e.g., *inference that a tool implementing a Transformer architecture is Deep Learning Software*).

---

## 🚀 Key Features

### 1. Automated Inference (Reasoning)
The ontology defines logical rules to classify instances automatically without manual labeling:
* **DeepLearningSoftware:** Inferred if a tool implements an algorithm using a Neural Network architecture.
* **ApacheProject:** Inferred if the developer is the Apache Foundation.
* **PurePythonTool:** Inferred if the tool is written exclusively in Python.

### 2. Dependency Management
* Uses **Transitive Properties** (`depends_on`) to trace full dependency chains (e.g., Tool A -> Tool B -> Tool C).
* Uses **Symmetric Properties** (`interoperates_with`) to automatically link compatible tools.

---

## 📂 Ontology Structure

| Domain | Key Classes |
| :--- | :--- |
| **Software** | `MLLibrary`, `CloudPlatform`, `NotebookEnvironment`, `DataVisualizationTool` |
| **Algorithms** | `SupervisedLearning`, `DeepLearningMethod`, `OptimizationAlgorithm` |
| **Architecture** | `NeuralNetwork`, `Transformer`, `Convolutional`, `Recurrent` |
| **Hardware** | `HardwareAccelerator` (GPU, TPU), `InstructionSet` |
| **Legal** | `OpenSourceLicense`, `ProprietaryLicense` |

---

## 🛠️ Installation & Requirements

### Prerequisites
* Python 3.8+
* Protege

### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/SeimirBaldzhiev/data-science-software-ontology.git]
    cd data-science-software-ontology
    ```

2.  Install dependencies:
    ```bash
    pip install owlready2
    ```

---

## 💻 Usage

### 1. Generating the Ontology
Run the main Python script to generate the ontology structure and populate it with individuals:

```bash
python ontology_builder.py
```

### 2. Open the generated .owl file with Protege to use it or use a web visualizer like WebVOWL