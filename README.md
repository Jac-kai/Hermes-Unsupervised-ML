# Hermes-Unsupervised-ML

A modular terminal-based unsupervised machine learning system for clustering, analysis, evaluation, and visualization.

Hermes-Unsupervised-ML is a Python-based unsupervised machine learning project designed as an interactive terminal system.  
It provides a structured workflow for loading datasets, selecting features, running clustering models, analyzing clustering results, visualizing outputs, and saving trained workflows.

This project was built not only to practice unsupervised machine learning, but also to demonstrate modular project architecture, reusable design, and engineering-oriented workflow development.

---

## Project Overview

Hermes is a menu-driven unsupervised learning framework that separates clustering workflows into organized modules.  
The project emphasizes both practical clustering usage and clean system design.

It includes:

- dataset loading and preprocessing workflow
- feature selection for clustering
- unsupervised model training
- clustering analysis and visualization
- modular engine / mission / model / config architecture
- result saving and workflow logging

---

## Features

- Interactive terminal menu system
- Load and manage machine learning datasets
- Select clustering feature column(s)
- Automatically support structured workflow for unsupervised learning tasks
- Support clustering model training and result inspection
- Modular engine-based workflow design
- Built-in clustering evaluation and plotting functions
- Logging support for workflow tracking and debugging

---

## Supported Workflows

Hermes focuses on unsupervised machine learning workflows, especially clustering-based analysis.

Depending on the selected model and task design, Hermes is intended to support workflows such as:

- clustering model training
- cluster summary inspection
- clustering evaluation
- cluster visualization
- report-style result review
- modular clustering pipeline experimentation

---

## Evaluation and Visualization

Depending on the selected clustering model and workflow, Hermes supports analysis and visualization features such as:

- clustering result summaries
- cluster distribution analysis
- clustering-related visualizations
- evaluation-oriented inspection workflows
- saved analysis outputs and reports

These tools help users better understand cluster structure, inspect grouping behavior, and interpret unsupervised learning results more clearly.

---

## Project Structure

```bash
Hermes-Unsupervised-ML/
│
├── ML_ClusterBox/
├── ML_UnSup_BaseConfigBox/
├── ML_UnSup_MissionBox/
│
├── Menu_Config.py
├── Menu_Helper_Decorator.py
├── Hermes_Logging.py
├── Hermus_ML_UnSup_Engine.py
├── Hermus_ML_UnSup_Main.py
├── Hermus_Menu1.py
├── Hermus_Menu2.py
├── Hermus_Menu3.py
├── Hermes_Model_Menu_Helper.py
└── .gitignore