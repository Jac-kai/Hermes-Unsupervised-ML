# Hermes-Unsupervised-ML

A modular terminal-based unsupervised machine learning system for clustering, evaluation, and visualization.

Hermes-Unsupervised-ML is a Python-based unsupervised learning project designed as an interactive terminal workflow system.  
It supports dataset loading, clustering feature selection, clustering model execution, result evaluation, visualization, and workflow-oriented experimentation in a structured modular architecture.

This project was built not only to practice unsupervised machine learning, but also to demonstrate reusable system design, menu-driven workflow engineering, and modular Python project organization.

---

## Project Overview

Hermes is a menu-driven unsupervised learning framework focused on clustering workflows.  
It separates data handling, clustering execution, evaluation, and visualization into organized modules so that the overall workflow remains clear, reusable, and extensible.

The system is designed around an engine / mission / model / config architecture and currently supports multiple clustering algorithms for structured experimentation and result inspection.

---

## Project Purpose

This project was developed to strengthen practical understanding of unsupervised machine learning, especially clustering workflows, while also improving modular system design and engineering-oriented Python development.

Hermes emphasizes not only algorithm usage, but also how to organize a machine learning workflow into reusable components for training, evaluation, visualization, and interactive experimentation.

---

## Features

- Interactive terminal menu system
- Load and manage tabular datasets for unsupervised learning workflows
- Select clustering input feature column(s)
- Support structured preprocessing before clustering
- Support multiple clustering models:
  - KMeans
  - DBSCAN
  - AgglomerativeClustering
- Built-in clustering evaluation and summary inspection
- Built-in clustering visualization and diagnostic plotting
- Modular engine-based workflow design
- Logging support for workflow tracking and debugging
- Save and reload trained clustering workflows

---

## Supported Workflows

Hermes is designed to support clustering-centered unsupervised learning workflows such as:

- dataset loading
- clustering input selection
- clustering model training
- clustering result inspection
- evaluation menu workflow
- clustering visualization
- modular clustering pipeline experimentation
- workflow saving and reuse

---

## Supported Clustering Models

Hermes currently supports the following clustering models:

### 1. KMeans
- configurable number of clusters
- centroid initialization control
- repeated initialization support
- elbow-method diagnostic plotting

### 2. DBSCAN
- configurable density radius (`eps`)
- configurable `min_samples`
- selectable distance metric
- noise-aware evaluation workflow
- k-distance diagnostic plotting

### 3. AgglomerativeClustering
- configurable number of clusters
- configurable linkage strategy
- metric selection depending on linkage compatibility
- hierarchical dendrogram visualization

---

## Evaluation and Visualization

Hermes includes evaluation and visualization utilities for clustering analysis, including:

- current model summary
- cluster evaluation summary
- cluster label preview
- clustered data preview
- cluster size summary
- cluster size bar plot
- cluster scatter plot
- cluster PCA plot
- silhouette plot
- cluster profile heatmap
- KMeans elbow plot
- DBSCAN k-distance plot
- Agglomerative dendrogram plot

These tools help users inspect cluster structure, compare grouping behavior, and understand unsupervised learning outputs more clearly.

---

## System Design

Hermes is organized using a modular workflow architecture.  
Its structure separates responsibilities into different layers so that data preparation, clustering execution, evaluation, and visualization remain easier to manage and extend.

### Core architectural ideas
- **Engine layer**: coordinates the full workflow
- **Model layer**: handles algorithm-specific clustering logic
- **Mission layer**: provides evaluation, summary, and plotting tools
- **Config layer**: manages shared preprocessing and pipeline configuration
- **Menu layer**: provides interactive terminal control flow

This design makes the project more maintainable and helps demonstrate engineering-oriented workflow separation beyond basic model fitting.

---

## Project Structure

```bash
Hermes-Unsupervised-ML/
│
├── ML_ClusterBox/               # Model-specific clustering implementations
├── ML_UnSup_BaseConfigBox/      # Shared clustering pipeline / preprocessing config
├── ML_UnSup_MissionBox/         # Evaluation, summary, and visualization logic
│
├── Menu_Config.py               # Menu configuration
├── Menu_Helper_Decorator.py     # CLI input and menu helpers
├── Hermes_Logging.py            # Logging setup
├── Hermus_ML_UnSup_Engine.py    # Main workflow engine
├── Hermus_ML_UnSup_Main.py      # Program entry point
├── Hermus_Menu1.py              # Menu layer 1
├── Hermus_Menu2.py              # Menu layer 2
├── Hermus_Menu3.py              # Menu layer 3
├── Hermes_Model_Menu_Helper.py  # Model parameter menu helpers
└── .gitignore