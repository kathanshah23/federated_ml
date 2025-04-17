BITE401L Network and Information Systems CLASS COURSE PROJECT:
Project Done Under Guidance of Prof. Aswani Kumar Cherukuri

Federated Machine Learning for Network Traffic Analysis
🧠 Project Overview
This project aims to demonstrate a federated learning approach for analyzing and detecting anomalies in network traffic data. Each client (representing an organization or region) trains a local machine learning model using its own data. The server then aggregates these models without accessing the raw data, preserving privacy and enabling global learning.

📌 Objectives
Perform local training of ML models using network traffic datasets.

Implement federated averaging on the server to produce a global model.

Include preprocessing techniques such as PCA (Principal Component Analysis) and SVM (Support Vector Machine)-based feature selection.

Evaluate model performance on metrics like accuracy, precision, recall, and F1-score.



🛠️ Project Structure
bash
Copy
Edit
├── client/
│   ├── client_federated_ml.py       # Client-side training and preprocessing
│   ├── dataset1.xlsx
│   ├── dataset2.xlsx
│   └── dataset3.xlsx
├── server/
│   └── server_aggregator.py         # Server-side model aggregation
├── models/
│   └── global_model.h5              # Final federated model
├── plots/
│   └── performance_graphs.png       # Evaluation graphs
├── README.md

📈 Methodology
Data Preprocessing:

Loaded Excel files

Standardized numeric features

Applied PCA for dimensionality reduction

Used SVM to select top features

Local Training:

Each client trains a neural network on its processed data

Federated Aggregation:

The server collects model weights from clients

Performs Federated Averaging to update the global model

Evaluation:

Evaluated on classification metrics and plotted performance graphs

