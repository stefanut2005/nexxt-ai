# FraudWatch

**FraudWatch** is a project that detects fraudulent transactions in real time.  
It combines **machine learning**, **databases**, and a **web dashboard** to show and analyze suspicious activity.

---

## Overview

FraudWatch uses both a **Random Forest** and a **Neural Network** model to identify fraud patterns.  
Predictions are stored in a **PostgreSQL** database and connected to an **AI assistant (Claude Sonnet 4 via AWS Bedrock)** with a **MCP Server** for automatic query analysis and data insights.  

A special **“Live Transactions”** page generates random transactions to test the models live, showing their risk scores and fraud predictions instantly.

---


## Components

### 1. Data & Model Training
- Data preprocessing and feature selection with **pandas** and **scikit-learn**  
- Trained **Random Forest** and **Neural Network** models  
- Saved models as `.pkl` and `.h5`  
- Results stored in a **PostgreSQL** database  

### 2. Database & API Layer
- **FastAPI** backend for queries and model predictions  
- MCP server connects database data to the LLM  
- Supports SQL execution and CSV imports  

### 3. AI Agent
- Predicts fraud using the trained models  
- Exposes endpoints like `/predict_fraud` and `/generate_and_predict_transaction`  
- Generates random (synthetic) transactions for testing  

### 4. Web Dashboard
- Built with **.NET 8 (Razor Pages)**  
- Displays fraud detection results in real time  
- “Live Transactions” tab shows continuously generated transactions with live model predictions  

---

## Technologies Used

| Area | Tools |
|------|-------|
| Machine Learning | Python 3, scikit-learn, TensorFlow/Keras |
| Database | PostgreSQL, SQLAlchemy |
| API | FastAPI, MCP Server |
| AI Analysis | AWS Bedrock (Claude Sonnet 4) |
| Frontend | .NET 8, C#, Razor Pages |
| Deployment | Docker, docker-compose |

---

## Main Features
- Fraud detection with ML models  
- PostgreSQL integration for storing and analyzing results  
- AI assistant for SQL and data insights  
- Real-time synthetic transaction simulation  
- Interactive web dashboard for visualization  