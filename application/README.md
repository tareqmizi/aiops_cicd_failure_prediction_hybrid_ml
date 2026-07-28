# Predictive AIOps Framework: Application Subsystem & Microservices

[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](./docker-compose.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](./api_core)
[![Node.js](https://img.shields.io/badge/Node.js-18+-339933?logo=node.js&logoColor=white)](./backend)
[![React](https://img.shields.io/badge/React-18.0-61DAFB?logo=react&logoColor=white)](./frontend)

## Executive Overview

The **Application Subsystem** provides a full-stack, microservices-based operational web application for pre-execution CI/CD pipeline failure prediction and automated log diagnostics. Built on a glassmorphic React 18 frontend, an Express.js backend proxy orchestrator, and a Python FastAPI machine learning microservice, this architecture allows software engineering teams to evaluate build failure risk prior to triggering automated CI/CD pipelines.

![AIOps Predictive Dashboard Running on Docker](./app_dashboard_screenshot.png)
*Figure: Operational UI Dashboard running on Docker Compose (`http://localhost:5173`), displaying commit risk score breakdown (XGBoost 84%, BiLSTM 32%, Ensemble 58%), Medium Risk Category alert, dynamic LLM Root Cause Analysis, and live telemetry status.*

---

## 📑 Table of Contents

- [System Subsystems & Microservices Architecture](#system-subsystems--microservices-architecture)
- [Repository Directory Structure](#repository-directory-structure)
- [Required Input Schema & Data Validation](#required-input-schema--data-validation)
- [Installation & Execution Guide](#installation--execution-guide)
  - [Option A: Containerized via Docker Compose (Recommended)](#option-a-containerized-via-docker-compose-recommended)
  - [Option B: Local Native Microservices Execution](#option-b-local-native-microservices-execution)
- [REST API Specifications](#rest-api-specifications)
- [Automated Integration Testing](#automated-integration-testing)
- [Static Frontend Deployment (GitHub Pages)](#static-frontend-deployment-github-pages)
- [Future System Roadmap](#future-system-roadmap)
- [Subsystem Navigation & Quick Links](#subsystem-navigation--quick-links)

---

## System Subsystems & Microservices Architecture

The application comprises three operational microservice layers:

```
+-------------------------------------------------------------------+
|                   Frontend Dashboard (React + Vite)                |
|                        http://localhost:5173                      |
|  - Commit List & On-Demand Predictor   - Batch CSV Processing     |
|  - Real-Time Risk Badges & RCA Modal   - Client-Side Predictions Map|
+---------------------------------+---------------------------------+
                                  |
                                  | REST API Requests (JSON)
                                  v
+-------------------------------------------------------------------+
|                  Backend Orchestrator (Node.js/Express)           |
|                        http://localhost:3001                      |
|  - API Request Proxy & Health Checks   - CORS & Environment Config|
+---------------------------------+---------------------------------+
                                  |
                                  | Microservice REST Call
                                  v
+-------------------------------------------------------------------+
|               Inference Microservice (Python FastAPI)              |
|                        http://localhost:8000                      |
|  - Feature Scaler & Normalization     - Failure Log RCA Parser    |
+------------------------+------------------------------------------+
                         |
           +-------------+-------------+
           |                           |
           v                           v
+--------------------+       +--------------------+
|   XGBoost Model    |       |    BiLSTM Model    |
| (Static Features)  |       | (Sequential Build  |
| Weight: w = 0.55   |       |   History Window)  |
+--------------------+       +--------------------+
```

### Subsystem Functionality:

1. **Ensemble Inference Engine (`api_core`)**:
   - **XGBoost Classifier**: Evaluates static repository metrics, author churn, SLOC, and historical committer failure statistics.
   - **BiLSTM Sequence Model**: Evaluates temporal dependencies across sequential build runs ($L=10$).
   - **Risk Classification Policy**: Blends probabilities ($P_{\text{ensemble}} = 0.55 \cdot P_{\text{XGB}} + 0.45 \cdot P_{\text{BiLSTM}}$) into **Low (<40%)**, **Medium (40%-70%)**, and **High (>70%)** risk tiers.

2. **Automated Root Cause Analysis (RCA)**:
   - Inspects build console log output to diagnose stack traces (`ECONNREFUSED`, `AssertionError`, `OutOfMemory`, compilation errors).
   - Generates contextual explanations and step-by-step resolution advice for developers.

3. **Data Ingestion & Feature Validation**:
   - **Schema Validation**: Validates uploaded CSV files for required headers (`gh_project_name`, `git_trigger_commit`, `git_branch`, `tr_status`, `git_diff_src_churn`).
   - **Memory-Efficient Parsing**: Handles quoted text blocks and large multi-gigabyte dataset ingestion.

4. **Execution & Caching**:
   - **On-Demand Single-Commit Evaluation**: Evaluates risk for individual selected commits.
   - **Batch Prediction**: Computes predictions across unpredicted commits concurrently with real-time progress indicators.
   - **Client-Side Caching**: Caches prediction results (`predictionsMap`) to eliminate redundant backend API calls.

---

## Repository Directory Structure

```
application/
├── docker-compose.yml                 # Multi-container orchestration specification
├── sample_test_data.csv               # Baseline 50-row test dataset
├── test_pipeline.py                   # E2E integration test script
├── README.md                          # Subsystem documentation (this file)
│
├── frontend/                          # React + Vite client web application
│   ├── Dockerfile                     # Container definition for frontend
│   ├── index.html                     # HTML entry template
│   ├── package.json                   # Client dependencies and Vite scripts
│   ├── vite.config.js                 # Vite bundler configuration
│   └── src/
│       ├── App.jsx                    # Main dashboard application logic
│       ├── App.css                    # Component-level styles
│       ├── index.css                  # Global glassmorphic styling & design tokens
│       ├── main.jsx                   # React DOM mounting entry point
│       └── sample_commits.json        # Backup static commit dataset
│
├── backend/                           # Node.js / Express proxy orchestrator
│   ├── Dockerfile                     # Container definition for backend
│   ├── package.json                   # Node.js dependencies (express, cors, dotenv)
│   └── server.js                      # API routes, proxying, and health checks
│
└── api_core/                          # Python FastAPI machine learning microservice
    ├── Dockerfile                     # Container definition for ML microservice
    ├── main.py                        # FastAPI endpoints (/health, /predict)
    ├── predictor.py                   # Feature transformation and hybrid inference pipeline
    ├── rca.py                         # Failure log pattern parsing module
    ├── requirements.txt               # Python package dependencies
    ├── aiops_xgboost_model.json       # Serialized XGBoost model weights
    ├── aiops_lstm_model.keras         # Serialized BiLSTM model weights
    └── aiops_lstm_scaler.pkl          # Serialized StandardScaler object
```

---

## Required Input Schema & Data Validation

When uploading custom CSV commit data, the ingestion engine validates the presence of the following feature fields:

| Column Name | Allowed Aliases | Expected Data Type | Description |
| :--- | :--- | :--- | :--- |
| `gh_project_name` | `repo_name` | String | Target repository identifier (e.g., `rspec/rspec-core`) |
| `git_trigger_commit` | `tr_original_commit`, `tr_build_id` | String | Unique commit hash or build ID |
| `git_branch` | - | String | Git branch name (e.g., `master`, `main`) |
| `tr_status` | - | String | Build execution outcome (`passed`, `failed`, `errored`) |
| `git_diff_src_churn` | - | Numeric | Count of modified source code lines |

---

## Installation & Execution Guide

### Option A: Containerized via Docker Compose (Recommended)

1. Clone the repository and navigate to the application directory:
   ```bash
   cd application
   ```

2. Build and launch all microservice containers:
   ```bash
   docker-compose up --build
   ```

3. Access the operational endpoints:
   - **Frontend UI Dashboard**: [`http://localhost:5173`](http://localhost:5173)
   - **Backend API Orchestrator**: [`http://localhost:3001`](http://localhost:3001)
   - **FastAPI ML Microservice**: [`http://localhost:8000/docs`](http://localhost:8000/docs)

4. Stop running containers:
   ```bash
   docker-compose down
   ```

---

### Option B: Local Native Microservices Execution

#### Step 1: Launch Python Machine Learning Microservice (`api_core`)
```bash
cd application/api_core
pip install -r requirements.txt
python main.py
```
*Runs on `http://localhost:8000`*

#### Step 2: Launch Express Backend Proxy Server (`backend`)
```bash
cd application/backend
npm install
npm run dev
```
*Runs on `http://localhost:3001`*

#### Step 3: Launch React Frontend Dashboard (`frontend`)
```bash
cd application/frontend
npm install
npm run dev
```
*Open `http://localhost:5173` in your browser.*

---

## REST API Specifications

### Health Check Endpoint
`GET /health`

**Response Payload**:
```json
{
  "status": "healthy",
  "api_core_connected": true,
  "service": "AIOps Backend Orchestrator"
}
```

### Commit Risk Prediction Endpoint
`POST /api/predict-commit`

**Request Payload**:
```json
{
  "commit": {
    "gh_project_name": "rspec/rspec-core",
    "git_branch": "master",
    "git_trigger_commit": "029e6972",
    "tr_status": "failed",
    "git_diff_src_churn": 240,
    "log_text": "npm ERR! code ECONNREFUSED"
  }
}
```

**Response Payload**:
```json
{
  "commit_hash": "029e6972",
  "project_name": "rspec/rspec-core",
  "predictions": {
    "xgboost_probability": 0.745,
    "lstm_probability": 0.712,
    "ensemble_score": 0.73,
    "risk_level": "High"
  },
  "rca": {
    "error_summary": "Dependency Connection Refused",
    "analysis": "Failed to connect to local port 8081 during npm package resolution.",
    "recommendations": [
      "Check proxy or local dependency server settings.",
      "Verify network connectivity before running CI steps."
    ]
  }
}
```

---

## Automated Integration Testing

To verify cross-service communication, payload parsing, and prediction generation, run the automated integration test script:

```bash
cd application
python test_pipeline.py
```

---

## Static Frontend Deployment (GitHub Pages)

To publish the static frontend dashboard to GitHub Pages:

1. Navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```

2. Install the `gh-pages` helper package:
   ```bash
   npm install --save-dev gh-pages
   ```

3. Add `homepage` and deployment scripts in `frontend/package.json`:
   ```json
   "homepage": "https://<username>.github.io/<repository-name>",
   "scripts": {
     "predeploy": "npm run build",
     "deploy": "gh-pages -d dist"
   }
   ```

4. Execute deployment:
   ```bash
   npm run deploy
   ```

---

## Future System Roadmap

1. 🔗 **GitHub App & Webhook Ingestion**: Intercept `push` and `pull_request` events to run automated predictions on active PRs.
2. 🔔 **Slack & Team Notifications**: Automatically dispatch alerts when a commit is flagged with **High Risk (>70%)**.
3. 🗄️ **Persistent Analytics Store**: Connect PostgreSQL / MongoDB to store historical predictions and committer reliability statistics over time.

---

## Subsystem Navigation & Quick Links

- 🏠 **[Root Project README](../README.md)**: System portal, mathematical formulations, and project overview.
- 🔬 **[Model Training Subsystem README](../model_training/README.md)**: Notebook execution guide, hardware specs, and dataset placement.
- 🧪 **[Sample Test Dataset](./sample_test_data.csv)**: 50-row test dataset for live execution.
