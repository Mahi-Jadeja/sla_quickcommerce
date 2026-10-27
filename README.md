# ⚡ Quick-Commerce SLA Risk Management MLOps Platform

![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue?style=for-the-badge&logo=github)
![Docker](https://img.shields.io/badge/Containerization-Docker%20Compose-2496ED?style=for-the-badge&logo=docker)
![AWS S3](https://img.shields.io/badge/Storage-AWS%20S3-FF9900?style=for-the-badge&logo=amazonaws)
![AWS EC2](https://img.shields.io/badge/Compute-AWS%20EC2-FF9900?style=for-the-badge&logo=amazonaws)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2?style=for-the-badge&logo=mlflow)
![DVC](https://img.shields.io/badge/Versioning-DVC-945DD6?style=for-the-badge)

## Project overview

An end-to-end MLOps platform that predicts **Service Level Agreement (SLA)
breaches** for quick-commerce deliveries. The classifier estimates whether an
order will exceed the 30-minute delivery threshold (`will_breach_sla = 1`)
using dispatch features such as rider details, weather, traffic, and
Haversine delivery distance.

The project includes DVC artifact tracking, MLflow experiment tracking,
automated tests, Dockerized FastAPI and Streamlit services, and runtime model
retrieval from S3. The model binary is not baked into the API image.

For the full system diagrams and lifecycle, see
[architecture.md](./architecture.md).

## Live environments and endpoints

| Layer | URL |
| --- | --- |
| Public portfolio UI (Streamlit Community Cloud) | [Open UI](https://quick-commerce-sla-mlops-jvumckgud4mzzemtzo2aj2.streamlit.app/) |
| Production EC2 operations UI | [http://35.154.79.54:8501](http://35.154.79.54:8501/) |
| FastAPI Swagger docs | [http://35.154.79.54:8000/docs](http://35.154.79.54:8000/docs) |
| API health and model status | [http://35.154.79.54:8000/health](http://35.154.79.54:8000/health) |
| Docker Hub API image | [mahi0409/quick-commerce-sla-api](https://hub.docker.com/r/mahi0409/quick-commerce-sla-api) |
| Docker Hub frontend image | [mahi0409/quick-commerce-sla-frontend](https://hub.docker.com/r/mahi0409/quick-commerce-sla-frontend) |
| GitHub repository | [Mahi-Jadeja/quick-commerce-sla-mlops](https://github.com/Mahi-Jadeja/quick-commerce-sla-mlops) |

These public endpoints use HTTP as configured for this deployment. Avoid
sending sensitive data to them.

## Key features

- **Single-order risk prediction:** breach flag, probability, risk level, and
  dispatch recommendation.
- **Batch prediction:** score multiple orders through `/predict/batch`.
- **Runtime S3 model download:** the API loads the model at startup if it is
  not available locally.
- **Distance feature engineering:** Haversine distance between restaurant and
  delivery coordinates.
- **Two Streamlit interfaces:** an EC2 Compose UI and an optional public
  Streamlit Community Cloud UI.
- **Health reporting:** `/health` exposes API and model readiness for
  operations and Compose health checks.

## Architecture overview

```text
Kaggle delivery CSV
        │
        ▼
Preprocessing and feature engineering
        │
        ▼
Model training ──► MLflow tracking
        │
        ├──► models/best_model.pkl ──► S3
        └──► metrics.json
                     │
                     ▼
Push / pull request ──► GitHub Actions CI (pytest)

Push to main ─────────► GitHub Actions CD ──► Docker Hub API image
                                                       │
                                                       ▼
                                           AWS EC2 / Docker Compose
                                             ┌─────────┴─────────┐
                                             ▼                   ▼
                                        FastAPI :8000       Streamlit :8501
                                             ▲
                                             │ model download at startup
                                             ▼
                                          AWS S3
```

The detailed architecture, deployment flow, and runtime configuration are in
[architecture.md](./architecture.md).

## Technology stack

| Layer | Technology | Role |
| --- | --- | --- |
| Language | Python 3.11 | Training, API, frontend, and tests |
| Data | Pandas, NumPy | Data cleaning and feature processing |
| Features | Haversine distance | Derive `distance_km` from location coordinates |
| Machine learning | scikit-learn, XGBoost, LightGBM | Train and evaluate classification pipelines |
| Experiment tracking | MLflow | Track candidate model runs and metrics locally |
| Artifact versioning | DVC and S3 | Track data and model artifacts |
| Model storage | AWS S3 | Store `models/best_model.pkl` |
| API | FastAPI, Uvicorn, Pydantic | Health, single-order, and batch inference endpoints |
| AWS SDK | boto3 | Download the model from S3 at API startup |
| Frontend | Streamlit, Requests | Operations console and API client |
| Containers | Docker, Docker Compose, Buildx | Package and run the API and frontend |
| Image registry | Docker Hub | Host container images |
| CI/CD | GitHub Actions | Run tests and publish the API image |
| Compute | AWS EC2, `ap-south-1` | Host the Compose deployment |
| Testing | pytest, httpx | Validate preprocessing, API contracts, and training |

## Data and model

- **Source:** Kaggle Food Delivery Dataset.
- **Raw data path:** `data/raw/quick_commerce.csv` (managed through DVC).
- **SLA label:** delivery time greater than 30 minutes is labeled as a breach.
- **Processed data:** `data/processed/train.csv` and `data/processed/test.csv`.
- **Production model:** `s3://mlops-sla-mahipune01/models/best_model.pkl`.
- **Model selection:** the training script chooses the candidate with the
  highest ROC-AUC and writes its evaluation summary to `metrics.json`.

### Current best-model snapshot

| Metric | Value |
| --- | ---: |
| Algorithm | LightGBM classifier |
| ROC-AUC | 0.9852 |
| F1 score | 0.8900 |
| Accuracy | 0.9385 |
| Target | `will_breach_sla` (`delivery_time_min > 30`) |

The values above are the current contents of `metrics.json`; they may change
when the model is retrained.

## Repository structure

```text
quick-commerce-sla-mlops/
├── .github/workflows/
│   ├── ci.yml                 # Tests on pushes and pull requests
│   └── cd.yml                 # Builds and publishes the API image
├── api/
│   ├── app.py                 # FastAPI routes: /, /health, /predict, /predict/batch
│   ├── model_loader.py        # Load local model or download from S3
│   └── schemas.py             # Pydantic request and response contracts
├── data/
│   ├── raw/                   # DVC-tracked source data
│   └── processed/             # Generated train and test data
├── frontend/
│   └── ui.py                  # Streamlit operations console
├── models/                    # Local model artifact (DVC-managed)
├── src/
│   ├── config.py              # Paths, SLA threshold, and AWS defaults
│   ├── logger.py              # Shared logging
│   ├── preprocessing.py       # Data cleaning, Haversine, and SLA label
│   └── train.py               # Model training, MLflow logging, metrics
├── tests/
│   ├── conftest.py            # CI fixture for a missing model artifact
│   ├── test_api.py            # API contracts and validation
│   ├── test_preprocessing.py  # Preprocessing and distance tests
│   └── test_train.py          # Training and model smoke tests
├── Dockerfile                 # FastAPI image; model is not copied into it
├── Dockerfile.frontend        # Streamlit image
├── docker-compose.yml         # Local Compose stack
├── docker-compose.prod.yml    # Production Compose configuration
├── metrics.json               # Best-model evaluation summary
└── architecture.md            # Detailed architecture documentation
```

## Local setup and execution

### 1. Create an environment and install dependencies

```bash
git clone https://github.com/Mahi-Jadeja/quick-commerce-sla-mlops.git
cd quick-commerce-sla-mlops

python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The raw dataset and model are DVC-managed rather than committed as ordinary
Git files. If you have access to the configured DVC remote, retrieve the
artifacts before preprocessing, training, or local inference:

```bash
dvc pull
```

### 2. Preprocess and train

```bash
python -m src.preprocessing
python -m src.train
```

Training writes the selected model to `models/best_model.pkl` and the metrics
summary to `metrics.json`.

### 3. View MLflow experiments

Run the tracking UI on the training machine:

```bash
mlflow ui --port 5001
```

Open [http://localhost:5001](http://localhost:5001) to compare model runs.
MLflow is intended to run locally rather than on the small EC2 instance.

### 4. Run tests

```bash
pytest tests/ -v --tb=short
```

The test fixtures can create a small dummy model when the model artifact is
missing, so API tests do not require AWS credentials.

### 5. Run services with Docker Compose

```bash
docker compose up --build
```

- API: [http://localhost:8000](http://localhost:8000)
- API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Streamlit UI: [http://localhost:8501](http://localhost:8501)

The API container does not include the model file. Provide AWS credentials
through your local AWS configuration or environment and allow access to the
configured S3 model object, unless adapting the image to mount a local model.
Never commit AWS credentials or other secrets.

## Production notes

- **Instance:** AWS EC2 `t3.micro`, region `ap-south-1`.
- **Ports:** `8000` (API) and `8501` (Streamlit UI); restrict inbound access
  through the EC2 security group as appropriate.
- **Model:** the API downloads `s3://mlops-sla-mahipune01/models/best_model.pkl`
  at startup when no local artifact is present.
- **Compose networking:** the frontend calls `http://api:8000`.
- **Health checks:** Compose checks the API before starting the frontend.
- **Image publishing:** the current CD workflow builds and pushes only the
  API image for `linux/amd64`; it does not build the frontend image or deploy
  the Compose stack to EC2.
- **Credentials:** use an EC2 instance role with least-privilege S3 access
  where possible. Do not place credentials in source control or Docker image
  layers.

## Testing and engineering

- `pytest` covers preprocessing, API validation, prediction, and training
  behavior.
- CI uses Python 3.11 and runs tests on pushes and pull requests to `dev` and
  `main`.
- The test fixture supplies a dummy pipeline if the model artifact is absent.
- The API and frontend use separate slim Docker images.
- The model is retrieved at runtime rather than embedded in the API image.

## Dataset

The project uses the Kaggle Food Delivery Dataset. The raw CSV is stored at
`data/raw/quick_commerce.csv` through DVC. The preprocessing step derives the
30-minute SLA breach label and engineered features before splitting the data
for training and evaluation.
