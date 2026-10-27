import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import DeliveryInput, PredictionOutput, HealthResponse, BatchDeliveryInput
from api.model_loader import get_model
from src.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to ensure model is pre-loaded on application startup."""
    logger.info("Initializing FastAPI Quick-Commerce SLA API...")
    try:
        model, source = get_model()
        logger.info(f"Model successfully loaded at startup from {source}")
    except Exception as e:
        logger.error(f"Error pre-loading model on startup: {e}")
    yield
    logger.info("Shutting down Quick-Commerce SLA API service...")


app = FastAPI(
    title="Quick-Commerce Delivery SLA Risk Prediction API",
    description="MLOps microservice predicting 30-minute delivery SLA breach risk for Q-Commerce operations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Quick-Commerce Delivery SLA Risk API",
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint evaluating service status and model readiness."""
    try:
        model, source = get_model()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_source=source,
            version="1.0.0",
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_source="Unavailable",
            version="1.0.0",
        )


def _predict_single(order: DeliveryInput, model) -> PredictionOutput:
    payload_df = pd.DataFrame([order.model_dump()])
    
    # Predict SLA breach binary (1 = breach > 30 mins) and probability
    pred = int(model.predict(payload_df)[0])
    proba = float(model.predict_proba(payload_df)[0][1])

    # Categorize Risk Level & Recommendation
    if proba >= 0.75:
        risk_level = "HIGH"
        recommendation = "High breach risk! Assign priority rider or split order dispatch immediately."
    elif proba >= 0.40:
        risk_level = "MEDIUM"
        recommendation = "Moderate risk. Monitor delivery route and optimize prep time."
    else:
        risk_level = "LOW"
        recommendation = "Normal operations. Delivery expected within 30-minute SLA."

    return PredictionOutput(
        will_breach_sla=pred,
        sla_breach_probability=round(proba, 4),
        risk_level=risk_level,
        recommendation=recommendation,
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict_sla_risk(payload: DeliveryInput):
    """Predict SLA breach probability and risk level for a single delivery order."""
    try:
        model, _ = get_model()
        return _predict_single(payload, model)
    except Exception as e:
        logger.error(f"Inference error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=list[PredictionOutput], tags=["Inference"])
def predict_batch_sla_risk(payload: BatchDeliveryInput):
    """Predict SLA breach probability and risk level for a batch of delivery orders."""
    try:
        model, _ = get_model()
        results = [_predict_single(order, model) for order in payload.orders]
        return results
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference execution failed: {str(e)}",
        )
