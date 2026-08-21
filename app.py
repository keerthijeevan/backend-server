from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
import time


# ------------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------------

MODEL_NAME = "cross-encoder/ettin-reranker-17m-v1"


# ------------------------------------------------------------------
# Load reranker model
# ------------------------------------------------------------------

print(f"Loading reranker: {MODEL_NAME}")

start = time.perf_counter()

model = CrossEncoder(MODEL_NAME)

load_time = time.perf_counter() - start

print(f"Reranker loaded in {load_time:.3f}s")


# ------------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------------

app = FastAPI(
    title="Prana Reranker Service",
    version="1.0.0",
)


# ------------------------------------------------------------------
# Request schema
# ------------------------------------------------------------------

class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int = Field(default=5, ge=1)


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


# ------------------------------------------------------------------
# Reranking endpoint
# ------------------------------------------------------------------

@app.post("/rerank")
def rerank(request: RerankRequest):

    # Nothing to rerank
    if not request.documents:
        return {
            "results": [],
            "inference_time": 0.0,
            "model": MODEL_NAME,
            "count": 0,
        }

    # Create query-document pairs
    pairs = [
        (request.query, document)
        for document in request.documents
    ]

    print(
        f"[RERANK] Query received with "
        f"{len(pairs)} documents"
    )

    # --------------------------------------------------------------
    # CrossEncoder inference
    # --------------------------------------------------------------

    start = time.perf_counter()

    scores = model.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False,
    )

    inference_time = time.perf_counter() - start

    print(
        f"[TIMING] Reranker inference: "
        f"{inference_time:.3f}s"
    )

    # --------------------------------------------------------------
    # Build results
    # --------------------------------------------------------------

    results = [
        {
            "index": index,
            "score": float(score),
        }
        for index, score in enumerate(scores)
    ]

    # Highest score first
    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # Return only requested number
    results = results[:request.top_k]

    # --------------------------------------------------------------
    # Response
    # --------------------------------------------------------------

    return {
        "results": results,
        "inference_time": inference_time,
        "model": MODEL_NAME,
        "count": len(request.documents),
    }
