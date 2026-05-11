"""
FastAPI application entry point.
Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → ChatResponse (see models.py)
"""

import logging
import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agent import run_agent
from app.models import ChatRequest, ChatResponse
from app.retrieval import retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize retriever at startup (warm up ChromaDB + BM25)."""
    log.info("Starting SHL Recommender — loading catalog and retrieval index...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, retriever.initialize)
    log.info("Retrieval index ready ✓")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL talent assessment recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    """Readiness check. Returns 200 once the service is up."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless conversational endpoint.
    Accepts the full message history and returns the next agent reply.
    """
    try:
        response = await asyncio.wait_for(
            run_agent(request.messages),
            timeout=25.0,  # Under the 30s evaluator cap
        )
        return response
    except asyncio.TimeoutError:
        log.error("Agent timed out after 25s")
        raise HTTPException(
            status_code=504,
            detail="Request timed out. Please try again.",
        )
    except Exception as e:
        log.exception(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
