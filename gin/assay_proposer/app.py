"""
gin.assay_proposer.app
----------------------
FastAPI app factory for the SEAR proposer sidecar.

One model, one decode at a time: a llama.cpp model is not safe to drive from
two threads, and FastAPI runs sync endpoints on a thread pool, so every
propose call takes the same lock. Receipts sends passes three at a time; the
extra requests queue here. propose_fn is injected so tests run without a
model, as gin/federation/server.py does.
"""
from __future__ import annotations

import threading
from typing import Callable

from fastapi import FastAPI, HTTPException

from .schema import Health, Proposal, ProposeRequest, ProposeResponse

ProposeFn = Callable[[ProposeRequest], list[dict]]


def create_app(propose_fn: ProposeFn, model_id: str) -> FastAPI:
    app = FastAPI(title="gin assay proposer")
    lock = threading.Lock()

    @app.get("/v1/health", response_model=Health)
    def health() -> Health:
        return Health(model=model_id)

    @app.post("/v1/propose", response_model=ProposeResponse)
    def propose(req: ProposeRequest) -> ProposeResponse:
        if req.model != model_id:
            raise HTTPException(status_code=409, detail=f"this proposer serves {model_id}, not {req.model}")
        try:
            with lock:
                raw = propose_fn(req)
            proposals = [Proposal.model_validate(p) for p in raw]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
        return ProposeResponse(model=model_id, proposals=proposals)

    return app
