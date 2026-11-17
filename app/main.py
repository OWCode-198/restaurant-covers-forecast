"""ASGI entrypoint.

Run it with:

    uvicorn app.main:app --reload

Interactive docs are at /docs; the OpenAPI schema is at /openapi.json.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers.api import router
from app.routers.web import router as web_router
from app.service import CoversService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DESCRIPTION = """
Covers forecasting for a Toast restaurant.

Daily covers are read from a **Toast** connection or a **CSV export**, enriched
with weather, public holidays and nearby ticketed events, and fed to the same
interpretable regression used in this project's notebooks and CLI.

Typical flow:

1. `POST /ingest` -- pull covers history into the store (safe to re-run; put it on a schedule)
2. `GET /forecast?days=7` -- the week ahead, with the inputs behind each number
3. `GET /accuracy` -- how the forecasts have actually been doing

Switch data sources with `COVERS_SOURCE=csv|toast`; `GET /health?check_source=true`
confirms the connection works.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # The source itself is built lazily, so bad Toast credentials surface on the
    # first request that needs them rather than preventing startup entirely.
    app.state.service = CoversService(settings)
    logger.info(
        "Covers API ready (source=%s, db=%s, location=%s)",
        settings.source,
        settings.db_path,
        app.state.service.location.name if app.state.service.location else "unset",
    )
    yield
    logger.info("Covers API shutting down.")


app = FastAPI(
    title="Predictive Covers API",
    description=DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
)

# A dashboard served from another origin is the expected consumer. Tighten
# `allow_origins` to your own domain before putting this on a public host.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
# The human-readable page. Unauthenticated by design -- it is meant to be
# opened on a phone with no login -- so put an auth proxy, a VPN or at minimum
# an unguessable path in front of it before this reaches a public host.
app.include_router(web_router)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "name": "Predictive Covers API",
        "docs": "/docs",
        "health": "/health",
        "forecast": "/forecast?days=7",
        "today": "/today",
    }
