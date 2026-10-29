"""FastAPI service that exposes the forecast to other systems (for example Spring Boot).

Run locally with ``make api`` (bound to 127.0.0.1). Do not expose it publicly without
authentication or a TLS reverse proxy. Export the contract with
``python -m fxhedge.api --export-openapi docs/openapi.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from fxhedge import __version__
from fxhedge.api_models import (
    DISCLAIMER,
    HORIZONS,
    ForecastResponse,
    ScenarioName,
    build_response,
)
from fxhedge.inference import predict
from fxhedge.model_training import ModelIntegrityError

logger = logging.getLogger(__name__)

ForecastFn = Callable[[date, int, ScenarioName], pd.DataFrame]


def _default_forecast(run_date: date, horizon: int, scenario: ScenarioName) -> pd.DataFrame:
    return predict(run_date, horizon, scenario)


def create_app(forecast_fn: ForecastFn = _default_forecast) -> FastAPI:
    """Build the application. ``forecast_fn`` can be replaced in tests."""
    app = FastAPI(title="fxhedge forecast API", version=__version__, description=DISCLAIMER)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/forecast", response_model=ForecastResponse)
    def forecast(
        horizon: int = 30,
        scenario: ScenarioName = "base",
        run_date: date | None = None,
    ) -> ForecastResponse:
        if horizon not in HORIZONS:
            raise HTTPException(status_code=422, detail=f"horizon must be one of {list(HORIZONS)}")
        day = run_date or date.today()
        try:
            frame = forecast_fn(day, horizon, scenario)
            return build_response(frame, day, horizon, scenario)
        except FileNotFoundError as exc:
            logger.error("Forecast unavailable: %s", exc)
            raise HTTPException(status_code=503, detail="Models or data are not available") from exc
        except ModelIntegrityError as exc:
            logger.error("Integrity check failed: %s", exc)
            raise HTTPException(status_code=503, detail="Model integrity check failed") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: export the OpenAPI contract."""
    parser = argparse.ArgumentParser(description="fxhedge API utilities.")
    parser.add_argument("--export-openapi", type=Path, required=True)
    args = parser.parse_args(argv)
    args.export_openapi.parent.mkdir(parents=True, exist_ok=True)
    args.export_openapi.write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
