"""Root entry point for the FastAPI prediction service.

    python service/main.py
    # or
    uvicorn air_raid_forecasting.service.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import uvicorn

from air_raid_forecasting.config import load_config


def main() -> None:
    cfg = load_config()
    uvicorn.run(
        "air_raid_forecasting.service.app:app",
        host=cfg.service.host,
        port=cfg.service.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
