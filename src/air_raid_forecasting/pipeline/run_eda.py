"""Pipeline step 3: exploratory data analysis (figures + summary JSON).

    python -m air_raid_forecasting.pipeline.run_eda
"""

from __future__ import annotations

import json

from air_raid_forecasting.config import load_config
from air_raid_forecasting.eda import run_eda
from air_raid_forecasting.logging_utils import get_logger

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> dict:
    cfg = load_config()
    summary = run_eda(cfg)
    log.info("EDA complete. Summary:\n%s", json.dumps(summary, indent=2)[:2000])
    return summary


if __name__ == "__main__":
    main()
