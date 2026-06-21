"""Unified ``arf`` command-line interface.

    arf ingest [--force]
    arf preprocess
    arf eda
    arf features
    arf train [--fast]
    arf all [--fast] [--skip-ingest]
    arf serve
    arf dashboard
"""

from __future__ import annotations

import argparse
import sys

from air_raid_forecasting.pipeline import (
    run_all,
    run_eda,
    run_features,
    run_ingest,
    run_preprocess,
    run_train,
)


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="arf", description="Air raid forecasting toolkit.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "preprocess", "eda", "features", "train", "all", "serve", "dashboard"):
        sub.add_parser(name, add_help=False)

    args, rest = parser.parse_known_args(argv)
    cmd = args.command

    if cmd == "ingest":
        run_ingest.main(rest)
    elif cmd == "preprocess":
        run_preprocess.main(rest)
    elif cmd == "eda":
        run_eda.main(rest)
    elif cmd == "features":
        run_features.main(rest)
    elif cmd == "train":
        run_train.main(rest)
    elif cmd == "all":
        run_all.main(rest)
    elif cmd == "serve":
        from air_raid_forecasting.service.main import main as serve_main
        serve_main()
    elif cmd == "dashboard":
        import subprocess
        subprocess.run(["streamlit", "run", "dashboard/streamlit_app.py", *rest], check=False)


if __name__ == "__main__":
    main()
