.PHONY: help setup ingest preprocess eda features train all test lint serve dashboard \
        docker-build docker-up clean

PY ?= python
VENV ?= .venv

help:
	@echo "Targets:"
	@echo "  setup        create .venv (py3.12) and install package + deep extra"
	@echo "  ingest       live-download raw alert data"
	@echo "  preprocess   clean + build hourly panels"
	@echo "  eda          generate EDA figures + summary"
	@echo "  features     build feature matrices + targets"
	@echo "  train        backtest, compare, optimize, train production models"
	@echo "  all          run the full pipeline end to end"
	@echo "  test         run the pytest suite"
	@echo "  lint         ruff check"
	@echo "  serve        run the FastAPI prediction service (:8000)"
	@echo "  dashboard    run the Streamlit dashboard (:8501)"
	@echo "  docker-build / docker-up   build & run containers"

setup:
	uv venv --python 3.12 $(VENV) || python3.12 -m venv $(VENV)
	$(VENV)/bin/pip install -e ".[deep,dev]"

ingest:
	$(VENV)/bin/python -m air_raid_forecasting.pipeline.run_ingest

preprocess:
	$(VENV)/bin/python -m air_raid_forecasting.pipeline.run_preprocess

eda:
	$(VENV)/bin/python -m air_raid_forecasting.pipeline.run_eda

features:
	$(VENV)/bin/python -m air_raid_forecasting.pipeline.run_features

train:
	$(VENV)/bin/python -m air_raid_forecasting.pipeline.run_train

all:
	$(VENV)/bin/python -m air_raid_forecasting.pipeline.run_all

test:
	$(VENV)/bin/python -m pytest tests/ -q

lint:
	$(VENV)/bin/ruff check src tests

serve:
	$(VENV)/bin/uvicorn air_raid_forecasting.service.app:app --host 0.0.0.0 --port 8000

dashboard:
	$(VENV)/bin/streamlit run dashboard/streamlit_app.py

docker-build:
	docker compose build

docker-up:
	docker compose up api dashboard

clean:
	rm -rf data/processed/* models/* reports/figures/* reports/*.json reports/*.md logs/*
	find . -type d -name __pycache__ -exec rm -rf {} +
