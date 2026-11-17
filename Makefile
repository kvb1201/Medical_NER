# Medical NER — HMM / MEMM / CRF
# Run `make help` for targets.

PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
export PYTHONPATH := .

HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: help install test data train eval run serve demo

help:
	@echo "Targets:"
	@echo "  make install   create .venv and install requirements.txt"
	@echo "  make test      run pytest"
	@echo "  make data      download BC5CDR and build dataset splits"
	@echo "  make train     train HMM, MEMM, CRF and write reports/"
	@echo "  make eval      evaluate existing models/ checkpoints only"
	@echo "  make run       start the demo at http://$(HOST):$(PORT)"
	@echo ""
	@echo "Optional: HOST=0.0.0.0 PORT=8080 make run"

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

data:
	$(PYTHON) src/data/prepare_dataset.py

train:
	$(PYTHON) src/models/train_and_evaluate.py

eval:
	$(PYTHON) src/models/train_and_evaluate.py --eval-only

run serve demo:
	$(PYTHON) app/server.py --host $(HOST) --port $(PORT)
