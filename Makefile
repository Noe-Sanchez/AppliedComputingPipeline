VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python

$(VENV_DIR): requirements.txt
	python3 -m venv $(VENV_DIR)
	$(PYTHON) -m pip install -r requirements.txt

run: $(VENV_DIR)
	$(PYTHON) main.py

test: $(VENV_DIR)
	$(PYTHON) -m pytest

clean:
	rm -rf $(VENV_DIR)

.PHONY: run test clean

