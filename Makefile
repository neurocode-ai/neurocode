.PHONY: clear-pycache
clear-pycache:
	./scripts/cc.sh

.PHONY: pip-install
pip-install:
	python -m pip install -r requirements.txt

.PHONY: unittest
unittest:
	python -m unittest

.PHONY: clean-unittest
clean-unittest: clear-pycache unittest
