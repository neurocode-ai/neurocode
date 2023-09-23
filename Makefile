.PHONY: clear-pycache
clear-pycache:
	./scripts/cc.sh

.PHONY: pip-install
pip-install:
	python3 -m pip install -r requirements.txt

.PHONY: unittest
unittest:
	python3 -m unittest

.PHONY: build
build:
	python3 -m pip install pip --upgrade
	python3 -m pip install build --upgrade
	python3 -m build

.PHONY: deploy
deploy:
	python3 -m pip install twine --upgrade
	python3 -m twine upload --repository pypi dist/*

.PHONY: clean-unittest
clean-unittest: clear-pycache unittest

.PHONY: build-and-deploy
build-and-deploy: build deploy
