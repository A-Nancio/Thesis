.PHONY: test
COMMIT_ID := $(shell git rev-parse HEAD)
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
TIMESTAMP := $(shell git show -s --format=%cI HEAD | cat)

all: install

clean:
	rm -rf .mypy_cache/ .pytest_cache/
	find . -type d -name __pycache__ | xargs rm -rf

install:
	pip install --upgrade pip
	pip install -r requirements.txt

run:
	python src/App.py

upgrade:
	pip install --upgrade pip-upgrader
	pip-upgrade --skip-package-installation requirements.txt
