PYTHONPATH := .
POETRY_MODULE := poetry run python -m
PYTEST := $(POETRY_MODULE) pytest

.PHONY: run_tests
run_tests:
	$(PYTEST) tests -v

.PHONY: watch
watch:
	$(POETRY_MODULE) pytest_watch --runner "python -m pytest -v -k $(K)"

.PHONY: lint
lint:
	$(POETRY_MODULE) mypy --install-types --non-interactive .

.PHONY: test_with_coverage
test_with_coverage:
	$(PYTEST) --cov=code_loader --cov-branch --no-cov-on-fail --cov-report term-missing --cov-report html -v tests/
