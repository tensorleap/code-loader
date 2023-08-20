POETRY_MODULE := poetry run python -m
PYTEST := $(POETRY_MODULE) pytest

.PHONY: run_tests
run_tests:
	(cd code-loader && $(PYTEST) tests -v)

.PHONY: watch
watch:
	(cd code-loader && $(POETRY_MODULE) pytest_watch --runner "python -m pytest -v -k $(K)")

.PHONY: lint
lint_code_loader:
	(cd code-loader && $(POETRY_MODULE) mypy --install-types --non-interactive .)

.PHONY: lint_strict_code
lint_strict_code_loader:
	(cd code-loader && $(POETRY_MODULE) mypy --install-types --non-interactive --strict code_loader)

lint_strict_helpers:
	(cd code-loader.helpers && $(POETRY_MODULE) mypy --install-types --non-interactive --strict code_loader)

.PHONY: lint_tests
lint_tests:
	(cd code-loader && $(POETRY_MODULE) mypy --install-types --non-interactive tests)

.PHONY: test_with_coverage
test_with_coverage:
	(cd code-loader && $(PYTEST) --cov=code_loader --cov-branch --no-cov-on-fail --cov-report term-missing --cov-report html -v tests/)
