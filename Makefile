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

.PHONY: lint_strict_helpers
lint_strict_helpers:
	(cd code-loader.helpers && $(POETRY_MODULE) mypy --install-types --non-interactive --strict code_loader --explicit-package-bases)

.PHONY: lint_tests
lint_tests:
	(cd code-loader && $(POETRY_MODULE) mypy --install-types --non-interactive tests)

.PHONY: test_with_coverage_code_loader
test_with_coverage_code_loader:
	(cd code-loader && $(PYTEST) --cov=code_loader --cov-branch --no-cov-on-fail --cov-report term-missing --cov-report html -v tests/)

.PHONY: test_with_coverage_helpers
test_with_coverage_helpers:
	(cd code_loader_helpers && $(PYTEST) --cov=code_loader_helpers --cov-branch --no-cov-on-fail --cov-report term-missing --cov-report html -v tests/)


.PHONY: install_code_loader_dependencies
install_code_loader_dependencies:
	(cd code-loader && poetry install)

.PHONY: install_helpers_dependencies
install_helpers_dependencies:
	(cd code_loader_helpers && poetry install)
