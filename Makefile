.PHONY: results check test setup

results:  ## rebuild results/ from the saved data (no API key)
	python3 analysis/make_results.py

check: results  ## fail if the committed results differ from a fresh build
	git diff --exit-code -- results/

test:
	uv run --group dev pytest -q

setup:  ## install dependencies for rerunning experiments
	uv sync
