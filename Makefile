# Pipeline stages share generated inputs and must run in order.
.NOTPARALLEL:
PY ?= python3

.PHONY: all data fit analysis clean site

all: verify-data data fit analysis

data:
	$(PY) src/run_01_build.py

fit:
	$(PY) src/run_02_fit.py

analysis:
	$(PY) src/run_03_league.py
	$(PY) src/run_04_players.py
	$(PY) src/run_05_reliability.py
	$(PY) src/run_06_defense.py
	$(PY) src/run_07_robustness.py

# Regenerate the GitHub Pages page from README.md. Needs `pip install -r requirements.txt`.
site:
	$(PY) tools/site.py

clean:
	rm -f figures/*.png output/*.csv data/processed/*

.PHONY: fetch-data verify-data
fetch-data:
	$(PY) tools/fetch_data.py

verify-data:
	$(PY) tools/fetch_data.py --check

.PHONY: report
report:
	$(PY) tools/report.py
