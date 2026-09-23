PY := python3

.PHONY: all data fit analysis clean site

all: data fit analysis

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

# Regenerate the GitHub Pages page from README.md. Needs `pip install markdown`.
site:
	$(PY) tools/site.py

clean:
	rm -f figures/*.png output/*.csv data/processed/*
