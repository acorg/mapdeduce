.PHONY: test check clean
test:
    pytest

check:
	pyflakes */*.py
	pycodestyle --ignore E402 */*.py

clean:
	rm -rf mapdeduce.egg-info test/__pycache__ mapdeduce/__pycache__ _trial_temp
