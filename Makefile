.PHONY: test check

test:
	pytest --ignore test/test_hwas.py

check:
	pyflakes **/*.py
	pycodestyle --ignore E402 **/*.py
