.PHONY: test check

test:
	trial test/test_*.py

check:
	pyflakes */*.py
	pycodestyle */*.py
