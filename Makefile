.PHONY: test check

test:
	trial test/*.py

check:
	pyflakes */*.py
	pycodestyle */*.py
