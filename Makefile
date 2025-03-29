.PHONY: test check

test:
    pytest

check:
	pyflakes **/*.py
	pycodestyle --ignore E402 **/*.py
