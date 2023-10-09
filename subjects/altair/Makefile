all: install

install:
	python setup.py install

test :
	black --diff --color --check .
	flake8 . --statistics
	mypy altair tests
	python -m pytest --pyargs --doctest-modules tests

test-coverage:
	python -m pytest --pyargs --doctest-modules --cov=altair --cov-report term altair

test-coverage-html:
	python -m pytest --pyargs --doctest-modules --cov=altair --cov-report html altair
