PYTHON ?= python3

.PHONY: all data run report clean guard
all: data run report

data:
	$(PYTHON) -m secom.data

run:
	$(PYTHON) -m secom.run

report:
	$(PYTHON) -m secom.report

clean:
	rm -rf results/* figures/*

guard:
	$(PYTHON) scripts/identity_guard.py
