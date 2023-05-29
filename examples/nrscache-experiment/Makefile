CONFIG_FILE ?= config.py
RESULTS_FILE ?= results.pickle
PRINT_FILE ?= results/0610/v_alpha_main.txt
CSV_FILE ?= results/0610/v_alpha_main.csv

.PHONY: check_installed run clean

all: run

check_installed:
	@[ -x "$(shell which icarus)" ] || { echo "Icarus is not installed. Install it first."; exit 1; }

run: check_installed
	icarus run --results $(RESULTS_FILE) $(CONFIG_FILE)

print: check_installed
	icarus results print $(RESULTS_FILE) > $(PRINT_FILE)

csv: check_installed
	python3 csvbuilder.py $(RESULTS_FILE) $(CSV_FILE)

clean:
	rm -rf $(RESULTS_FILE)
