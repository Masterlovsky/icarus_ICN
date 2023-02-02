CONFIG_FILE ?= config.py
RESULTS_FILE ?= results.pickle
PRINT_FILE ?= results/pps_10e8_10x100.txt

.PHONY: check_installed run clean

all: run

check_installed:
	@[ -x "$(shell which icarus)" ] || { echo "Icarus is not installed. Install it first."; exit 1; }

run: check_installed
	icarus run --results $(RESULTS_FILE) $(CONFIG_FILE)

print: check_installed
	icarus results print $(RESULTS_FILE) > $(PRINT_FILE)

clean:
	rm -rf $(RESULTS_FILE)
