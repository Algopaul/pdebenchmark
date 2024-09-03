include config/runner_definition.mk
include config/setup.mk

bps = $(shell seq 6 0.01 7)

$(addsuffix .npy,$(addprefix ham_wave_2d_,$(bps))): ham_wave_2d_%.npy: .venv
	$(RUN) .venv/bin/python pdebenchmark/driver.py\
		--flagfile=config/ham_wave_flags.flags\
		--bump_width=${*}\
		--outfile=ham_wave_2d_${*}\

all_ham_wave: $(addsuffix .npy,$(addprefix ham_wave_2d_,$(bps)))
