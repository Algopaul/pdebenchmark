#!/bin/bash

# Output file
output_file="config/runner_definition.mk"
exp_run="config/exp_run"
mkdir -p config

# Generating setup file
echo "
.venv:
	\$(DEFAULTPYTHON) -m venv .venv
	.venv/bin/pip install -e .
	.venv/bin/pip install -e .[dev]

install: .venv
	.venv/bin/pip install -e .
	.venv/bin/pip install -e .[dev]

reinstall: .venv
	.venv/bin/pip install --force-upgrade -e .
" > config/setup.mk

echo "
--example=ham_wave_2d
--grid_dim=2
--n_points_0=600
--n_points_1=600
--init_funcs=bump
--init_funcs=constant
--init_funcs=constant
--offsets=0
--offsets=0
--offsets=0
--dom_ax_0=-4.0
--dom_ax_0=4.0
--dom_ax_1=-4.0
--dom_ax_1=4.0
--ode_t0=0.0
--ode_t1=8.0
--bump_shifts=2.0
--bump_shifts=2.0
--saveat_dt=5.0e-3
--ode_dt0=5.0e-3
--print_time=True
" > config/ham_wave_flags.flags


read -p "Pleaser enter your preferred python command (python/python3/python3.11) " python_cmd
echo -e "DEFAULTPYTHON=$python_cmd" > "$output_file"

if command -v srun &> /dev/null; then
    # Prompt the user for any extra flags
    read -p "Please enter any extra flags for srun: " extra_flags

    # If srun is available, set the RUN variable accordingly
    echo "RUN=srun $extra_flags --time 20:00 --mem 32G --cpus-per-task=32 --ntasks=1" >> "$output_file"
    echo "srun $extra_flags --time 20:00 --mem 32G --cpus-per-task=32 --ntasks=1" > "$exp_run"

    # Prompt the user for their email
    read -p "Please enter your email: " email


    # Write the extra target into output_file
    echo "run_all:" >> "$output_file"
    echo -e "\t-bash run.sh" >> "$output_file"
    echo -e "\tsrun  --job-name \"QMDEIM jobs complete\" --mail-type END --mail-user $email $extra_flags ls -a"  >> "$output_file"
else
    # If srun is not available, set the RUN variable to an empty string
    echo 'RUN=' >> "$output_file"
    echo "" > "$exp_run"
fi
