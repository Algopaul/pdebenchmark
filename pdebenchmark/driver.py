"""driver for generating data"""

from absl import app
from absl import flags
from absl import logging
from pdebenchmark.derivatives import get_deriv_dict
from pdebenchmark.grids import get_grid, get_stepsizes, get_axes, reshape_solution
from pdebenchmark.initial_conditions import get_init_cond, get_num_quantities
from pdebenchmark.pdes import get_ode
from pdebenchmark.ode_solving import solve_ode
from pdebenchmark.io import store_solution_to_h5py
import json
import os
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

OUTFILE = flags.DEFINE_string('outfile', '', 'Output file name')


def setup_discretization():
  grid = get_grid()
  u0 = get_init_cond(grid)
  deriv_dict = get_deriv_dict(get_stepsizes())
  ode_rhs = get_ode(deriv_dict, grid)
  return grid, u0, deriv_dict, ode_rhs


def main(_):
  if OUTFILE.value == '':
    logging.fatal('Output file name must be specified')
  _, u0, _, ode_rhs = setup_discretization()

  _, ys = solve_ode(ode_rhs, u0)

  outarray = np.array(ys).T
  logging.info(f'Shape: {outarray.shape}')
  np.save(f'{OUTFILE.value}.npy', outarray)
  pass


if __name__ == '__main__':
  app.run(main)
