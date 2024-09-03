# Copyright 2024 Algopaul
"""Utils for grid creation"""

from absl import flags, app
import jax.numpy as jnp

AX_0 = flags.DEFINE_multi_float('dom_ax_0', [-1.0, 1.0],
                                'The 0-th boundary of the domain')
AX_1 = flags.DEFINE_multi_float('dom_ax_1', [-1.0, 1.0],
                                'The 1-th boundary of the domain')
N_0 = flags.DEFINE_integer('n_points_0', 100,
                           'The number of points in the 0-th axis')
N_1 = flags.DEFINE_integer('n_points_1', 1,
                           'The number of points in the 1-th axis')
GRID_DIM = flags.DEFINE_integer('grid_dim', 1, 'The domain dim of the grid')


def validate_flags():
  if GRID_DIM.value == 1 and (not N_1.value == 1):
    raise ValueError('If grid_dim=1, then n_points_1 must be 1.')


def get_grid():
  validate_flags()
  ax0 = AX_0.value
  x0 = jnp.linspace(ax0[0], ax0[1], N_0.value, endpoint=False)
  if GRID_DIM.value == 1:
    return x0
  elif GRID_DIM.value == 2:
    ax1 = AX_1.value
    x1 = jnp.linspace(ax1[0], ax1[1], N_1.value, endpoint=False)
    grids = jnp.meshgrid(x0, x1, indexing='ij')
    zipped_grids = [m.flatten() for m in grids]
    return jnp.array(zipped_grids).T


def get_axes():
  ax0 = AX_0.value
  ax1 = AX_1.value
  x0 = jnp.linspace(ax0[0], ax0[1], N_0.value, endpoint=False)
  x1 = jnp.linspace(ax1[0], ax1[1], N_1.value, endpoint=False)
  return x0, x1


def get_stepsizes():
  x0, x1 = get_axes()
  return [x0[1] - x0[0], x1[1] - x1[0]]


def get_grid_lens():
  return [N_0.value, N_1.value]


def reshape_solution(solution, num_quantities):
  n0, n1 = get_grid_lens()
  return jnp.reshape(solution, (solution.shape[0], num_quantities, n0, n1))


def main(_):
  g = get_grid()
  print(g)


if __name__ == '__main__':
  app.run(main)
