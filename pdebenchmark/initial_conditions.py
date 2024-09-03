"""Functions used as initial conditions"""
from absl import flags
import jax.numpy as jnp

# One string per value (zero or bump)
INIT_FUNCS = flags.DEFINE_multi_string(
    'init_funcs', ['bump'], 'The init funcs for the different values')
BUMP_SHIFTS = flags.DEFINE_multi_float('bump_shifts', [0.0],
                                       'Where to shift the bump')
BUMP_WIDTH = flags.DEFINE_float('bump_width', 1.0, 'The bump width')
OFFSETS = flags.DEFINE_multi_float(
    'offsets', [0.0], 'By how much to offset the different values')


def bump(x, offset):
  bump_shifts = jnp.asarray(BUMP_SHIFTS.value)
  return offset + jnp.exp(
      -BUMP_WIDTH.value**2 * jnp.linalg.norm(x - bump_shifts, axis=1)**2)


def constant(x, offset):
  return 0 * jnp.linalg.norm(x, axis=1) + offset


def get_init_cond(grid):
  func_dict = {
      'bump': bump,
      'constant': constant,
  }
  u = []
  if len(grid.shape) == 1:
    grid = jnp.expand_dims(grid, 1)
  for func, offset in zip(INIT_FUNCS.value, OFFSETS.value):
    u.append(func_dict[func](grid, offset))
  return jnp.hstack(u)


def get_num_quantities():
  return len(INIT_FUNCS.value)
