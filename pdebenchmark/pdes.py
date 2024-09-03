"""Definitions of pde problems"""
from absl import flags
from pdebenchmark.grids import get_grid_lens
import jax.numpy as jnp
from functools import wraps
from jax.experimental.host_callback import id_print

EXAMPLE = flags.DEFINE_string('example', 'transport_1d',
                              'The example to generate.')
DIFFUSION = flags.DEFINE_float('diffusion', 1.0e-3,
                               'The coefficient for diffusion terms')
ADV_SPEED = flags.DEFINE_float('adv_speed', 1.0,
                               'The coefficient for advection terms')
PRINT_TIME = flags.DEFINE_bool(
    'print_time', False,
    'Whether to print the current time of the integration.')


def get_linear_transport_ode(deriv_dict, grid):
  del grid

  def transport_1d_rhs(t, u, _):
    del t
    return -ADV_SPEED.value * deriv_dict['d_dx0'](u)

  return transport_1d_rhs


def get_burgers_ode(deriv_dict, grid):
  del grid

  def burgers_rhs(t, u, _):
    del t
    adv_part = -ADV_SPEED.value * u * deriv_dict['d_dx0'](u)
    diff_part = DIFFUSION.value * deriv_dict['d2_dx0'](u)
    return adv_part + diff_part

  return burgers_rhs


def get_fields_from_vector(v, num_quantities, shape):
  return jnp.reshape(v, [num_quantities, *shape])


def get_hamiltonian_wave_2d(deriv_dict, grid):
  del grid

  shape = get_grid_lens()
  d_dx = deriv_dict['d_dx0']
  d_dy = deriv_dict['d_dx1']

  def ham_wave_2d(t, u, _):
    del t
    rho, vx, vy = get_fields_from_vector(u, 3, shape)
    drho_dt = -(d_dx(vx) + d_dy(vy))
    dvx_dt = -d_dx(rho)
    dvy_dt = -d_dy(rho)
    return jnp.reshape(jnp.asarray((drho_dt, dvx_dt, dvy_dt)), -1)

  return ham_wave_2d


def get_hamiltonian_wave_1d(deriv_dict, grid):
  del grid

  shape = get_grid_lens()
  d_dx = deriv_dict['d_dx0']

  def ham_wave_1d(t, u, _):
    del t
    rho, vx = get_fields_from_vector(u, 2, shape)
    drho_dt = -d_dx(vx)
    dvx_dt = -d_dx(rho)
    return jnp.reshape(jnp.asarray((drho_dt, dvx_dt)), -1)

  return ham_wave_1d


def id_print_wrapper(ode_rhs):

  @wraps(ode_rhs)
  def wrapper(t, u, args):
    id_print(t)
    res = ode_rhs(t, u, args)
    return res

  return wrapper


def get_ode(deriv_dict, grid):
  ode_dict = {
      'transport_1d': get_linear_transport_ode,
      'burgers_1d': get_burgers_ode,
      'ham_wave_2d': get_hamiltonian_wave_2d,
      'ham_wave_1d': get_hamiltonian_wave_1d,
  }
  ode_rhs = ode_dict[EXAMPLE.value](deriv_dict, grid)
  if PRINT_TIME.value:
    ode_rhs = id_print_wrapper(ode_rhs)
  return ode_rhs
