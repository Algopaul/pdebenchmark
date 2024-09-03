"""IO Functions"""
import h5py
import jax.numpy as jnp


def store_solution_to_h5py(
    solution,
    space_grid,
    time_grid,
    config_dict,
    filename,
):
  with h5py.File(f'{filename}.hdf5', 'w') as f:
    num_quantities = solution.shape[1]
    f.create_dataset('u', data=solution)
    f.create_group('config')
    f['config'].attrs.update(config_dict)
    f['t'] = time_grid
    f['t'].make_scale('t')
    f['u'].dims[0].attach_scale(f['t'])
    f['x0'] = space_grid[0]
    f['x1'] = space_grid[1]
    f['x0'].make_scale('x0')
    f['x1'].make_scale('x1')
    f['u'].dims[2].attach_scale(f['x0'])
    f['u'].dims[3].attach_scale(f['x1'])
    f['q'] = jnp.arange(0, num_quantities, 1)
    f['q'].make_scale('q')
    f['u'].dims[1].attach_scale(f['q'])
    pass
