"""ODE solving"""
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController, ConstantStepSize
import jax.numpy as jnp
from absl import flags

DT0 = flags.DEFINE_float(
    'ode_dt0', 1e-4,
    'ode int initial time step/time step when using constant step size')
T0 = flags.DEFINE_float('ode_t0', 0.0, 'ode int start time')
T1 = flags.DEFINE_float('ode_t1', 1.0, 'ode int end time')
SAVEAT_DT = flags.DEFINE_float('saveat_dt', 1.0e-2, 'ode int end time')
ATOL = flags.DEFINE_float('ode_atol', 1.0e-8, 'ode int absolute tolerance')
RTOL = flags.DEFINE_float('ode_rtol', 1.0e-8, 'ode int relative tolerance')
USE_ADAPTIVE_STEP = flags.DEFINE_boolean(
    'ode_use_adaptive_step', False,
    'Whether to use an adaptive step size for ode solution.')
MAX_STEPS = flags.DEFINE_integer('ode_max_steps', 10000,
                                 'Maximum number of ode integration steps')


def solve_ode(ode_rhs, y0):
  term = ODETerm(ode_rhs)
  solver = Dopri5()
  t_save = jnp.arange(T0.value, T1.value + SAVEAT_DT.value-SAVEAT_DT.value/10, SAVEAT_DT.value)
  saveat = SaveAt(ts=t_save)
  print(t_save)
  if USE_ADAPTIVE_STEP.value:
    stepsize_controller = PIDController(rtol=RTOL.value, atol=ATOL.value)
  else:
    stepsize_controller = ConstantStepSize()
  solution = diffeqsolve(
      term,
      solver,
      t0=T0.value,
      t1=T1.value,
      dt0=DT0.value,
      y0=y0,
      saveat=saveat,
      stepsize_controller=stepsize_controller,
      max_steps=MAX_STEPS.value)
  return solution.ts, solution.ys
