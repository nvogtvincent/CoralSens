import numpy as np
import ecoevo
import importlib
importlib.reload(ecoevo)
from datetime import timedelta
from matplotlib import pyplot as plt

# BASE PARAMETERS
n_param = 50
years   = 100  # Proper run
years_S = 1000 # Spin-up
sites   = n_param**2

T_mean  = 26.
T_seas  = 4.
dT      = 3.

r0_base = 2.45
m0_base = 2.45
w_base  = 6.
f_base  = 0.01 # Assuming f0 = 250/polyp, r=2e-4, retention=10%
V_base  = 0.05

# PARAMETER RANGES
dw = 4
w = np.linspace(w_base-dw, w_base+dw, num=n_param)
V = np.logspace(-1, 1, num=n_param)*V_base
w, V = np.meshgrid(w, V)

w = w.flatten()
V = V.flatten()

# SPIN UP SIMULATION
sim = ecoevo.simulation(i=sites, j=years_S*12) # New simulation

# Create boundary conditions with a seasonal cycle
t_spin_up = np.linspace(0, years_S, num=(years_S*12)+1)
T_spin_up = -T_seas*np.cos(2*np.pi*t_spin_up) + T_mean
sim.set_bc(T=T_spin_up[1:], I=0.0, zc_offset=0.0) # No immigration

# Create initial conditions
sim.set_ic(z=T_mean+w, c=1.00)

# Set parameters
sim.set_param(r0=2.45, m0=2.45, w=w, f=0.01, V=V, cmin=0.001)

# Z SHOULD NOT BE CHANGING!!!

# Run simulation
sim.run(output_dt=120)
