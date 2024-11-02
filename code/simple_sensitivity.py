import numpy as np
import statsmodels.api as sm
import pandas as pd
import ecoevo
import cmasher as cmr
import importlib
importlib.reload(ecoevo)
from datetime import timedelta
from matplotlib import pyplot as plt

# BASE PARAMETERS
n_param = 500
years   = 100  # Proper run
years_S = 100 # Spin-up
sites   = n_param**2

T_mean  = 26.
T_seas  = 4.
dT      = 3.

r0_base = 0.37 # Based on mu=-4.2, sigma=1.9, e=0.05
m0_base = 4.6
w_base  = 4.
f_base  = 0.01 # Assuming f0 = 250/polyp, r=2e-4, retention=10%
V_base  = 0.05

# PARAMETER RANGES
dw = 2
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
sim.set_ic(z=T_mean+T_seas, c=0.50)

# Set parameters
sim.set_param(r0=r0_base, m0=m0_base, w=w, f=0.01, V=V, cmin=0.001)

# Run simulation
sim.run(output_dt=1)
init_c = sim.output.c[:, -1].data
init_z = sim.output.z[:, -1].data

# TEMPERATURE RAMP-UP SIMULATION
sim = ecoevo.simulation(i=sites, j=years*12) # New simulation

# Create boundary conditions with a seasonal cycle
t = np.linspace(0, years, num=(years*12)+1)
T = -T_seas*np.cos(2*np.pi*t) + T_mean + np.linspace(0, dT, num=(years*12)+1)
sim.set_bc(T=T[1:], I=0.0, zc_offset=0.0) # No immigration

# Create initial conditions
sim.set_ic(z=init_z, c=init_c)

# Set parameters
sim.set_param(r0=r0_base, m0=m0_base, w=w, f=0.01, V=V, cmin=0.001)

# Run simulation
sim.run(output_dt=12)

# ANALYSES
# Compute change in coral cover
rel_change = (sim.output.c.min(dim='time')-sim.output.c[:, 0])/sim.output.c[:, 0]

# Construct an OLS least-squares model to identify key parameters
df = pd.DataFrame(data={'decline': rel_change, 'w': w, 'V': V})
endog = df['decline']
exog = sm.add_constant(df[['w', 'V']])
ols = sm.OLS(endog=endog, exog=exog).fit()

# Plot
_decline = df['decline'].values.reshape((n_param, n_param))
_w = df['w'].values.reshape((n_param, n_param))
_V = df['V'].values.reshape((n_param, n_param))

plt.close()
f, ax = plt.subplots(1, 1, figsize=(6, 6))

cplot = ax.contourf(_w, _V, _decline, levels=np.linspace(-1, 0, num=11),
                    cmap=cmr.lavender, vmin=-1, vmax=0)
ax.set_xlabel('Thermal tolerance (C)')
ax.set_ylabel('Additive genetic variance (C$^2$)')
#ax.set_yscale('log')

plt.colorbar(cplot, label='Relative change in coral cover after ' + str(dT) + 'C warming')
