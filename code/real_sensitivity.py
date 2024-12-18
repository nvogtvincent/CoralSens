import numpy as np
import statsmodels.api as sm
import pandas as pd
import ecoevo
import cmasher as cmr
import xarray as xr
import importlib
importlib.reload(ecoevo)
from datetime import timedelta
from matplotlib import pyplot as plt
from pandas import date_range
from datetime import datetime

# CLIMATE DATA DIRECTORY
scenario = '245'
data_dir = '../data/ocean/SSP' + scenario + '.nc'

# BASE PARAMETERS
n_param  = 10  # Number of parameter values for each parameter
years_su = 100 # Spin-up
keep_su  = True 

r0_base = 0.37 # Based on mu=-4.2, sigma=1.9, e=0.05
m0_base = 4.6
w_base  = 4.
f_base  = 0.01 # Assuming f0 = 250/polyp, r=2e-4, retention=10%
V_base  = 0.05

# PARAMETER RANGES
dw = 2
_var_params = {'w': np.linspace(w_base-dw, w_base+dw, num=n_param),
               'V': np.logspace(-1, 1, num=n_param)*V_base,
               'f': np.logspace(-1, 1, num=n_param)*f_base,
               'r0': np.logspace(-1, 1, num=n_param)*r0_base}
_perms = np.meshgrid(*[_var_params[var] for var in _var_params.keys()], indexing='ij')
var_params = {var: _perms[i].flatten() for i, var in enumerate(_var_params.keys())}

# PREPROCESS CLIMATE DATA
data = xr.open_dataset(data_dir)
n_sites = len(data.lon)
data = data.assign_coords(time=date_range(start=datetime(year=2000, month=1, day=1),
                                                periods=len(data.ocean_time), freq='1D'))
data = data.drop_vars('ocean_time')
data = data.resample(time='1ME').mean(dim='time')

# Remove the first two years (spin-up)
data = data.where(data.time.dt.year >= 2002, drop=True)
data['lat'] = data.lat[0, :].drop_vars('time')
data['lon'] = data.lon[0, :].drop_vars('time')
data = data.assign_coords({'pts': np.arange(len(data.lon))})

# Create a seasonal climatology for the first ten years
data_monclim = data.temp[:120, :].groupby(data.time[:120].dt.month).mean()

# Extract amplitude of the seasonal cycle
data_seas = data_monclim.max(dim='month') - data_monclim.min(dim='month')

# Repeat for required duration to create spin-up temperature time-series
T_spin_up = np.tile(data_monclim.data, reps=[years_su, 1])

# Create full temperature time-series
T_full = data.temp.data

# CREATE OUTPUT ARRAY
shape_spin_up = list(T_spin_up.shape) + list(_perms[0].shape)
shape_full = list(T_full.shape) + list(_perms[0].shape)
j_spin_up = shape_spin_up[0]
j_full = shape_full[0]

coords_spin_up = {'time': np.arange(j_spin_up)/12,
                  'site': np.arange(n_sites)}
coords_full = {'time': data.time,
               'site': np.arange(n_sites)}
for var in _var_params.keys():
    coords_spin_up[var] = _var_params[var]
    coords_full[var] = _var_params[var]
    
dims = ['time', 'site'] + list(_var_params.keys())

if keep_su:
    output_su = xr.Dataset(data_vars = {'c': (dims, np.zeros(shape_spin_up, dtype=np.float32)),
                                        'z': (dims, np.zeros(shape_spin_up, dtype=np.float32)),
                                        'sst': (['time', 'site'], np.zeros(T_spin_up.shape ,dtype=np.float32))},
                           coords=coords_spin_up)
    output_su['sst'].data = T_spin_up

output = xr.Dataset(data_vars = {'c': (dims, np.zeros(shape_full, dtype=np.float32)),
                                 'z': (dims, np.zeros(shape_full, dtype=np.float32)),
                                 'sst': (['time', 'site'], np.zeros(T_full.shape ,dtype=np.float32))},
                    coords=coords_full)
output['sst'].data = T_full







_shape = list(data_monclim.shape) + [1 for item in range(len(_perms[0].shape)-1)]
shape = list(data_monclim.shape) + list(_perms[0].shape)[1:]
tiles = [int(i/j) for i, j in zip(shape, _shape)]
T_spin_up = np.tile(data_monclim.reshape(_shape), tiles)


runs = sites*n_param**4



# Compute number of runs

# SPIN UP SIMULATION
sim = ecoevo.simulation(i=runs, j=data_monclim.shape[0]) # New simulation

# Create boundary conditions with a seasonal cycle
sim.set_bc(T=T_spin_up, I=0.0, zc_offset=0.0) # No immigration

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
