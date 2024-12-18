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
n_param  = 5  # Number of parameter values for each parameter
years_su = 100 # Spin-up

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
n_runs = len(var_params[list(_var_params.keys())[0]])
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
T_spin_up = np.tile(data_monclim.data, reps=[years_su, 1]).T

# Create full temperature time-series
T_full = data.temp.data.T

# CREATE OUTPUT ARRAY
shape_spin_up = [T_spin_up.shape[0]] + list(_perms[0].shape) + [T_spin_up.shape[-1]]
shape_full = [T_full.shape[0]] + list(_perms[0].shape) + [T_full.shape[-1]]
j_spin_up = shape_spin_up[-1]
j_full = shape_full[-1]

coords_spin_up = {'site': np.arange(n_sites),
                  'time': np.arange(j_spin_up)/12}
coords_full = {'site': np.arange(n_sites),
               'time': data.time}
for var in _var_params.keys():
    coords_spin_up[var] = _var_params[var]
    coords_full[var] = _var_params[var]
    
dims = ['site'] + list(_var_params.keys()) + ['time']

output_su = xr.Dataset(data_vars = {'c': (dims, np.zeros(shape_spin_up, dtype=np.float32)),
                                    'z': (dims, np.zeros(shape_spin_up, dtype=np.float32)),
                                    'sst': (['site', 'time'], np.zeros(T_spin_up.shape ,dtype=np.float32))},
                       coords=coords_spin_up)
output_su['sst'].data = T_spin_up

output = xr.Dataset(data_vars = {'c': (dims, np.zeros(shape_full, dtype=np.float32)),
                                 'z': (dims, np.zeros(shape_full, dtype=np.float32)),
                                 'sst': (['site', 'time'], np.zeros(T_full.shape ,dtype=np.float32))},
                    coords=coords_full)
output['sst'].data = T_full

## RUN SIMULATIONS
for site in output.site.data:
    print('Simulating site ' + str(site+1) + '/' + str(n_sites))

    # SPIN UP SIMULATION
    sim = ecoevo.simulation(i=n_runs, j=j_spin_up) # New simulation

    # Create boundary conditions with a seasonal cycle
    sim.set_bc(T=output_su.sst.loc[site], I=0.0, zc_offset=0.0) # No immigration

    # Create initial conditions
    sim.set_ic(z=data_monclim.max(dim='month').loc[site], c=0.50)

    # Set parameters
    sim.set_param(r0=var_params['r0'], m0=m0_base, w=var_params['w'],
                  f=var_params['f'], V=var_params['V'], cmin=0.001)

    # Run simulation
    sim.run(output_dt=1)
    init_c = sim.output.c[:, -1].data
    init_z = sim.output.z[:, -1].data
    
    # Unpack output
    for i, var in enumerate(list(_var_params.keys())):
        assert np.array_equal(_perms[i], var_params[var].reshape(_perms[i].shape))
        output_shape = list(_perms[i].shape) + [j_spin_up]
        
    output_su['c'].data[site] = sim.output.c[:, 1:].data.reshape(output_shape)
    output_su['z'].data[site] = sim.output.z[:, 1:].data.reshape(output_shape)
    
    # FUTURE SIMULATION
    sim = ecoevo.simulation(i=n_runs, j=j_full) # New simulation

    # Create boundary conditions with a seasonal cycle
    sim.set_bc(T=output.sst.loc[site], I=0.0, zc_offset=0.0) # No immigration

    # Create initial conditions
    sim.set_ic(z=init_z, c=init_c)

    # Set parameters
    sim.set_param(r0=var_params['r0'], m0=m0_base, w=var_params['w'],
                  f=var_params['f'], V=var_params['V'], cmin=0.001)

    # Run simulation
    sim.run(output_dt=1)
    
    # Unpack output
    for i, var in enumerate(list(_var_params.keys())):
        assert np.array_equal(_perms[i], var_params[var].reshape(_perms[i].shape))
        output_shape = list(_perms[i].shape) + [j_full]
        
    output['c'].data[site] = sim.output.c[:, 1:].data.reshape(output_shape)
    output['z'].data[site] = sim.output.z[:, 1:].data.reshape(output_shape)
    

# # ANALYSES
# # Compute change in coral cover
# rel_change = (sim.output.c.min(dim='time')-sim.output.c[:, 0])/sim.output.c[:, 0]

# # Construct an OLS least-squares model to identify key parameters
# df = pd.DataFrame(data={'decline': rel_change, 'w': w, 'V': V})
# endog = df['decline']
# exog = sm.add_constant(df[['w', 'V']])
# ols = sm.OLS(endog=endog, exog=exog).fit()

# # Plot
# _decline = df['decline'].values.reshape((n_param, n_param))
# _w = df['w'].values.reshape((n_param, n_param))
# _V = df['V'].values.reshape((n_param, n_param))

# plt.close()
# f, ax = plt.subplots(1, 1, figsize=(6, 6))

# cplot = ax.contourf(_w, _V, _decline, levels=np.linspace(-1, 0, num=11),
#                     cmap=cmr.lavender, vmin=-1, vmax=0)
# ax.set_xlabel('Thermal tolerance (C)')
# ax.set_ylabel('Additive genetic variance (C$^2$)')
# #ax.set_yscale('log')

# plt.colorbar(cplot, label='Relative change in coral cover after ' + str(dT) + 'C warming')
