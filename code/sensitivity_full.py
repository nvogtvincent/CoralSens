import numpy as np
import pandas as pd
import ecoevo
import cmasher as cmr
import xarray as xr
import importlib
importlib.reload(ecoevo)
from matplotlib import pyplot as plt
from pandas import date_range
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor 
from sklearn.inspection import permutation_importance

# CLIMATE DATA DIRECTORY
scenario = '370'
data_dir = '../data/ocean/SSP' + scenario + '.nc'

# BASE PARAMETERS
# Simulation parameters
n_param  = 6  # Number of parameter values for each parameter
years_su = 50 # Spin-up

# Variable parameters
V_range = [0.01, 0.5, True]
r0_range = [0.02, 0.6, True] # Min, Max, Log
DHW_range = [4, 24, False]
w_range = [2, 8, False]
f_range = [0.0001, 0.4, True]
I_range = [0.0001, 0.1, True]
zi_range = [-0.5, 0.5, False]

# PARAMETER RANGES
def get_values(parameter_range, num):
    # Return an array with test parameter values for input in the form [min, max, log]
    _min = parameter_range[0]
    _max = parameter_range[1]
    _log = parameter_range[2]
    
    if _log:
        return np.logspace(np.log10(_min), np.log10(_max), num=num)
    else:
        return np.linspace(_min, _max, num=num)
    
_var_params = {'V': get_values(V_range, n_param),
               'r_0': get_values(r0_range, n_param),
               'w': get_values(w_range, n_param),
               'f': get_values(f_range, n_param),
               'I': get_values(I_range, n_param),
               'z_I': get_values(zi_range, n_param),
               'DHW_{50}': get_values(DHW_range, n_param)}
_perms = np.meshgrid(*[_var_params[var] for var in _var_params.keys()], indexing='ij')
_perms = {var: _perms[i] for i, var in enumerate(_var_params.keys())}

# Compute m0 based on actual w
var_params = {var: _perms[var].flatten() for var in _var_params.keys()}
var_params['m0'] = 312.4*((var_params['w']/var_params['DHW_{50}'])**2)
_perms['m0'] = 312.4*((_perms['w']/_perms['DHW_{50}'])**2)


# PREPROCESS CLIMATE DATA
data = xr.open_dataset(data_dir)
n_sites = len(data.lon)
n_runs = len(var_params[list(var_params.keys())[0]])
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
shape_spin_up = [T_spin_up.shape[0]] + list(_perms['V'].shape) + [T_spin_up.shape[-1]]
shape_full = [T_full.shape[0]] + list(_perms['V'].shape) + [T_full.shape[-1]]
j_spin_up = shape_spin_up[-1]
j_full = shape_full[-1]

coords_spin_up = {'site': np.arange(n_sites),
                  'time': pd.date_range(end=datetime(year=int(data.time[0].dt.year-1), month=12, day=31),
                                        periods=j_spin_up, freq='1ME')}
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
    sim.set_bc(T=output_su.sst.loc[site], I=var_params['I'], zc_offset=var_params['z_I'])

    # Create initial conditions
    sim.set_ic(z=data_monclim.max(dim='month').loc[site], c=1.0)

    # Set parameters
    sim.set_param(r0=var_params['r_0'], m0=var_params['m0'], w=var_params['w'],
                  f=var_params['f'], V=var_params['V'], cmin=0.001)

    # Run simulation
    sim.run(output_dt=1)
    init_c = sim.output.c[:, -1].data
    init_z = sim.output.z[:, -1].data
    
    # Assert convergence based on annual means
    _c_annual = sim.output.c.groupby(np.ceil(sim.output.c.time)).mean()
    _z_annual = sim.output.z.groupby(np.ceil(sim.output.c.time)).mean()
    _dc = abs(100*(_c_annual[:, -1] - _c_annual[:, -10])/_c_annual[:, -10])
    _dz = abs(100*(_z_annual[:, -1] - _z_annual[:, -10])/_z_annual[:, -10])
    
    if _dc.quantile(0.99) > 1 or _dz.quantile(0.99) > 1:
        raise Exception('Spin-up has not converged (dc99: ' + str(np.round(float(_dc.quantile(0.99)), 1)) + ', dz99: ' + str(np.round(float(_dz.quantile(0.99)), 1)) + ').')
    
    # Unpack output
    for var in list(_var_params.keys()):
        assert np.array_equal(_perms[var], var_params[var].reshape(_perms[var].shape))
        output_shape = list(_perms[var].shape) + [j_spin_up]
        
    output_su['c'].data[site] = sim.output.c[:, 1:].data.reshape(output_shape)
    output_su['z'].data[site] = sim.output.z[:, 1:].data.reshape(output_shape)
    
    # FUTURE SIMULATION
    sim = ecoevo.simulation(i=n_runs, j=j_full) # New simulation

    # Create boundary conditions with a seasonal cycle
    sim.set_bc(T=output.sst.loc[site], I=var_params['I'], zc_offset=var_params['z_I']) 

    # Create initial conditions
    sim.set_ic(z=init_z, c=init_c)

    # Set parameters
    sim.set_param(r0=var_params['r_0'], m0=var_params['m0'], w=var_params['w'],
                  f=var_params['f'], V=var_params['V'], cmin=0.001)

    # Run simulation
    sim.run(output_dt=1)
    
    # Unpack output
    for var in list(_var_params.keys()):
        assert np.array_equal(_perms[var], var_params[var].reshape(_perms[var].shape))
        output_shape = list(_perms[var].shape) + [j_full]
        
    output['c'].data[site] = sim.output.c[:, 1:].data.reshape(output_shape)
    output['z'].data[site] = sim.output.z[:, 1:].data.reshape(output_shape)
    
    print('')

print('')
print('Simulations complete.')    

# ANALYSES
c_decade = output.c.groupby(np.ceil(output.time.dt.year/10).astype(int)*10).mean()
del output_su
del output
c_start = c_decade[:, :, :, :, :, :, :, :, 0].drop('year')
c_min = c_decade.min(dim='year')
c_rel = 100*(c_min - c_start)/c_start
c_rel = c_rel.rename('c_rel')

# Flatten arrays for regression
c_rel_vec = c_rel.mean(dim='site').stack(param=list(_var_params.keys()))
c_rel_df = c_rel_vec.to_dataframe().reset_index(drop=True)

# Log-transform log-distributed variables
c_rel_df['V'] = np.log10(c_rel_df['V'])
c_rel_df['f'] = np.log10(c_rel_df['f'])
c_rel_df['I'] = np.log10(c_rel_df['I'])
c_rel_df['r_0'] = np.log10(c_rel_df['r_0'])

# RANDOM FOREST REGRESSION
# Prepare X, y variables
X = c_rel_df[list(_var_params.keys())]
y = c_rel_df['c_rel']

# Compute feature importance across 100 permutations
reg = RandomForestRegressor(n_jobs=-1).fit(X, y)
importance = permutation_importance(reg, X, y, n_repeats=10, random_state=999, scoring='r2')

# Plot importance
f, ax = plt.subplots(1, 1, figsize=(5, 4))

n_features = 7
features = list(var_params)[:n_features]
features = [r'$' + feature + '$' for feature in features]

# Assign colours to features
cmap = cmr.take_cmap_colors(cmr.ember, N=len(var_params))

# Sort by importance
order = np.argsort(importance.importances_mean)[::-1]
importance_sorted = importance.importances_mean[order]
colors_sorted = [cmap[i] for i in order]
features_sorted = [features[i] for i in order]
min_sorted = (importance.importances_mean - importance.importances.min(axis=1))[order]
max_sorted = (importance.importances.max(axis=1) - importance.importances_mean)[order]

ax.bar(x=np.arange(n_features), height=importance_sorted, color=colors_sorted,
       width=0.4)
ax.errorbar(x=np.arange(n_features), y=importance_sorted, yerr=[min_sorted, max_sorted],
            ecolor='k', elinewidth=0.5, linestyle='')

ax.set_xticks(np.arange(n_features))
ax.set_xticklabels(features_sorted)
ax.set_xlabel('Parameter')
ax.set_ylabel('Performance degradation')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([0, 1])

# Annotation
scenario_text = 'Emissions: ' + {'126': 'SSP1-2.6', '245': 'SSP2-4.5', '370': 'SSP3-7.0'}[scenario]
ax.text(0.98, 0.98, scenario_text, ha='right', va='top', fontsize=10,
        transform=ax.transAxes)

plt.savefig('figures/sens_' + scenario + '.pdf', bbox_inches='tight')