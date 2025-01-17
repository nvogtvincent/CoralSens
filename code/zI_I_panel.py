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
from matplotlib.gridspec import GridSpec
from matplotlib import ticker

# CLIMATE DATA DIRECTORY
scenario = '245'
data_dir = '../data/ocean/SSP' + scenario + '.nc'

# BASE PARAMETERS
# Simulation parameters
n_param  = 50  # Number of parameter values for each parameter
years_su = 50 # Spin-up

# Variable parameters
V_range = [0.01, 0.5, True]
r0_range = [0.02, 0.6, True] # Min, Max, Log

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
               'w': np.array([4.]),
               'f': np.array([0.01]),
               'I': np.array([0., 0.01, 0.1]),
               'z_I': np.array([0., 0.1, 0.5]),
               'DHW_{50}': np.array([12.])}
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
c_start = c_decade[:, :, :, :, :, :, :, :, 0].drop('year')
c_min = c_decade.min(dim='year')
c_rel = 100*(c_min - c_start)/c_start
c_rel = c_rel.rename('c_rel').mean(dim='site')

# SURFACE PLOTS
f = plt.figure(constrained_layout=True, figsize=(7, 7.2))
gs = GridSpec(4, 3, figure=f, height_ratios=[1, 1, 1, 0.08])
ax = []
cax = []

for i in range(3):
    for j in range(3):
        if not (i in [0, 2] and j == 0):
            ax.append(f.add_subplot(gs[i, j]))
            submatrix = c_rel[:, :, 0, 0, j, i, 0].T
            cplot = ax[-1].contourf(submatrix.V, submatrix.r_0, submatrix, levels=np.linspace(-100, 0, num=26),
                                    cmap=cmr.sunburst)
            
            if i == 2 and j == 1:
                ax[-1].set_xlabel('Additive genetic variance (K$^2$)', fontsize=12)
                
            if j == 0 and i == 1:
                ax[-1].set_ylabel('Growth rate (y$^{-1}$)', fontsize=12)
            ax[-1].set_xscale('log')
            ax[-1].set_yscale('log')
            
            if i == 0 and j == 1:
                ax[-1].set_title({'126': 'SSP1-2.6', '245': 'SSP2-4.5', '370': 'SSP3-7.0'}[scenario], fontsize=14, y=1.05)
            
            # Add labels
            im_text = {0: 'No immigration', 0.001: 'Moderate immigration', 0.01: 'High immigration', 0.1: 'Very high immigration'}[float(submatrix.I)]
            zc_text = r'$z_I=+$' + str(float(submatrix.z_I)) + 'C'
            textcolor = 'w' 
            ax[-1].text(0.05, 0.05, im_text, ha='left', va='bottom', fontsize=10, transform=ax[-1].transAxes, c=textcolor)
            if j > 0:
                ax[-1].text(0.05, 0.15, zc_text, ha='left', va='bottom', fontsize=10, transform=ax[-1].transAxes, c=textcolor)

cax = f.add_subplot(gs[-1, :])
plt.colorbar(cplot, cax=cax, orientation='horizontal')
cax.tick_params(axis='x', labelsize=10)
cax.set_xticks([-100, -75, -50, -25, 0])
cax.set_xlabel('Maximum coral cover change', fontsize=12)

plt.savefig('figures/immigration_' + scenario +'.pdf', bbox_inches='tight')

# # Recovery potential (high immigration / no immigration)
# f = plt.figure(figsize=(5, 8))
# gs = GridSpec(5, 2, figure=f, height_ratios=[1, 1, 1, 0.06, 0.08], wspace=0.3, hspace=0.3)
# ax = []
# cax = []

# for i in range(3):
#     for j in range(2):
#         c_pot = 100*(c_end[:, :, :, i, j+1] - c_end[:, :, :, i, 0]).mean(dim='site')
        
#         ax.append(f.add_subplot(gs[i, j]))
#         c_mean = c_rel.mean(dim='site')
#         cplot = ax[-1].contourf(c_mean.V, c_mean.r0, c_pot.T, levels=np.logspace(-1, 2, num=31),
#                                 cmap=cmr.dusk, locator=ticker.LogLocator(), extend='min')
        
#         if i == 2 and j == 0:
#             ax[-1].set_xlabel('Additive genetic variance (K$^2$)', fontsize=10)
#             ax[-1].xaxis.set_label_coords(1.2, -0.2)
            
#         if i == 1 and j == 0:
#             ax[-1].set_ylabel('Growth rate (y$^{-1}$)', fontsize=10)
        
#         if i == 0 and j == 0:
#             ax[-1].set_title({'126': 'SSP1-2.6', '245': 'SSP2-4.5', '370': 'SSP3-7.0'}[scenario], fontsize=12,
#                              x=1.1, y=1.05)
            
#         ax[-1].set_xscale('log')
#         ax[-1].set_yscale('log')

#         # Add labels
#         im_text = {0: 'No immigration', 0.01: 'Low immigration', 0.1: 'High immigration'}[float(c_end[:, :, :, i, j+1].I)]
#         zc_text = r'$z_c=+$' + str(float(c_mean[:, :, i, j].zc)) + 'C'
#         textcolor = 'w' 
#         ax[-1].text(0.95, 0.95, im_text, ha='right', va='top', fontsize=10, transform=ax[-1].transAxes, c=textcolor)
#         ax[-1].text(0.95, 0.85, zc_text, ha='right', va='top', fontsize=10, transform=ax[-1].transAxes, c=textcolor)

# cax = f.add_subplot(gs[-1, :])
# plt.colorbar(cplot, cax=cax, orientation='horizontal')
# cax.tick_params(axis='x', labelsize=10)
# cax.set_xticks([0.1, 1, 10, 100])
# cax.set_xlabel('Effect of immigration on 2100 coral cover (%)', fontsize=12)
# plt.savefig('figures/im_eff_' + scenario +'.pdf', dpi=400, bbox_inches='tight')
