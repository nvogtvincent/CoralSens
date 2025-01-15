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
scenario = '126'
data_dir = '../data/ocean/SSP' + scenario + '.nc'

# BASE PARAMETERS
n_param  = 100  # Number of parameter values for each parameter
years_su = 50 # Spin-up

r0_base = 0.1 # Based on mu=-4.2, sigma=1.9, e=0.05
w_base  = 5
f_base  = 0.01 # Assuming f0 = 250/polyp, r=2e-4, retention=10%
V_base  = 0.1
I_base  = 0.01
DHW_base  = 12
m0_base = 312.4*(w_base/DHW_base)**2

# PARAMETER RANGES
dw = 3
dDHW = 10
_var_params = {'V': np.logspace(-1, 1, num=n_param)*V_base,
               'r0': np.logspace(-1, 1, num=n_param)*r0_base,
               'zc': np.array([0., 0.1, 0.5]),
               'I': np.array([0., 0.01, 0.1])}

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
    sim.set_bc(T=output_su.sst.loc[site], I=var_params['I'], zc_offset=var_params['zc'])

    # Create initial conditions
    sim.set_ic(z=data_monclim.max(dim='month').loc[site], c=1.0)

    # Set parameters
    sim.set_param(r0=var_params['r0'], m0=m0_base, w=w_base,
                  f=f_base, V=var_params['V'], cmin=0.001)

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
    for i, var in enumerate(list(_var_params.keys())):
        assert np.array_equal(_perms[i], var_params[var].reshape(_perms[i].shape))
        output_shape = list(_perms[i].shape) + [j_spin_up]
        
    output_su['c'].data[site] = sim.output.c[:, 1:].data.reshape(output_shape)
    output_su['z'].data[site] = sim.output.z[:, 1:].data.reshape(output_shape)
    
    # FUTURE SIMULATION
    sim = ecoevo.simulation(i=n_runs, j=j_full) # New simulation

    # Create boundary conditions with a seasonal cycle
    sim.set_bc(T=output.sst.loc[site], I=var_params['I'], zc_offset=var_params['zc']) 

    # Create initial conditions
    sim.set_ic(z=init_z, c=init_c)

    # Set parameters
    sim.set_param(r0=var_params['r0'], m0=m0_base, w=w_base,
                  f=f_base, V=var_params['V'], cmin=0.001)

    # Run simulation
    sim.run(output_dt=1)
    
    # Unpack output
    for i, var in enumerate(list(_var_params.keys())):
        assert np.array_equal(_perms[i], var_params[var].reshape(_perms[i].shape))
        output_shape = list(_perms[i].shape) + [j_full]
        
    output['c'].data[site] = sim.output.c[:, 1:].data.reshape(output_shape)
    output['z'].data[site] = sim.output.z[:, 1:].data.reshape(output_shape)
    
    print('')

print('')
print('Simulations complete.')    

# ANALYSES
# Compute change from 2002-2011 to 2091-2100
c_annual = output.c.resample(time='1YE').mean()
c_start = c_annual.where((c_annual.time.dt.year >= 2002)*(c_annual.time.dt.year <= 2011)).mean(dim='time')
c_end = c_annual.where((c_annual.time.dt.year >= 2091)*(c_annual.time.dt.year <= 2100)).mean(dim='time')
c_rel = 100*(c_end - c_start)/c_start
c_rel = c_rel.rename('c_rel')
c_mean = c_rel.mean(dim='site')

# SURFACE PLOTS
f = plt.figure(constrained_layout=True, figsize=(7, 7.2))
gs = GridSpec(4, 3, figure=f, height_ratios=[1, 1, 1, 0.08])
ax = []
cax = []

for i in range(3):
    for j in range(3):
        if not (i in [0, 2] and j == 0):
            ax.append(f.add_subplot(gs[i, j]))
            c_mean = c_rel.mean(dim='site')
            cplot = ax[-1].contourf(c_mean.V, c_mean.r0, c_mean[:, :, i, j].T, levels=np.linspace(-100, 0, num=26),
                                    cmap=cmr.sunburst, vmin=-100, vmax=0)
            
            if i == 2 and j == 1:
                ax[-1].set_xlabel('Additive genetic variance (K$^2$)', fontsize=12)
                
            if j == 0 and i == 1:
                ax[-1].set_ylabel('Growth rate (y$^{-1}$)', fontsize=12)
            ax[-1].set_xscale('log')
            ax[-1].set_yscale('log')
            
            if i == 0 and j == 1:
                ax[-1].set_title({'126': 'SSP1-2.6', '245': 'SSP2-4.5', '370': 'SSP3-7.0'}[scenario], fontsize=14)
            
            # Add labels
            im_text = {0: 'No immigration', 0.01: 'Low immigration', 0.1: 'High immigration'}[float(c_mean[:, :, i, j].I)]
            zc_text = r'$z_c=+$' + str(float(c_mean[:, :, i, j].zc)) + 'C'
            textcolor = 'w' if scenario == '370' else 'k'
            ax[-1].text(0.95, 0.95, im_text, ha='right', va='top', fontsize=10, transform=ax[-1].transAxes, c=textcolor)
            if j > 0:
                ax[-1].text(0.95, 0.85, zc_text, ha='right', va='top', fontsize=10, transform=ax[-1].transAxes, c=textcolor)

cax = f.add_subplot(gs[-1, :])
plt.colorbar(cplot, cax=cax, orientation='horizontal')
cax.tick_params(axis='x', labelsize=10)
cax.set_xticks([-100, -75, -50, -25, 0])
cax.set_xlabel('Relative coral cover change over 21st century (%)', fontsize=12)

plt.savefig('figures/surf_' + scenario +'.pdf', bbox_inches='tight')

# Recovery potential (high immigration / no immigration)
f = plt.figure(figsize=(5, 8))
gs = GridSpec(5, 2, figure=f, height_ratios=[1, 1, 1, 0.06, 0.08], wspace=0.3, hspace=0.3)
ax = []
cax = []

for i in range(3):
    for j in range(2):
        c_pot = 100*(c_end[:, :, :, i, j+1] - c_end[:, :, :, i, 0]).mean(dim='site')
        
        ax.append(f.add_subplot(gs[i, j]))
        c_mean = c_rel.mean(dim='site')
        cplot = ax[-1].contourf(c_mean.V, c_mean.r0, c_pot.T, levels=np.logspace(-1, 2, num=31),
                                cmap=cmr.dusk, locator=ticker.LogLocator(), extend='min')
        
        if i == 2 and j == 0:
            ax[-1].set_xlabel('Additive genetic variance (K$^2$)', fontsize=10)
            ax[-1].xaxis.set_label_coords(1.2, -0.2)
            
        if i == 1 and j == 0:
            ax[-1].set_ylabel('Growth rate (y$^{-1}$)', fontsize=10)
        
        if i == 0 and j == 0:
            ax[-1].set_title({'126': 'SSP1-2.6', '245': 'SSP2-4.5', '370': 'SSP3-7.0'}[scenario], fontsize=12,
                             x=1.1, y=1.05)
            
        ax[-1].set_xscale('log')
        ax[-1].set_yscale('log')

        # Add labels
        im_text = {0: 'No immigration', 0.01: 'Low immigration', 0.1: 'High immigration'}[float(c_end[:, :, :, i, j+1].I)]
        zc_text = r'$z_c=+$' + str(float(c_mean[:, :, i, j].zc)) + 'C'
        textcolor = 'w' 
        ax[-1].text(0.95, 0.95, im_text, ha='right', va='top', fontsize=10, transform=ax[-1].transAxes, c=textcolor)
        ax[-1].text(0.95, 0.85, zc_text, ha='right', va='top', fontsize=10, transform=ax[-1].transAxes, c=textcolor)

cax = f.add_subplot(gs[-1, :])
plt.colorbar(cplot, cax=cax, orientation='horizontal')
cax.tick_params(axis='x', labelsize=10)
cax.set_xticks([0.1, 1, 10, 100])
cax.set_xlabel('Effect of immigration on 2100 coral cover (%)', fontsize=12)
plt.savefig('figures/im_eff_' + scenario +'.pdf', dpi=400, bbox_inches='tight')
