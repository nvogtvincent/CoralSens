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
n_param  = 1000  # Number of parameter values for each parameter
years_su = 50 # Spin-up

# Variable parameters
r0_range = [0.001, 1.0, False] # Min, Max, Log

# We're assuming here that r/V dominate the effects so we don't need to worry about
# other species-specific traits

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
    
_var_params = {'V': np.array([0.01, 0.1, 0.5]),
               'r_0': get_values(r0_range, n_param),
               'w': np.array([4.]),
               'f': np.array([0.01]),
               'I': np.array([0.001]),
               'z_I': np.array([0.]),
               'DHW_{50}': np.array([12.])}
_perms = np.meshgrid(*[_var_params[var] for var in _var_params.keys()], indexing='ij')
_perms = {var: _perms[i] for i, var in enumerate(_var_params.keys())}

# Compute m0 based on actual w
var_params = {var: _perms[var].flatten() for var in _var_params.keys()}
var_params['m0'] = 312.4*((var_params['w']/var_params['DHW_{50}'])**2)
_perms['m0'] = 312.4*((_perms['w']/_perms['DHW_{50}'])**2)

# GET GROWTH RATE DATA FROM CORAL TRAITS DATABASE
traits = pd.read_csv('../data/traits/growth_rate.csv', usecols=['species_name', 'trait_name', 'standard_unit', 'value'])
traits = traits[traits['trait_name'] == 'Growth rate']
traits = traits[traits['standard_unit'].isin(['cm month^-1', 'cm yr^-1', 'mm month^-1', 'mm yr^-1'])]
traits['value'] = traits['value'].astype(float)
traits = traits[traits['value'] > 0.0]
traits.loc[traits['standard_unit'] == 'cm month^-1', 'value'] = traits['value'][traits['standard_unit'] == 'cm month^-1']*0.12 # Convert cm/month to m/yr
traits.loc[traits['standard_unit'] == 'cm yr^-1', 'value'] = traits['value'][traits['standard_unit'] == 'cm yr^-1']*0.01 # Convert cm/yr to m/yr
traits.loc[traits['standard_unit'] == 'mm month^-1', 'value'] = traits['value'][traits['standard_unit'] == 'mm month^-1']*0.012 # Convert mm/month to m/yr
traits.loc[traits['standard_unit'] == 'mm yr^-1', 'value'] = traits['value'][traits['standard_unit'] == 'mm yr^-1']*0.001 # Convert mm/yr to m/yr

# Convert to r0
traits['value'] = traits['value']*5.08 # (assuming typical size structure from Meesters et al. 2001)

# Get percentiles for each species
count = traits.groupby(['species_name'])['value'].count() >= 3
g05 = traits.groupby(['species_name'])['value'].quantile(0.05)[count]
g50 = traits.groupby(['species_name'])['value'].quantile(0.50)[count]
g95 = traits.groupby(['species_name'])['value'].quantile(0.95)[count]

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

# PLOTS
species_list = ['Montipora capitata', 'Montipora digitata', 'Porites compressa',
                'Porites lobata', 'Acropora abrotanoides', 'Acropora aspera',
                'Acropora cervicornis', 'Poccillopora damicornis',
                'Poccillopora meandrina', 'Stylophora pistilata', 'Diploria labyrinthiformis',
                'Sidastrea radians', 'Sidastrea siderea', 'Pavona gigantea',
                'Pavona clavus', 'Pleuractis granulosa', 'Orbicella faveolata']
species_list = [item for item in species_list if item in g50.index]

output = np.zeros((len(species_list), len(_var_params['V']), 3), dtype=np.float32)
output = xr.DataArray(data=output, dims=['species', 'V', 'bound'],
                      coords={'species': (['species'], species_list),
                              'V': _var_params['V'],
                              'bound': ['lower', 'median', 'upper']})

for species in species_list:
    for V in _var_params['V']:
        output.loc[species, V, 'lower'] = c_rel.loc[V][:, 0, 0, 0, 0, 0].interp(r_0=g05[species])
        output.loc[species, V, 'median'] = c_rel.loc[V][:, 0, 0, 0, 0, 0].interp(r_0=g50[species])
        output.loc[species, V, 'upper'] = c_rel.loc[V][:, 0, 0, 0, 0, 0].interp(r_0=g95[species])
        
# Sort by median
species_list_sorted = np.array(species_list)[np.argsort(output.loc[:, 0.1, 'median'].data)[::-1]]

# Plotting
bw = 1 # Bar width
bs = 0.1 # Bar spacing
ss = 1 # Species spacing
cdict = {0.01: 'orangered', 0.1: 'darkorange', 0.5: 'green'}

f, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(5, 5))

labelpos = []
xpos = 0.0
for species in species_list_sorted:
    for V in _var_params['V']:
        height = output.loc[species, V, 'upper'].data - output.loc[species, V, 'lower'].data 
        bottom = output.loc[species, V, 'lower'].data 
        if species == species_list_sorted[0]:
            ax.bar(xpos, height=height, bottom=bottom, width=bw, color=cdict[V],
                   label=str(V)+' C$^2$')
        else:
            ax.bar(xpos, height=height, bottom=bottom, width=bw, color=cdict[V])
        ax.scatter(xpos, output.loc[species, V, 'median'].data, c='k', marker='.', s=50)
        if V == 0.1:
            labelpos.append(xpos)
        xpos += (bw + bs)
    xpos += ss
ax.legend(frameon=False, title='$V$')
ax.set_xticks(labelpos)
ax.set_xticklabels(species_list_sorted, rotation=-60, ha='left', fontsize=7, style='italic')
ax.tick_params(axis='x', pad=0)
ax.set_ylabel('Maximum decline in coral cover (relative %)', fontsize=10)
ax.tick_params(axis='y', labelsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('figures/species_decline_' + scenario +'.pdf', bbox_inches='tight')