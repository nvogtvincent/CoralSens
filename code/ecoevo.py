import numpy as np
import xarray as xr
import numexpr as ne
import pandas as pd
from tqdm import tqdm

class simulation:
    '''
    Create a coral eco-evo simulation following McManus et al. 2021
    Evolution reverses the effect of network structure on metapopulation persistence across
    a set of independent populations i, and time-steps j.

    '''

    def __init__(self, name='Simulation', i=1, j=None, dt=1):

        # Initialise simulation
        self.params = {}
        self.name = name

        # Track simulation status
        self.status = {'params': False,
                       'ic': False,
                       'bc': False}
        
        self.dt = dt
        self.i = int(i)
        self.j = int(j)   
        
        print('###################################')
        print('###   Simulation initialised!   ###')
        print('###################################')
        print('')
        
        _plural = '' if self.dt == 1 else 's'
        print('Name: ' + self.name)
        print('Time-step: ' + str(self.dt) + ' month' + _plural)
        print('Number of sites: ' + str(self.i))
        print('Number of time-steps: ' + str(self.j))
        print('Simulation time-span: ' + str(int(self.dt*self.j/12)) + ' years')
        
        # Pre-define ICs, BCs
        self.I = None
        self.T = None
        self.zc_offset = 0.
        self.z0 = None
        self.c0 = None
            
    def check_dim(self, dim, val):
        '''
        Check if dimension dim has a value val. Returns None if
        it does (set value if it does not exist), exception otherwise.
        '''
        
        if not hasattr(self, dim):
            raise Exception('Dimension ' + dim + ' does not exist.')
        else:
            if getattr(self, dim) == val:
                return None
            else:
                raise Exception('Dimension ' + dim + ' is incompatible.')
            
    
    def set_bc(self, T=None, I=None, zc_offset=None): 
        '''
        Set the boundary conditions for a simulation. 
            T: numpy array with values of T (temperature)            Units: degC
                 Either [i * j] or [1 * j] or [j]
            I: numpy array with values of I (immigrant larval flux)  Units: larvae m-2
                Either [i * j] or [1 * j] or [j]
            zc: None/numeric (thermal optimum of immigrant larvae), one of the following options:
                None    : zc = z
                numeric : zc = z + constant                          Units: degC
        ''' 
        
        for var_name, var in zip(['T', 'I'], [T, I]):
            if var is not None:
                if type(var) in [int, float]:
                    setattr(self, var_name, var*np.ones((self.j), dtype=np.float32))
                elif var.ndim == 1:
                    self.check_dim('j', len(var))
                    setattr(self, var_name, var.flatten().astype(np.float32))
                elif var.ndim == 2:
                    self.check_dim('i', var.shape[0])
                    self.check_dim('j', var.shape[1])
                    setattr(self, var_name, var.astype(np.float32))
                else:
                    raise Exception('Boundary conditions must be 1D or 2D.')
        
        if zc_offset is not None:
            assert type(zc_offset) in [float, int]
            self.zc_offset = zc_offset    

        # Update status
        if hasattr(self, 'T') and hasattr(self, 'I') and hasattr(self, 'zc_offset'):
            self.status['bc'] = True
    
    def set_ic(self, z=None, c=None):
        '''
        Set the initial conditions for a simulation. 
            z: numpy array (thermal optimum)           Units: degC
                Either [i] or [i * 1] or scalar
            c: numpy array (coral cover)               Units: fraction
                Either [i] or [i * 1] or scalar
        ''' 
        
        for var_name, var in zip(['z0', 'c0'], [z, c]):
            if var is not None:
                if type(var) in [int, float]:
                    setattr(self, var_name, var*np.ones((self.i), dtype=np.float32))
                elif var.ndim == 1:
                    self.check_dim('i', len(var))
                    setattr(self, var_name, var.flatten().astype(np.float32))
                elif var.ndim == 2:
                    self.check_dim('i', var.shape[0])
                    assert var.shape[1] == 1
                    setattr(self, var_name, var.flatten().astype(np.float32))
                else:
                    raise Exception('Boundary conditions must be 1D or 2D.')

        # Update status
        if hasattr(self, 'z0') and hasattr(self, 'c0'):
            self.status['ic'] = True

    def set_param(self, **kwargs):
        '''
        Set parameters for the simulation.
            kwarg: numpy array  
                Either [i] or scalar
            
        Permitted kwargs:
            r0, w, f, V, cmin
            
            r0 : Growth rate parameter (K y-1)
            w  : Thermal tolerance breadth (K)
            f  : Self-recruitment (no units)
            V  : Additive genetic variance (K2)
            cmin : Population szize throttling threshold
        '''        
        
        _permitted_kwargs = ['r0', 'w', 'f', 'V', 'cmin']
        for kwarg in kwargs:
            if kwarg in _permitted_kwargs:
                if type(kwargs[kwarg]) in [float, int]:
                    setattr(self, kwarg, kwargs[kwarg]*np.ones((self.i,), dtype=np.float32))
                else:
                    if kwargs[kwarg].ndim == 2:
                        assert kwargs[kwarg].shape[1] == 1
                        kwargs[kwarg] = kwargs[kwarg].flatten()
                        setattr(self, kwarg, kwargs[kwarg].astype(np.float32))
                    elif kwargs[kwarg].ndim == 1:
                        setattr(self, kwarg, kwargs[kwarg].astype(np.float32))
                    else:
                        raise Exception('Parameter ' + kwarg + ' must be at most 1D.')
                    
                    if len(getattr(self, kwarg)) != self.i:
                        raise Exception('Expected sites: ' + str(self.i) + '\nProvided sites: ' + str(len(self.kwarg)))
                    
        # Check if all parameters have been assigned
        _check = 1
        for kwarg in _permitted_kwargs:
            if not hasattr(self, kwarg):
                _check *= 0
        
        if _check == 1:
            self.status['params'] = True
                    
    def run(self, output_dt=None, params=True, bc=True):
        '''
        Run a simulation.
            output_dt: None - output final state only
                       int  - output every *int* time-steps
            params:    Bool - Include parameters in output
            bc    :    Bool - Include boundary conditions in output
        '''
        
        if not (self.status['params'] and self.status['bc'] and self.status['ic']):
            raise Exception('Simulation is not fully configured.')
        
        # Create the time axis (in years)
        if output_dt is None:
            _time = np.array([0, self.j*self.dt/12], dtype=np.float32)
            _t_idx = [0, -1]
        else:
            if output_dt%self.dt != 0:
                raise Exception('Output frequency must be multiple of time-step.')
            if (self.j*self.dt)%output_dt != 0:
                raise Exception('Output frequency must be a factor of the number of time-steps.')
            if output_dt < self.dt:
                raise Exception('Output frequency cannot be less than time-step.')
                
            _time = np.arange(0, (self.j+1)*self.dt, output_dt, dtype=np.float32)/12
            
            # Time indices to extract for export
            _t_idx = np.arange(output_dt, self.j+int(output_dt/self.dt), int(output_dt/self.dt), dtype=int) - 1
            
        # Create the site axis
        _site = np.arange(self.i, dtype=np.int32)
        
        # Create the dataset
        self.output = xr.Dataset(data_vars={'c': (['site', 'time'], np.zeros((self.i, len(_time)), dtype=np.float32)),
                                            'z': (['site', 'time'], np.zeros((self.i, len(_time)), dtype=np.float32))},
                                 coords={'site': ('site', _site),
                                         'time': ('time', _time, {'units': 'years'})})
        
        # Store initial conditions to output
        self.output.c[:, 0] = self.c0
        self.output.z[:, 0] = self.z0
        _c, _z = self.c0, self.z0
        
        if params:
            param_list = ['r0', 'w', 'f', 'V', 'cmin']
            for param in param_list:
                self.output[param] = xr.DataArray(data=getattr(self, param),
                                                  dims=['site'], coords={'site': (['site'], _site)})
        
        if bc:
            bc_list = ['T', 'I']
            for _bc in bc_list:
                if getattr(self, _bc).ndim == 1:
                    bc_subset = np.concatenate([np.nan*np.ones((1,), dtype=np.float32), getattr(self, _bc)[_t_idx]])
                    self.output[_bc] = xr.DataArray(data=bc_subset,
                                                    dims=['time'], coords={'time': (['time'], _time, {'units': 'years'})})
                else:
                    bc_subset = np.concatenate([np.nan*np.ones((self.i, 1), dtype=np.float32), getattr(self, _bc)[:, _t_idx]], axis=0)
                    self.output[_bc] = xr.DataArray(data=bc_subset,
                                                    dims=['site', 'time'],
                                                    coords={'site': (['site'], _site),
                                                            'time': (['time'], _time, {'units': 'years'})})
        
        # Start integration loop
        print('')
        print('Starting simulation...')
        
        _write_line = 1 # Which line in output to write
        
        with tqdm(total=self.j/12, unit=' years') as progress:
            for it in range(self.j):
                
                # Get current month
                month = it%12
                
                # Get boundary conditions
                _T = self.T[it] if self.T.ndim == 1 else self.T[:, it]
                _I = self.I[it] if self.I.ndim == 1 else self.I[:, it]
                
                # Iterate
                _c, _z = self.forward(c=_c, z=_z, T=_T, I=_I, zc_offset=self.zc_offset, dt=self.dt/12) # Converting dt to years
                
                # Error checking
                if np.any(_c > 1):
                    raise Exception('Coral cover exceeds 100% in step ' + str(it) + '!')
                
                # Save to output
                if it in _t_idx:
                    self.output.c[:, _write_line] = _c
                    self.output.z[:, _write_line] = _z
                    _write_line += 1                
                
                # Update progress bar
                if month == 11:
                    progress.update(1)
    
    def forward(self, c=None, z=None, T=None, I=None, zc_offset=None, dt=None):
        '''
        Perform one integration of the eco-evo time-stepping scheme.
        c, z, T, I: 1D consistent numpy arrays
        zc_offset : numeric (C)
        dt        : numeric  (years) 
        '''
        
        # Compute immigration
        c = 1 - (1 - c)*np.exp(-(self.f*c + I))
        
        
        
        return c, z