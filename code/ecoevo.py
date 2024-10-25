import numpy as np
import xarray as xr
import numexpr as ne
import pandas as pd

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
        self.bc = None
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
            
    
    def set_bc(self, T=None, I=None): 
        '''
        Set the boundary conditions for a simulation. 
            T: numpy array with values of T
                 Either [i * j] or [1 * j] or [j]
            I: numpy array with values of I
                Either [i * j] or [1 * j] or [j]
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

        # Update status
        if hasattr(self, 'T') and hasattr(self, 'I'):
            self.status['bc'] = True
    
    def set_ic(self, z=None, c=None):
        '''
        Set the initial conditions for a simulation. 
            z: numpy array (thermal optimum)
                Either [i] or [i * 1] or scalar
            c: numpy array (coral cover)
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
            r0, w, beta, V, cmin
        '''        
        
        _permitted_kwargs = ['r0', 'w', 'beta', 'V', 'cmin']
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
        _template = np.zeros((self.i, len(_time)), dtype=np.float32)
        self.output = xr.Dataset(data_vars={'c': (['site', 'time'], _template),
                                            'z': (['site', 'time'], _template)},
                                 coords={'site': ('site', _site),
                                         'time': ('time', _time, {'units': 'years'})})
        
        if params:
            param_list = ['r0', 'w', 'beta', 'V', 'cmin']
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