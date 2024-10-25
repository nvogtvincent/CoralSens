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
                    setattr(self, var_name, var*np.ones((self.i), dtype=np.float32))
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
                    
    def run(self, output_dt=None):
        print()