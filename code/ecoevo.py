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

    def __init__(self, name='simulation', sites=1, dt=1):

        # Initialise simulation
        self.params = {}
        self.name = name

        # Track simulation status
        self.status = {'params': False,
                       'ic': False,
                       'bc': False}
        
        self.dt = dt
        self.i = int(sites)
        
        print('###################################')
        print('###   Simulation initialised!   ###')
        print('###################################')
        print('')
        
        _plural = '' if self.dt == 1 else 's'
        print('Name: ' + self.name)
        print('Time-step: ' + str(self.dt) + ' month' + _plural)
        print('Number of sites: ' + str(self.dt))
        
        # Pre-define ICs, BCs
        self.bc = None
        self.z0 = None
        self.c0 = None
    
    def set_bc(self, bc=None): 
        '''
        Set the boundary conditions for a simulation. 
           bc: numpy array with values of T
                Either [i * j] or [1 * j] or [j]
        ''' 
        
        # Check input has correct datatype 
        if type(bc) not in [xr.DataArray, np.ndarray]:
            print('Hey')
            raise Exception('Boundary conditions must be a numpy array or xarray DataArray.')
        elif type(bc) == xr.DataArray:
            bc = bc.data
        
        # Get dimensions of array, check if consistent with records
        if bc.ndim == 1:
            self.j = len(bc)
        elif bc.ndim == 2:
            if bc.shape[0] == 1:
                self.j = len(bc)
                bc = bc.flatten()
            else:
                self.j = bc.shape[1]
                if bc.shape[0] != self.i:
                    raise Exception('Expected sites: ' + str(self.i) + '\nProvided sites: ' + str(bc.shape[0]))
        else:
            raise Exception('Boundary conditions must be 1D or 2D.')
        
        # Save boundary conditions
        self.bc = bc.astype(np.float32)
        
        # Return length of simulation
        print('Simulation length: ' + str(int(self.dt*self.j/12)) + ' years')
            
        # Update status
        self.status['bc'] = True
    
    def set_ic(self, z=None, c=None):
        '''
        Set the initial conditions for a simulation. 
            z: numpy array (thermal optimum)
                Either [i] or [i * 1] or scalar
            c: numpy array (coral cover)
                Either [i] or [i * 1] or scalar
        ''' 
        
        # Get dimensions of array, check if consistent with records
        if z is not None:
            if type(z) in [int, float]:
                z = np.ones((self.i,), dtype=np.float32)*z
            elif z.ndim in [1, 2]:
                if z.ndim == 2:
                    assert z.shape[1] == 1
                    z = z.flatten()
                
                if len(z) != self.i:
                    raise Exception('Expected sites: ' + str(self.i) + '\nProvided sites: ' + str(len(z)))
            else:
                raise Exception('Initial conditions must be at most 1D.')
            
            self.z0 = z.astype(np.float32)
        
        if c is not None:
            if type(c) in [int, float]:
                c = np.ones((self.i,), dtype=np.float32)*c
            elif c.ndim in [1, 2]:
                if c.ndim == 2:
                    assert c.shape[1] == 1
                    c = c.flatten()
                
                if len(c) != self.i:
                    raise Exception('Expected sites: ' + str(self.i) + '\nProvided sites: ' + str(len(c)))
            else:
                raise Exception('Initial conditions must be at most 1D.')
            
            self.c0 = c.astype(np.float32)
        
        if self.c0 is not None and self.z0 is not None:
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
                    
        
        
        
    def run(self, output_dt=None):
        print()