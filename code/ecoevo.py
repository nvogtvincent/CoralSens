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
                       'initial': False,
                       'boundary': False}
        
        self.dt = dt
        self.i = int(sites)
        
        print('###################################')
        print('###   Simulation initialised!   ###')
        print('###################################')
        print('')
        
        _plural = '' if self.dt == 1 else 's'
        print('Time-step: ' + str(self.dt) + ' month' + _plural)
        print('Number of sites: ' + str(self.dt))
    
    def set_bc(self, bc=None): 
        '''
        Set the boundary conditions for a simulation. 
           bc: numpy array or xarray DataArray with values of T
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
        self.bc = bc
        
        # Return length of simulation
        print('Simulation length: ' + str(int(self.dt*self.j/12)) + ' years')
            
        # Update status
        self.status['boundary'] = True
    
    def set_ic(self, z=None, c=None):
        '''
        Set the initial conditions for a simulation. 
            z: numpy array or xarray DataArray (thermal optimum)
                Either [i] or [i * 1] or scalar
            c: numpy array or xarray DataArray (coral cover)
                Either [i] or [i * 1] or scalar
        ''' 
        
        # Get dimensions of array, check if consistent with records
        if z is not None:
            if type(z) in [int, float]:
                pass
            elif z.ndim in [1, 2]:
                if z.ndim == 2:
                    assert z.shape[1] == 1
                    z = z.flatten()
                
                if len(z) != self.i:
                    raise Exception('Expected sites: ' + str(self.i) + '\nProvided sites: ' + str(len(z)))
            else:
                raise Exception('Initial conditions must be at most 1D.')
            
            self.z0 = z
        
        if c is not None:
            if type(c) in [int, float]:
                pass
            elif c.ndim in [1, 2]:
                if c.ndim == 2:
                    assert c.shape[1] == 1
                    c = c.flatten()
                
                if len(c) != self.i:
                    raise Exception('Expected sites: ' + str(self.i) + '\nProvided sites: ' + str(len(c)))
            else:
                raise Exception('Initial conditions must be at most 1D.')
            
            self.c0 = c
            
