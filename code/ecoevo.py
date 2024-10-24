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

    def __init__(self, name='simulation'):

        # Initialise simulation
        self.params = {}
        self.name = name

        # Track simulation status
        self.status = {'params': False,
                       'initial': False,
                       'boundary': False}
    
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
        _i, _j = None, None
        if bc.ndim == 1:
            _j = len(bc)
        elif bc.ndim == 2:
            if bc.shape[0] == 1:
                _j = len(bc)
                bc = bc.flatten()
            else:
                _j = bc.shape[1]
                _i = bc.shape[0]
        else:
            raise Exception('Boundary conditions must be 1D or 2D.')
        
        if _i is not None:
            if hasattr(self, 'i') and _i != self.i:
                raise Exception('Inconsistent number of sites. Reset the simulation.')
            else:
                self.i = _i
        
        if hasattr(self, 'j') and _j != self.j:
            raise Exception('Inconsistent number of time points. Reset the simulation.')
        else:
            self.j = _j
        
        # Save boundary conditions
        self.bc = bc
            
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
            if z.ndim == 1:
                _i = len(z)
            elif z.ndim == 2:
                if z.shape[1] == 1:
                    _i = len(z)
                    z = z.flatten()
                else:
                    raise Exception('Initial conditions must be at most 1D.')
            else:
                raise Exception('Initial conditions must be at most 1D.')
            
            if hasattr(self, 'i') and _i != self.i:
                raise Exception('Inconsistent number of sites. Reset the simulation.')
            else:
                self.i = _i
            
            self.z0 = z
        
        if c is not None:
            if c.ndim == 1:
                _i = len(c)
            elif c.ndim == 2:
                if c.shape[1] == 1:
                    _i = len(c)
                    c = c.flatten()
                else:
                    raise Exception('Initial conditions must be at most 1D.')
            else:
                raise Exception('Initial conditions must be at most 1D.')
            
            if hasattr(self, 'c') and _c != self.c:
                raise Exception('Inconsistent number of sites. Reset the simulation.')
            else:
                self.c = _c
            
            self.c0 = c
            
