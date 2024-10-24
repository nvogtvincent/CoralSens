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
    
    def set_bc(self, arr): 
        '''
        Set the boundary conditions for a simulation. 
           arr: numpy array or xarray DataArray
                Either [i * j] or [1 * j] or [j]
        ''' 
        
        # Check input has correct datatype 
        if type(arr) not in [xr.DataArray, np.ndarray]:
            print('Hey')
            raise Exception('Boundary conditions must be a numpy array or xarray DataArray.')
        elif type(arr) == xr.DataArray:
            arr = arr.data
        
        # Get dimensions of array, check if consistent with records
        _i, _j = None, None
        if arr.ndim == 1:
            _j = len(arr)
        elif arr.ndim == 2:
            if arr.shape[0] == 1:
                _j = len(arr)
                arr = arr.flatten()
            else:
                _j = arr.shape[1]
                _i = arr.shape[0]
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
        self.bc = arr
            
        # Update status
        self.status['boundary'] = True
