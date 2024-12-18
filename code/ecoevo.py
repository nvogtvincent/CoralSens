import numpy as np
import xarray as xr
from tqdm import tqdm

class simulation:
    '''
    Create a coral eco-evo simulation following McManus et al. 2021
    Evolution reverses the effect of network structure on metapopulation persistence across
    a set of independent populations i, and time-steps j.

    '''

    def __init__(self, name='Simulation', i=1, j=None):

        # Initialise simulation
        self.params = {}
        self.name = name

        # Track simulation status
        self.status = {'params': False,
                       'ic': False,
                       'bc': False}
        
        self.dt = 1
        self.i = int(i)
        self.j = int(j)   
        self.spawning_months = [0]
        
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
                if type(var) == xr.DataArray:
                    var = var.data
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
                if type(var) == xr.DataArray:
                    var = var.data
                    var = var.item() if var.shape == () else var
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
                    raise Exception('Initial conditions must be 0D, 1D or 2D.')

        # Update status
        if hasattr(self, 'z0') and hasattr(self, 'c0'):
            self.status['ic'] = True

    def set_param(self, **kwargs):
        '''
        Set parameters for the simulation.
            kwarg: numpy array  
                Either [i] or scalar
            
        Permitted kwargs:
            r0, m0, w, f, V, cmin
            
            r0 : Growth rate parameter (y-1)
            m0 : Mortality rate parameter (m-1)
            w  : Thermal tolerance breadth (K)
            f  : Self-recruitment (no units)
            V  : Additive genetic variance (K2)
            cmin : Population szize throttling threshold
        '''        
        
        _permitted_kwargs = ['r0', 'm0', 'w', 'f', 'V', 'cmin']
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
    
    def dcdt(self, c, z, T, core):
        '''
        Compute dc/dt (for RK4 loop)
        '''
        _m0 = np.where(T > z, self.m0, 0.) 
        supp = self.r0*(1-c) + _m0 # Repeated supplementary term
        g = core*supp - _m0
        gzz = core*supp*(1/(self.w**2))*(((T-z)**2)/(self.w**2) - 1)
        
        return g*c + 0.5*self.V*gzz*c
    
    def dzdt(self, c, z, T, q):
        '''
        Compute dz/dt (for RK4 loop)
        '''
        
        _m0 = np.where(T > z, self.m0, 0.) 
        core = np.exp(-((T-z)**2)/(2*(self.w**2)))
        supp = self.r0*(1-c) + _m0 # Repeated supplementary term
        gz = core*supp*((T-z)/(self.w**2))
        
        return q*self.V*gz
    
    def forward(self, c=None, z=None, T=None, I=None, zc_offset=None, dt=None, spawning=None):
        '''
        Perform one integration of the eco-evo time-stepping scheme.
        c, z, T, I: 1D consistent numpy arrays
        zc_offset : numeric (C)
        dt        : numeric  (years) 
        '''
              
        # Compute immigration 
        if spawning:
            _incoming = self.f*c + I
            _incoming_safe = np.copy(_incoming)
            _incoming_safe[_incoming_safe == 0] = 1. # Avoid division by zero
                
            dc = 1 - (1 - c)*np.exp(-_incoming) - c  # Change in coral cover 
            zc = (z*self.f*c + (z + zc_offset)*I)/_incoming_safe # Mean thermal optimum among immigrants
            np.divide(c*z + dc*zc, c + dc, out=z, where=(c + dc > 0))
            c = c + dc                                        # New coral cover
        
        # RK4 LOOPS        
        # Compute population growth common (core) term
        core = np.exp(-((T-z)**2)/(2*(self.w**2)))
        
        # RK4 terms (dc/dt)
        k1 = (self.dt/12)*self.dcdt(c, z, T, core)
        k2 = (self.dt/12)*self.dcdt(c+0.5*k1, z, T, core)
        k3 = (self.dt/12)*self.dcdt(c+0.5*k2, z, T, core)
        k4 = (self.dt/12)*self.dcdt(c+k3, z, T, core)
        c = c + (k1/6) + (k2/3) + (k3/3) + (k4/6)
            
        # Compute selection
        q = np.maximum(0, 1-(self.cmin/np.maximum(self.cmin, 2*c)))
        
        # RK4 terms (dz/dt)
        k1 = (self.dt/12)*self.dzdt(c, z, T, q)
        k2 = (self.dt/12)*self.dzdt(c, z+0.5*k1, T, q)
        k3 = (self.dt/12)*self.dzdt(c, z+0.5*k2, T, q)
        k4 = (self.dt/12)*self.dzdt(c, z+k3, T, q)
        z = z + (k1/6) + (k2/3) + (k3/3) + (k4/6)    
        
        return c, z
                    
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
            param_list = ['r0', 'm0', 'w', 'f', 'V', 'cmin']
            for param in param_list:
                self.output[param] = xr.DataArray(data=getattr(self, param),
                                                  dims=['site'], coords={'site': (['site'], _site)})
        
        if bc:
            bc_list = ['T', 'I']
            for _bc in bc_list:
                if getattr(self, _bc).ndim == 1:
                    if output_dt is None:
                        bc_subset = getattr(self, _bc)[_t_idx]
                    else:
                        bc_subset = np.concatenate([np.nan*np.ones((1,), dtype=np.float32), getattr(self, _bc)[_t_idx]])
                    self.output[_bc] = xr.DataArray(data=bc_subset,
                                                    dims=['time'], coords={'time': (['time'], _time, {'units': 'years'})})
                else:
                    if output_dt is None:
                        bc_subset = getattr(self, _bc)[:, _t_idx]
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
        if output_dt:
            _c_buffer = 0.*_c  # Compute moving averages for output
            _z_buffer = 0.*_c  # Compute moving averages for output
        
        with tqdm(total=self.j/12, unit=' years') as progress:
            for it in range(self.j):
                
                # Get current month
                month = it%12
                
                # Get boundary conditions
                _T = self.T[it] if self.T.ndim == 1 else self.T[:, it]
                _I = self.I[it] if self.I.ndim == 1 else self.I[:, it]
                
                # Evaluate whether this is a spawning month
                spawning = month in self.spawning_months
                
                # Iterate
                _c, _z = self.forward(c=_c, z=_z, T=_T, I=_I, 
                                      zc_offset=self.zc_offset, 
                                      dt=self.dt/12, spawning=spawning) # Converting dt to years
                
                # Error checking
                _c = np.minimum(_c, 1.)
                _c = np.maximum(_c, 0.)
                
                # Save to output
                if output_dt:
                    _c_buffer += _c/output_dt
                    _z_buffer += _z/output_dt
                    
                    if it in _t_idx:
                        self.output.c[:, _write_line] = _c_buffer
                        self.output.z[:, _write_line] = _z_buffer
                        _write_line += 1         
                        _c_buffer = 0.*_c
                        _z_buffer = 0.*_z
                else:
                    if it in _t_idx:
                        self.output.c[:, _write_line] = _c
                        self.output.z[:, _write_line] = _z
                        _write_line += 1   
                                        
                # Update progress bar
                if month == 11:
                    progress.update(1)