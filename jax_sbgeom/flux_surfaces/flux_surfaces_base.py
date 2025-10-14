import jax.numpy as jnp
import h5py 
import jax
from dataclasses import dataclass

def _create_mpol_vector(ntor : int, mpol : int):    
    return jnp.array([0 for i in range(ntor + 1)] + sum([[i for j in range(2 * ntor + 1)]for i in range(1, mpol )], []), dtype=int)

def _create_ntor_vector(ntor : int, mpol : int, symm : int):
    return jnp.array(list(range(0, (ntor + 1) * symm , symm)) + sum([list(range(-ntor * symm, (ntor + 1) * symm, symm)) for i in range(mpol - 1)], []), dtype=int)


def _cylindrical_to_cartesian(RZphi : jnp.ndarray):
    R = RZphi[..., 0]
    Z = RZphi[..., 1]
    phi = RZphi[..., 2]
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    return jnp.stack([x, y, Z], axis=-1)

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FluxSurfaceSettings:
    mpol : int
    ntor : int
    nfp  : int
    nsurf : int    

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FluxSurfaceData:
    Rmnc : jnp.ndarray
    Zmns : jnp.ndarray
    mpol_vector : jnp.ndarray
    ntor_vector : jnp.ndarray

    @classmethod
    def from_rmnc_zmns_settings(cls, Rmnc : jnp.ndarray, Zmns : jnp.ndarray, settings : FluxSurfaceSettings):
        mpol_vector = _create_mpol_vector(settings.ntor, settings.mpol)
        ntor_vector = _create_ntor_vector(settings.ntor, settings.mpol, settings.nfp)
        assert(Rmnc.shape == Zmns.shape)
        assert(Rmnc.shape[0] == settings.nsurf)
        assert(Rmnc.shape[1] == len(mpol_vector))
        return cls(Rmnc=Rmnc, Zmns=Zmns, mpol_vector=mpol_vector, ntor_vector=ntor_vector)
    


@dataclass(frozen=True)
class FluxSurface:
    ''' 
    Class representing a set of flux surfaces using a VMEC-like representation.


    Attributes:
    -----------
    Rmnc : (nsurf, nmodes) jnp.ndarray
        Radial Fourier coefficients for the R coordinate.
    Zmns : (nsurf, nmodes) jnp.ndarray
        Vertical Fourier coefficients for the Z coordinate.
    settings : FluxSurfaceSettings
        Settings object containing parameters like mpol, ntor, nfp, and the mode vectors
    '''
    data            : FluxSurfaceData = None
    settings        : FluxSurfaceSettings = None
    

    @classmethod
    def from_hdf5(cls, filename : str):
        """Load a FluxSurface from an VMEC-type HDF5 file.

        Parameters:
        ----------
        filename : str
            Path to the HDF5 file.
        Returns:
        -------
        FluxSurface
            The loaded FluxSurface object.

        """

        with h5py.File(filename) as f:
            Rmnc = jnp.array(f['rmnc'])            
            Zmns = jnp.array(f['zmns'])            

            mpol = int(f['mpol'][()])
            ntor = int(f['ntor'][()])                        
            nfp  = int(f['nfp'][()])

            assert( jnp.all( _create_mpol_vector(ntor, mpol) == jnp.array(f['xm'])))      # sanity check
            assert( jnp.all( _create_ntor_vector(ntor, mpol, nfp) == jnp.array(f['xn']))) # sanity check
            
            nsurf = Zmns.shape[0]
            settings = FluxSurfaceSettings(
                mpol=mpol,
                ntor=ntor,
                nfp=nfp,                
                nsurf=nsurf
            )

            data = FluxSurfaceData.from_rmnc_zmns_settings(Rmnc, Zmns, settings)
        return cls(data=data, settings=settings)
    

    def cylindrical_position(self, s, theta, phi):
        return _cylindrical_position_interpolated_jit(self.data, self.settings, s, theta, phi)
    
    def cartesian_position(self, s, theta, phi):
        RZphi = self.cylindrical_position(s, theta, phi)
        return _cylindrical_to_cartesian(RZphi)
        

def _cylindrical_position_single(Rmnc : jnp.ndarray, Zmns : jnp.ndarray, mpol_vector : jnp.ndarray, ntor_vector  : jnp.ndarray,  settings  : FluxSurfaceSettings,  theta  : jnp.ndarray, phi : jnp.ndarray):
    ''' 
    Internal 

    Function to compute cylindrical position from Fourier coefficients.

    Rmnc and Zmns can be 1D arrays of Fourier coefficients, or they can be broadcasted arrays
    with shape (..., nmodes), where ... represents the shape of theta and phi.
    This ensures that both single-surfaces (where Rmnc and Zmns are 1D) and multiple surfaces
    (where Rmnc and Zmns have shape (nsurf, nmodes)) can be handled within the same function.
    

    Parameters:
    -----------
    Rmnc : (..., nmodes) jnp.ndarray
        Radial Fourier coefficients for the R coordinate.
    Zmns : (... , nmodes) jnp.ndarray
        Vertical Fourier coefficients for the Z coordinate.
    settings : FluxSurfaceSettings
        Settings object containing parameters like mpol, ntor, nfp, and the mode vectors [static]
        Currently unused.

    theta : ( *phi.shape ) jnp.ndarray
        Poloidal angles [radians].
    phi : ( *phi.shape ) jnp.ndarray
        Toroidal angles [radians].

    Returns:
    --------
    (3, *phi.shape) jnp.ndarray
        Cylindrical coordinates (R, Z, phi).
    '''

    def r_sum(i, val):        
        return val + Rmnc[..., i] * jnp.cos(mpol_vector[i] * theta - ntor_vector[i] * phi)
    
    def z_sum(i, val):
        return val + Zmns[..., i] * jnp.sin(mpol_vector[i] * theta - ntor_vector[i] * phi)
    
    R_init = jnp.zeros(theta.shape)
    Z_init = jnp.zeros(theta.shape)
    
    R = jax.lax.fori_loop(0, Rmnc.shape[-1], r_sum, R_init)
    Z = jax.lax.fori_loop(0, Zmns.shape[-1], z_sum, Z_init)
    return R, Z

def _interpolate_fractions(s, nsurf):
    
    s_start =  s * (nsurf-1)
    i0      = jnp.floor(s_start).astype(int)
    i1      = jnp.minimum(i0 + 1, nsurf - 1)    
    ds      = s_start - i0    
    return i0, i1, ds 



def _cylindrical_position_interpolated(data : FluxSurfaceData, settings  : FluxSurfaceSettings, s : jnp.ndarray, theta  : jnp.ndarray, phi : jnp.ndarray):
    s_1d     = jnp.atleast_1d(s) 
    theta_1d = jnp.atleast_1d(theta) 
    phi_1d   = jnp.atleast_1d(phi)

    # Since it is possible that s is a scalar, we first interpolate just s
    # Then, we index the Rmnc and Zmns arrays. So if s is a small array, 
    # this saves significant memory compared to first broadcasting then indexing.
    
    i0, i1, ds   = _interpolate_fractions(s_1d, settings.nsurf)
    
    # shape is now (s_1d.shape, n_modes)
    Rmnc_i0 = data.Rmnc[i0]
    Zmns_i0 = data.Zmns[i0]

    Rmnc_i1 = data.Rmnc[i1]
    Zmns_i1 = data.Zmns[i1]

    # Now we broadcast everything to the final shape
    s_bc, theta_bc, phi_bc = jnp.broadcast_arrays(s_1d, theta_1d, phi_1d)
    Rmnc_i0_bc = jnp.broadcast_to(Rmnc_i0, s_bc.shape + (Rmnc_i0.shape[-1],))
    Zmns_i0_bc = jnp.broadcast_to(Zmns_i0, s_bc.shape + (Zmns_i0.shape[-1],))

    Rmnc_i1_bc = jnp.broadcast_to(Rmnc_i1, s_bc.shape + (Rmnc_i1.shape[-1],))
    Zmns_i1_bc = jnp.broadcast_to(Zmns_i1, s_bc.shape + (Zmns_i1.shape[-1],))
    
    R_0, Z_0   = _cylindrical_position_single(Rmnc_i0_bc, Zmns_i0_bc, data.mpol_vector, data.ntor_vector, settings, theta_bc, phi_bc)
    R_1, Z_1   = _cylindrical_position_single(Rmnc_i1_bc, Zmns_i1_bc, data.mpol_vector, data.ntor_vector, settings, theta_bc, phi_bc)   
    R_final = (1 - ds) * R_0 + ds * R_1
    Z_final = (1 - ds) * Z_0 + ds * Z_1
    return jnp.stack([R_final, Z_final , phi_bc], axis=-1)
    
_cylindrical_position_interpolated_jit = jax.jit(_cylindrical_position_interpolated)

