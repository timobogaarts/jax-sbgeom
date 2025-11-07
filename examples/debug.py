import jax_sbgeom as jsb
import pytest
import jax
#jax.config.update("jax_enable_x64", True)
def _get_files():
    vmec_files = ["/home/tbogaarts/stellarator_paper/base_data/vmecs/helias3_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/helias5_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/squid_vmec.nc4"]
    return vmec_files



def test_bvh_probing(_get_flux_surface_jax):
    fs_jax = jsb.flux_surfaces.FluxSurface.from_hdf5(_get_files()[0])
    
    surface_mesh = jsb.flux_surfaces.flux_surface_meshing.mesh_surface(fs_jax, 1.0, jsb.flux_surfaces.ToroidalExtent.full(), 65,62)

    lbvh         = jsb.jax_utils.raytracing.build_lbvh(surface_mesh[0], surface_mesh[1])

    aabb_total = jsb.jax_utils.raytracing._create_aabb(surface_mesh[0][surface_mesh[1]])

    hits_bvh_probing = jsb.jax_utils.raytracing.probe_bvh(lbvh, surface_mesh[0][0:1000])
    print(aabb_total)
    @jax.jit
    def bulk_all(aabb, points):
        final_hits = jax.vmap(jax.vmap(jsb.jax_utils.raytracing._point_in_aabb, in_axes=(None, 0)), in_axes=(0, None))(points, aabb)  
        return final_hits
    hits_bulk = bulk_all(aabb_total, surface_mesh[0][0:1000])

    print(hits_bvh_probing, hits_bulk)

test_bvh_probing(0)
