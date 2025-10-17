import jax_sbgeom as jsb
import os

# or locally with

import h5py

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
#jax.config.update("jax_logging_level", "DEBUG")

import numpy as onp
import sys 
import os
import os


import jax.numpy as jnp

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)
vmec_files = ["/home/tbogaarts/stellarator_paper/base_data/vmecs/helias3_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/helias5_vmec.nc4", "/home/tbogaarts/stellarator_paper/base_data/vmecs/squid_vmec.nc4"]

from jax_sbgeom.flux_surfaces.flux_surfaces_base import _cartesian_position_interpolated_jit, _cylindrical_position_interpolated, _cartesian_position_interpolated_grad, ToroidalExtent
from tests.flux_surfaces.flux_surface_base import test_position, _get_flux_surfaces, _sampling_grid, _1d_sampling_grid, test_normals, test_meshing_surface, test_principal_curvatures, test_all_closed_surfaces

test_pos             = True
test_norm            = True
test_meshing_surf    = True
test_principal_curv  = True
test_closed_surf     = True
for vmec_file in vmec_files:
    print(f"\n--- Testing VMEC file: {vmec_file} ---")
    
    
    if test_pos:
        try:
            test_position(vmec_file, n_repetitions=1)
        except Exception as e:
            print(f"test_position failed for {vmec_file} with error: {e}")

    if test_norm:
        try:
            test_normals(vmec_file, n_repetitions=1)
        except Exception as e:
            print(f"test_normals failed for {vmec_file} with error: {e}")

    if test_meshing_surf:
        try:
            test_meshing_surface(vmec_file, n_repetitions=1)
            test_meshing_surface(vmec_file, tor_extent='full', n_repetitions=1)
        except Exception as e:
            print(f"test_meshing_surface failed for {vmec_file} with error: {e}")
    if test_principal_curv:
        try:
            test_principal_curvatures(vmec_file, n_repetitions=1)
        except Exception as e:
            print(f"test_principal_curvatures failed for {vmec_file} with error: {e}")
    if test_closed_surf:
        try:
            test_all_closed_surfaces(vmec_file, n_repetitions=1)
        except Exception as e:
            print(f"test_all_closed_surfaces failed for {vmec_file} with error: {e}")