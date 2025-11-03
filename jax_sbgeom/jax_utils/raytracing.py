import jax 
import jax.numpy as jnp
from typing import Type
from dataclasses import dataclass


def _create_morton_codes(positions : jnp.ndarray, connectivity : jnp.ndarray) -> jnp.ndarray:
    '''
    Create morton codes for the triangles defined by the connectivity on the positions.

    See https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

    Parameters:
    ------------
    '''    
    N_BIT = 10
    clip_max = jnp.astype(2 ** N_BIT - 1, jnp.uint32)
    value_scale = 2.0**N_BIT
    
    centroids = jnp.mean(positions[connectivity], axis=1)  # (M, 3)

    r_min     = jnp.min(centroids, axis=0)
    r_max     = jnp.max(centroids, axis=0)  

    safe_divisor = jnp.where(r_max - r_min == 0, 1.0, r_max - r_min)

    normalized_centroids = (centroids - r_min) / safe_divisor    

    def expand_bits(v):
        # Directly from https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
        v = (v * 0x00010001) & 0xFF0000FF
        v = (v * 0x00000101) & 0x0F00F00F
        v = (v * 0x00000011) & 0xC30C30C3
        v = (v * 0x00000005) & 0x49249249
        return v
    
    def morton_3d(norm_centroids):
        int_coords = jnp.clip(jnp.floor(norm_centroids * value_scale).astype(jnp.uint32), 0, clip_max)
        return expand_bits(int_coords[:, 0]) * 4 + expand_bits(int_coords[:, 1]) * 2 + expand_bits(int_coords[:, 2])
    
    morton_codes  = morton_3d(normalized_centroids)
    return morton_codes

def _clz_float(x, dtype_float : Type = jnp.float64):
    # converts to float and back; see line "GPU Implementation" in https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
    significant_bit = jnp.floor(jnp.log2(x.astype(dtype_float)))
    return jnp.where(x ==0, 32, 31 - significant_bit.astype(jnp.int32))

def _common_prefix_length(a : jnp.ndarray, b : jnp.ndarray) -> jnp.ndarray:
    return _clz_float(jnp.bitwise_xor(a,b))

@jax.jit
def _delta_ij(sorted_morton_codes : jnp.ndarray, i : int, j : int):
    i_safe = jnp.clip(i, 0, sorted_morton_codes.shape[0]-1)
    j_safe = jnp.clip(j, 0, sorted_morton_codes.shape[0]-1)

    safe_mask = jnp.logical_and(i >= 0, i < sorted_morton_codes.shape[0])
    safe_mask = jnp.logical_and(safe_mask, j >= 0)
    safe_mask = jnp.logical_and(safe_mask, j < sorted_morton_codes.shape[0])

    # if the morton codes are the same, we should fallback to the _common_prefix_length of the indices themselves
    # this should not however be the case if i_safe or j_safe are out of bounds: then always -1
    equal_mask = jnp.equal(sorted_morton_codes[i_safe], sorted_morton_codes[j_safe])
    
    return jnp.where(safe_mask,                 
                        jnp.where(equal_mask, 
                                  _common_prefix_length(i_safe.astype(sorted_morton_codes.dtype), j_safe.astype(sorted_morton_codes.dtype)),
                                  _common_prefix_length(sorted_morton_codes[i_safe], sorted_morton_codes[j_safe])
                                 )
                     , -1
                     )




# morton_codes_manual = jnp.array([
#     0x00000001,
#     0x00000010,
#     0x00000100,
#     0x00000101,
#     0x00010011,
#     0x00011000,
#     0x00011001,
#     0x00011110])

@jax.jit
def _create_parallel_binary_radix_tree(morton_codes):
    morton_order = jnp.argsort(morton_codes)
    sorted_morton_codes = morton_codes[morton_order]

    N = morton_codes.shape[0]    
    # 

    internal_nodes = N - 1
    internal_idx   = jnp.arange(0, internal_nodes, dtype = morton_order.dtype)


    max_doublings = jnp.ceil(jnp.log2(N)).astype(jnp.int32) + 2
    l0            = 1


    def internal_node_function(idx):

        
        d = jnp.sign(_delta_ij(sorted_morton_codes, idx + 1, idx) - _delta_ij(sorted_morton_codes, idx, idx - 1))        
        delta_min = _delta_ij(sorted_morton_codes, idx, idx - d)

        # Binary exponential
        def lmax_condition(state):
            l, step = state
            return jnp.logical_and(_delta_ij(sorted_morton_codes, idx, idx + d * l) > delta_min, step < max_doublings)
        def lmax_body(state):
            l, step = state            
            return l * 2, step + 1
        
        l_final, _ = jax.lax.while_loop(lmax_condition, lmax_body, (l0, 0))
        
        
        def binary_search_condition(state):
            t, _ = state
            return t > 0
        
        def binary_search_body(state):
            t, l = state
            l_carry = jax.lax.cond(_delta_ij(sorted_morton_codes, idx, idx + d * (l+t))> delta_min, lambda _ : l + t, lambda _ : l, operand=None)
            t_carry = t // 2
            return t_carry, l_carry
        
        t_final, l_final = jax.lax.while_loop(binary_search_condition, binary_search_body, (l_final //2 , 0))
        
        j_idx = idx + d * l_final

        # Split position

        delta_node = _delta_ij(sorted_morton_codes, idx, j_idx)
        
        s = jnp.astype(0, j_idx.dtype)

        def split_condition(state):
            t, _ = state
            # t is a float here because otherwise you cannot 
            return t >= 0.5 # this continues including the t=1 case (i.e. if it is 1.2, it should continue another iteration because that iteration will be 2 and we need the 1 as well.)
        
        def split_body(state):
            t, s = state 

            t_int = jnp.ceil(t).astype(j_idx.dtype)

            s_carry = jax.lax.cond(_delta_ij(sorted_morton_codes, idx, idx + d * (s + t_int)) > delta_node, lambda _ : s + t_int, lambda _ : s, operand=None)

            t_carry = t  / 2
            return t_carry, s_carry
        
        t_final_split, s_final = jax.lax.while_loop(split_condition, split_body, (l_final /2 , s))        
        gamma                  = idx + s_final * d + jnp.minimum(d, 0 )        
        left_idx               = jax.lax.cond(jnp.minimum(idx, j_idx) == gamma, lambda _ : gamma, lambda _ : gamma + N, operand=None)
        right_idx              = jax.lax.cond(jnp.maximum(idx, j_idx) == gamma + 1, lambda _ : gamma + 1, lambda _ : gamma + 1 + N, operand=None)
        return left_idx, right_idx    
    
    left_idx, right_idx = jax.vmap(internal_node_function)(internal_idx)
        
    return left_idx, right_idx

@jax.jit
def _create_aabb(primitive_coordinates : jnp.ndarray):
    """Create axis-aligned bounding box for a given primitive.

    Parameters
    ----------
    primitive:  jnp.ndarray of shape (N, M, 3)
        N coordinates of a primitive with shape M

    Returns
    -------
    jnp.ndarray of shape (N, 2, 3)
      representing min and max corners of the AABB.
    """
    min_corner = jnp.min(primitive_coordinates, axis=-2)
    max_corner = jnp.max(primitive_coordinates, axis=-2)
    return jnp.stack([min_corner, max_corner], axis = 1)

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BVH:
    left_idx  : jnp.ndarray 
    right_idx : jnp.ndarray
    aabb      : jnp.ndarray


@jax.jit
def build_lbvh(positions, connectivity):
    morton_codes = _create_morton_codes(positions, connectivity)

    N = morton_codes.shape[0]

    lbvh           = _create_parallel_binary_radix_tree(morton_codes)
    
    
    
    aabb_leaves    = _create_aabb(positions[connectivity])

    def condition(state):
        
        return jnp.logical_not(jnp.all(state[0]))

    def compute_aabb_compound(aabb, left_i, right_i):    
        return jnp.stack([
            jnp.minimum(aabb[left_i,0], aabb[right_i,0]),
            jnp.maximum(aabb[left_i,1], aabb[right_i,1])
        ])
    
    def try_compute(state, left_i, right_i):
        available, aabb              = state
        left_processed_right_process = jnp.logical_and(available[left_i], available[right_i])
        return jax.lax.cond(
            left_processed_right_process,
            lambda _ : (True, compute_aabb_compound(aabb, left_i, right_i)),
            lambda _ : (False, jnp.zeros((2,3))),
            operand = None
        )

    def single_iteration_i(state, left_i, right_i, i):
        available, aabb = state
        return jax.lax.cond(
            available[i],
            lambda _ : (True,   aabb[i]),
            lambda _ : try_compute(state, left_i, right_i),
            operand = None
        )
        
    def single_iteration(state):
        
        avaiable_internal, aabb_internal =  jax.vmap(single_iteration_i, in_axes=(None, 0, 0, 0))(
            state,
            lbvh[0],
            lbvh[1],
            N + jnp.arange(N - 1)
        )
        return jnp.concatenate([
            jnp.ones(N, dtype = jnp.bool), avaiable_internal, 
        ]), jnp.concatenate([aabb_leaves, aabb_internal], axis = 0)
    
    initial_aabb   = jnp.zeros((2 * N - 1, 2, 3))
    initial_aabb   = initial_aabb.at[:N].set(aabb_leaves)

    initial_available = jnp.zeros((2 *  N - 1), dtype = jnp.bool) 
    initial_available = initial_available.at[:N].set(True)


    _, total_aabb_array = jax.lax.while_loop(condition, single_iteration, (initial_available, initial_aabb))

    N_leafs = jnp.arange(0,N)
    return BVH(
        left_idx  = jnp.concatenate([N_leafs, lbvh[0]]),
        right_idx = jnp.concatenate([N_leafs, lbvh[1]]),
        aabb      = total_aabb_array)
    
    