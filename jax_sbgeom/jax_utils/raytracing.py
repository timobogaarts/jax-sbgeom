import jax 
import jax.numpy as jnp
from typing import Type
from dataclasses import dataclass

def _get_norm_centroids(positions, connectivity):
    '''
    Get normalized centroids for the triangles defined by the connectivity on the positions.

    Parameters:
    ------------
    positions : jnp.ndarray
        (N, 3) array of vertex positions
    connectivity : jnp.ndarray
        (M, 3) array of connectivity indices
    Returns:
    ------------
    normalized_centroids : jnp.ndarray
        (M, 3) array of normalized centroids
    '''
    centroids = jnp.mean(positions[connectivity], axis=1)  # (M, 3)

    r_min     = jnp.min(centroids, axis=0)
    r_max     = jnp.max(centroids, axis=0)  

    safe_divisor = jnp.where(r_max - r_min == 0, 1.0, r_max - r_min)

    normalized_centroids = (centroids - r_min) / safe_divisor    
    return normalized_centroids


def _create_morton_codes(normalized_positions : jnp.ndarray) -> jnp.ndarray:
    '''
    Create 32 bit morton codes for the triangles defined by the connectivity on the positions.

    See https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

    Parameters:
    ------------
    positions : jnp.ndarray
        (N, 3) array of vertex positions
    connectivity : jnp.ndarray
        (M, 3) array of connectivity indices
    Returns:
    ------------
    morton_codes : jnp.ndarray
        (M,) array of morton codes
    '''    
    N_BIT = 10
    clip_max = jnp.astype(2 ** N_BIT - 1, jnp.uint32)
    value_scale = 2.0**N_BIT

    def expand_bits(v):
        # ensure unsigned 32-bit
        v = jnp.asarray(v, dtype=jnp.uint32)

        # Use JAX integer constants (avoid raw Python ints in bitwise expressions)
        c1 = jnp.uint32(0x00010001)  # 65537
        m1 = jnp.uint32(0xFF0000FF)
        c2 = jnp.uint32(0x00000101)  # 257
        m2 = jnp.uint32(0x0F00F00F)
        c3 = jnp.uint32(0x00000011)  # 17
        m3 = jnp.uint32(0xC30C30C3)
        c4 = jnp.uint32(0x00000005)  # 5
        m4 = jnp.uint32(0x49249249)

        v = (v * c1) & m1
        v = (v * c2) & m2
        v = (v * c3) & m3
        v = (v * c4) & m4

        return v


    
    def morton_3d(norm_centroids):
        int_coords = jnp.clip(jnp.floor(norm_centroids * value_scale).astype(jnp.uint32), 0, clip_max)
        return expand_bits(int_coords[:, 0]) * 4 + expand_bits(int_coords[:, 1]) * 2 + expand_bits(int_coords[:, 2])
    
    morton_codes  = morton_3d(normalized_positions)
    return morton_codes

def _common_prefix_length(a : jnp.ndarray, b : jnp.ndarray) -> jnp.ndarray:
    return jax.lax.clz(jnp.bitwise_xor(a,b))


def _delta_ij(sorted_morton_codes : jnp.ndarray, i : int, j : int, int_dtype : jnp.dtype = jnp.int32):

    # To ensure no OOB access, first clip the indices
    i_safe = jnp.clip(i, 0, sorted_morton_codes.shape[0]-1)
    j_safe = jnp.clip(j, 0, sorted_morton_codes.shape[0]-1)

    # However, we need to have a mask to indicate whether the original indices were OOB. If so, the result should be -1
    safe_mask = jnp.logical_and( jnp.logical_and( i >= 0, i < sorted_morton_codes.shape[0]), 
                                 jnp.logical_and( j >= 0, j < sorted_morton_codes.shape[0]))
    
    # Furthermore, if the morton codes are the same, we should fallback to the _common_prefix_length + the base common prefix length of the indices themselves.
    # (jnp.where computes eagerly anyway, so need to compute all of them)
    equal_mask                   = jnp.equal(sorted_morton_codes[i_safe], sorted_morton_codes[j_safe])
    common_prefix_length_morton  = jnp.astype(_common_prefix_length(sorted_morton_codes[i_safe],              sorted_morton_codes[j_safe]), int_dtype)
    common_prefix_length_indices = jnp.astype(_common_prefix_length(i_safe.astype(sorted_morton_codes.dtype), j_safe.astype(sorted_morton_codes.dtype)), int_dtype)
    
    return jnp.where(safe_mask,                 
                        jnp.where(equal_mask, 
                                  common_prefix_length_morton + common_prefix_length_indices,
                                  common_prefix_length_morton
                                 )
                     , -1
                     )

@jax.jit
def _create_parallel_binary_radix_tree(morton_codes : jnp.ndarray):
    # See https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2012/11/karras2012hpg_paper.pdf

    N = morton_codes.shape[0]                      
    morton_order        = jnp.argsort(morton_codes) # int_dtype
    sorted_morton_codes = morton_codes[morton_order]# uint_dtype

    morton_order_type   = morton_order.dtype        # type(morton_order): used for indexing
    
    internal_nodes = N - 1 # size_type
    internal_idx   = jnp.arange(0, internal_nodes, dtype = morton_order_type) # size_type

    max_doublings = jnp.ceil(jnp.log2(N)).astype(morton_order_type) + 2
    l0            = jnp.astype(1, morton_order_type)

    def internal_node_function(idx):
        d = jnp.sign(_delta_ij(sorted_morton_codes, idx + 1, idx) - _delta_ij(sorted_morton_codes, idx, idx - 1))        
        delta_min = _delta_ij(sorted_morton_codes, idx, idx - d)
        
        # Exponential search to find upper bound on lmax
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
        
        t_final, l_final = jax.lax.while_loop(binary_search_condition, binary_search_body, (l_final //2 , jnp.astype(0, morton_order_type)))
        
        j_idx = idx + d * l_final

        # Split position

        delta_node = _delta_ij(sorted_morton_codes, idx, j_idx)
        
        s = jnp.astype(0, morton_order_type)

        def split_condition(state):
            _, _, t_one_done = state
            
            return ~t_one_done # this continues including the t=1 case just once
        
        def split_body(state):
            t, s, _    = state 
            
            s_carry = jax.lax.cond(_delta_ij(sorted_morton_codes, idx, idx + d * (s + t)) > delta_node, lambda _ : s + t, lambda _ : s, operand=None)
            t_carry = (t+1)  // 2
            return t_carry, s_carry, jnp.equal(t, 1)
        
        t_final_split, s_final, _ = jax.lax.while_loop(split_condition, split_body, ((l_final+1) //2 , s, False))        
        gamma                     = idx + s_final * d + jnp.minimum(d, 0 )        
        
        left_idx               = jax.lax.cond(jnp.minimum(idx, j_idx) == gamma, lambda _ : gamma, lambda _ : gamma + N, operand=None)
        right_idx              = jax.lax.cond(jnp.maximum(idx, j_idx) == gamma + 1, lambda _ : gamma + 1, lambda _ : gamma + 1 + N, operand=None)
        return left_idx, right_idx    
    
    left_idx, right_idx = jax.vmap(internal_node_function)(internal_idx)

    total_left_idx = jnp.concatenate([jnp.arange(N, dtype = morton_order_type), left_idx])
    total_right_idx = jnp.concatenate([jnp.arange(N, dtype = morton_order_type), right_idx])
    return total_left_idx, total_right_idx, morton_order

@jax.jit
def _check_binary_radix_tree(left_idx, right_idx):
    N_leaves = (left_idx.shape[0] +1) //2
    
    initial_available = jnp.zeros((2 *  N_leaves - 1), dtype = jnp.bool) 
    initial_available = initial_available.at[:N_leaves].set(True)

    def condition(state):
        total_computed, i_loop  = state
        return jnp.any(~total_computed[N_leaves:]) & (i_loop < 2 * N_leaves)
    
    def single_iteration(state):
        computed, i_loop = state
        left_available  = computed[left_idx ]
        right_available = computed[right_idx]
        computed       = (left_available & right_available)
        return computed, i_loop + 1
    
    state = (initial_available, 0)
    final_state = jax.lax.while_loop(condition, single_iteration, state)
    total_computed, _ = final_state
    return jnp.sum(total_computed) == total_computed.shape[0]

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
    left_idx      : jnp.ndarray # left child indices for each node
    right_idx     : jnp.ndarray # right child indices for each node
    aabb          : jnp.ndarray # AABB for each node
    leafs         : jnp.ndarray # bool array
    order         : jnp.ndarray # order from primitives -> BVH
    inverse_order : jnp.ndarray # order from BVH -> primitives


@jax.jit
def build_lbvh(positions, connectivity):
    '''
    Build a Linear Bounding Volume Hierarchy (LBVH) for a given set of 3D positions and connectivity.
    Parameters:
    -----------
        positions: jnp.ndarray
            An array of shape (N, 3) representing the 3D coordinates of points.
        connectivity jnp.ndarray:
            An array of shape (M, K) representing the connectivity of the points,
            where each row defines a primitive (e.g., triangle) using indices into the positions array.
    Returns:
    -----------
        BVH:
            Dataclass containing the LBVH structure with left and right child indices, AABBs, leaf indicators, and ordering information.
    '''
    normalized_centroids  = _get_norm_centroids(positions, connectivity)
    morton_codes          = _create_morton_codes(normalized_centroids)
    N                     = morton_codes.shape[0]
    

    left_idx, right_idx, morton_order  = _create_parallel_binary_radix_tree(morton_codes)        
    aabb_leaves                        = _create_aabb(positions[connectivity])
    aabb_leaves_sorted                 = aabb_leaves[morton_order] 

    left_internal  = left_idx [N:]
    right_internal = right_idx[N:]

    def condition(state):
        return ~jnp.all(state[1])
    
    def compute_aabb_compound(aabb, left_i, right_i):
        return jnp.stack([
            jnp.minimum(aabb[left_i,0], aabb[right_i,0]),
            jnp.maximum(aabb[left_i,1], aabb[right_i,1])
        ], axis = 1)
    
    def compute(state):
        aabb, computed = state
        new_aabb       = compute_aabb_compound(aabb, left_internal, right_internal)
        new_computed   = computed[left_internal] & computed[right_internal]
        return jnp.concatenate([aabb_leaves_sorted, new_aabb], axis=0), jnp.concatenate([jnp.ones((N,), dtype=bool), new_computed], axis=0)
    
    initial_aabb        = jnp.concatenate([aabb_leaves_sorted, jnp.zeros((N-1,2,3))], axis=0)
    initial_available   = jnp.concatenate([jnp.ones((N,), dtype=bool), jnp.zeros((N-1,), dtype=bool)], axis=0)
    
    aabb_total, _       = jax.lax.while_loop(condition, compute, (initial_aabb, initial_available))
    
    return BVH(left_idx, right_idx, 
               aabb_total, initial_available, 
               morton_order, jnp.argsort(morton_order))

def _point_in_aabb(point : jnp.ndarray, aabb : jnp.ndarray) -> jnp.ndarray:
    """
    Check if a point is inside an axis-aligned bounding box (AABB).

    Parameters
    ----------
    point: (3,) jnp.ndarray
      point coordinates
    aabb:  (2, 3) jnp.ndarray
      AABB defined by min and max corners

    Returns
    -------
    inside: bool - True if point is inside the AABB, False otherwise
    """
    min_corner = aabb[0]
    max_corner = aabb[1]
    inside = jnp.all((point >= min_corner) & (point <= max_corner))
    return inside


points_in_aabbs   = jax.vmap(jax.vmap(_point_in_aabb, in_axes=(0, None)), in_axes = (None, 0))

points_in_aabb    = jax.vmap(_point_in_aabb, in_axes=(0, None))

points_in_aabbvec = jax.vmap(_point_in_aabb, in_axes=(0, 0))

@jax.jit
def probe_bvh(bvh : BVH, points : jnp.ndarray, stack_size : int = 64, max_hit_size : int = 64):

    N_leafs = (bvh.leafs.shape[0] + 1) // 2


    N_points = points.shape[0]
    n_points_arange = jnp.arange(N_points)

    @jax.jit
    def condition(state):
        stack_idx, _, _, _, loop_idx  = state
        return  jnp.any(stack_idx >= 0 ) & (loop_idx < N_leafs)

    # Leaf 

    def loop(state):
        stack_idx, stack, hits, no_hits, loop_idx  = state
        
        current_idx  = stack[n_points_arange, stack_idx]                
        
        
        left_idx     = bvh.left_idx[ current_idx]
        right_idx    = bvh.right_idx[current_idx]
    
        left_contains  = points_in_aabbvec(points, bvh.aabb[left_idx])    # shape (N_points, )
        right_contains = points_in_aabbvec(points, bvh.aabb[right_idx])  # shape (N_points, )
        
        hits_with_left = jnp.where(jnp.logical_and(left_contains, bvh.leafs[left_idx])[:,None], 
             hits.at[n_points_arange, no_hits].set(left_idx), hits)
        
        no_hits_with_left = jnp.where(jnp.logical_and(left_contains, bvh.leafs[left_idx]),
            no_hits + 1, no_hits)
        
        hits_with_both    = jnp.where(jnp.logical_and(right_contains, bvh.leafs[right_idx])[:,None], 
             hits_with_left.at[n_points_arange, no_hits_with_left].set(right_idx), hits_with_left)
        
        no_hits_with_both = jnp.where(jnp.logical_and(right_contains, bvh.leafs[right_idx]),
            no_hits_with_left + 1, no_hits_with_left)
        
        
        traverse_left    = jnp.logical_and(left_contains, jnp.logical_not(bvh.leafs[left_idx]))
        traverse_right   = jnp.logical_and(right_contains, jnp.logical_not(bvh.leafs[right_idx]))
        
        transverse_any = jnp.logical_or(traverse_left, traverse_right)
        transverse_both = jnp.logical_and(traverse_left, traverse_right)

        new_stack_idx = stack_idx + transverse_both.astype(jnp.int32) - (~transverse_any).astype(jnp.int32)
    
        new_stack = jnp.where(
            transverse_any[:, None],
            jnp.where(
                transverse_both[:, None],
                stack.at[n_points_arange, stack_idx + 1].set(right_idx).at[n_points_arange, stack_idx].set(left_idx),
                jnp.where(
                    traverse_left[:, None],
                    stack.at[n_points_arange, stack_idx].set(left_idx),
                    stack.at[n_points_arange, stack_idx].set(right_idx)
                )
            ),
            stack.at[n_points_arange, stack_idx].set(-1)
        )

        return new_stack_idx, new_stack, hits_with_both, no_hits_with_both, loop_idx + 1
        
    
    N_points = points.shape[0]
    stack = jnp.full((N_points, stack_size,  ), -1) 
    hits  = jnp.full((N_points, max_hit_size,), -1)    
    
    initial_stack = stack.at[..., 0].set(N_leafs)  # start with root node
    initial_state = (jnp.zeros(N_points, dtype=int), initial_stack, hits,jnp.zeros(N_points, dtype= int),0)    
    
    final_stack_idx, final_stack, final_hits, final_no_hits, n_loops = jax.lax.while_loop(
        condition,
        loop,
        initial_state)

    return final_hits, n_loops

def ray_intersects_aabb(origin, direction, aabb):
    '''
    Vectorized 

    '''
    inv_dir = jnp.where(direction == 0.0, jnp.inf, 1.0 / direction)  # precompute to avoid divides
    tmin = (aabb[..., 0, :] - origin) * inv_dir
    tmax = (aabb[..., 1, :] - origin) * inv_dir

    # if direction is negative, swap
    t1 = jnp.minimum(tmin, tmax)
    t2 = jnp.maximum(tmin, tmax)

    t_enter = jnp.max(t1, axis=-1)
    t_exit  = jnp.min(t2, axis=-1)
    hit = (t_exit >= jnp.maximum(t_enter, 0.0))
    return hit#, t_enter, t_exit



@jax.jit
def ray_traversal_bvh(bvh : BVH, points : jnp.ndarray, directions : jnp.ndarray, stack_size : int = 64, max_hit_size : int = 64):

    N_leafs = (bvh.leafs.shape[0] + 1) // 2

    N_points = points.shape[0]
    n_points_arange = jnp.arange(N_points)

    @jax.jit
    def condition(state):
        stack_idx, _, _, _, loop_idx  = state
        return  jnp.any(stack_idx >= 0 ) & (loop_idx < N_leafs)

    def loop(state):
        stack_idx, stack, hits, no_hits, loop_idx  = state
        
        current_idx  = stack[n_points_arange, stack_idx]                
        
        left_idx     = bvh.left_idx[ current_idx]
        right_idx    = bvh.right_idx[current_idx]
    
        left_contains  = ray_intersects_aabb(points, directions, bvh.aabb[left_idx])    # shape (N_points, )
        right_contains = ray_intersects_aabb(points, directions, bvh.aabb[right_idx])   # shape (N_points, )
        
        hits_with_left = jnp.where(jnp.logical_and(left_contains, bvh.leafs[left_idx])[:,None], 
             hits.at[n_points_arange, no_hits].set(left_idx), hits)
        
        no_hits_with_left = jnp.where(jnp.logical_and(left_contains, bvh.leafs[left_idx]),
            no_hits + 1, no_hits)
        
        hits_with_both    = jnp.where(jnp.logical_and(right_contains, bvh.leafs[right_idx])[:,None], 
             hits_with_left.at[n_points_arange, no_hits_with_left].set(right_idx), hits_with_left)
        
        no_hits_with_both = jnp.where(jnp.logical_and(right_contains, bvh.leafs[right_idx]),
            no_hits_with_left + 1, no_hits_with_left)
        
        
        traverse_left    = jnp.logical_and(left_contains, jnp.logical_not(bvh.leafs[left_idx]))
        traverse_right   = jnp.logical_and(right_contains, jnp.logical_not(bvh.leafs[right_idx]))
        
        transverse_any = jnp.logical_or(traverse_left, traverse_right)
        transverse_both = jnp.logical_and(traverse_left, traverse_right)

        new_stack_idx = stack_idx + transverse_both.astype(jnp.int32) - (~transverse_any).astype(jnp.int32)
    
        new_stack = jnp.where(
            transverse_any[:, None],
            jnp.where(
                transverse_both[:, None],
                stack.at[n_points_arange, stack_idx + 1].set(right_idx).at[n_points_arange, stack_idx].set(left_idx),
                jnp.where(
                    traverse_left[:, None],
                    stack.at[n_points_arange, stack_idx].set(left_idx),
                    stack.at[n_points_arange, stack_idx].set(right_idx)
                )
            ),
            stack.at[n_points_arange, stack_idx].set(-1)
        )

        return new_stack_idx, new_stack, hits_with_both, no_hits_with_both, loop_idx + 1
        
    
    N_points = points.shape[0]
    stack = jnp.full((N_points, stack_size,  ), -1) 
    hits  = jnp.full((N_points, max_hit_size,), -1)    
    
    initial_stack = stack.at[..., 0].set(N_leafs)  # start with root node
    initial_state = (jnp.zeros(N_points, dtype=int), initial_stack, hits,jnp.zeros(N_points, dtype= int),0)    
    
    final_stack_idx, final_stack, final_hits, final_no_hits, n_loops = jax.lax.while_loop(
        condition,
        loop,
        initial_state)
    return final_hits


@jax.jit
def ray_traversal_bvh_single(bvh : BVH, point : jnp.ndarray, direction : jnp.ndarray, stack_size : int = 128, max_hit_size : int = 128):

    N_leafs = (bvh.leafs.shape[0] + 1) // 2

    @jax.jit
    def condition(state):
        stack_idx, _, _, _, loop_idx  = state
        return  jnp.any(stack_idx >= 0 ) & (loop_idx < N_leafs)

    def loop(state):
        stack_idx, stack, hits, no_hits, loop_idx  = state
        
        current_idx  = stack[stack_idx]                
        
        left_idx     = bvh.left_idx[ current_idx]
        right_idx    = bvh.right_idx[current_idx]
    
        left_contains  = ray_intersects_aabb(point, direction, bvh.aabb[left_idx])    # shape (N_points, )
        right_contains = ray_intersects_aabb(point, direction, bvh.aabb[right_idx])   # shape (N_points, )
        
        hits_with_left = jnp.where(jnp.logical_and(left_contains, bvh.leafs[left_idx]), 
             hits.at[no_hits].set(left_idx), hits)
        
        no_hits_with_left = jnp.where(jnp.logical_and(left_contains, bvh.leafs[left_idx]),
            no_hits + 1, no_hits)
        
        hits_with_both    = jnp.where(jnp.logical_and(right_contains, bvh.leafs[right_idx]), 
             hits_with_left.at[no_hits_with_left].set(right_idx), hits_with_left)
        
        no_hits_with_both = jnp.where(jnp.logical_and(right_contains, bvh.leafs[right_idx]),
            no_hits_with_left + 1, no_hits_with_left)
        
        
        traverse_left    = jnp.logical_and(left_contains, jnp.logical_not(bvh.leafs[left_idx]))
        traverse_right   = jnp.logical_and(right_contains, jnp.logical_not(bvh.leafs[right_idx]))
        
        transverse_any = jnp.logical_or(traverse_left, traverse_right)
        transverse_both = jnp.logical_and(traverse_left, traverse_right)

        new_stack_idx = stack_idx + transverse_both.astype(jnp.int32) - (~transverse_any).astype(jnp.int32)
    
        new_stack = jnp.where(
            transverse_any,
            jnp.where(
                transverse_both,
                stack.at[stack_idx + 1].set(right_idx).at[stack_idx].set(left_idx),
                jnp.where(
                    traverse_left,
                    stack.at[ stack_idx].set(left_idx),
                    stack.at[ stack_idx].set(right_idx)
                )
            ),
            stack.at[ stack_idx].set(-1)
        )

        return new_stack_idx, new_stack, hits_with_both, no_hits_with_both, loop_idx + 1        
    
    N_points = point.shape[0]
    stack = jnp.full(( stack_size,  ), -1) 
    hits  = jnp.full(( max_hit_size,), -1)    
    
    initial_stack = stack.at[..., 0].set(N_leafs)  # start with root node
    initial_state = (0, initial_stack, hits,0,0)    
    
    final_stack_idx, final_stack, final_hits, final_no_hits, n_loops = jax.lax.while_loop(
        condition,
        loop,
        initial_state)
    return final_hits

ray_traversal_bvh_vectorized = jax.jit(jnp.vectorize(ray_traversal_bvh_single, excluded=(0,), signature='(3),(3)->(max_hit_size)'))
# =========================================================================================================

@jax.jit
def ray_triangle_intersection_single(point, direction, triangle, eps=1e-8):
    """
    Compute intersections of one ray with one trianlge

    Parameters
    ----------
    origin:    (3,)
        Ray origin
    direction: (3,)

    Returns
    -------
    t:        (T,) distance along ray (jnp.inf if no hit)
    u, v:     (T,) barycentric coordinates
    mask:     (T,) boolean array, True if hit
    """
    v0 = triangle[0, :]
    v1 = triangle[1, :]
    v2 = triangle[2, :]

    e1 = v1 - v0 # (3,)
    e2 = v2 - v0 # (3,)

    pvec = jnp.cross(direction, e2) #(3,)
    det  = jnp.dot(e1, pvec)  #(,)

    valid_det = jnp.abs(det) > eps #(,)
    inv_det   = jnp.where(valid_det, 1.0 / det, 0.0) #(,)

    tvec      = point - v0 # (3,)
    u         = jnp.dot(tvec, pvec) * inv_det # (,)

    qvec = jnp.cross(tvec, e1) # (3,)
    v = jnp.dot(direction, qvec) * inv_det # (,)

    t =  jnp.dot(e2, qvec) * inv_det #(,)

    mask = (valid_det &
            (u >= 0.0) &
            (v >= 0.0) &
            ((u + v) <= 1.0) &
            (t > eps))

    t = jnp.where(mask, t, jnp.inf)
    return t

ray_triangle_intersection_vectorized = jax.jit(jnp.vectorize(ray_triangle_intersection_single, signature='(3),(3),(3,3)->()'))

@jax.jit
def find_minimum_distance_to_mesh(points, directions, mesh):
    bvh = build_lbvh(mesh[0], mesh[1]) # BVH
    hits_possible = ray_traversal_bvh_vectorized(bvh, points, directions)    
    mesh_total = jnp.moveaxis(mesh[0][mesh[1][bvh.order[hits_possible]]], -3, 0) # we move the possible hits to the front.
    return jnp.nanmin(ray_triangle_intersection_vectorized(points, directions, mesh_total), axis=0)

# theta_rt = jnp.linspace(0, 2*jnp.pi, n_theta_rt, endpoint=False)
# phi_rt   = jnp.linspace(0, 2*jnp.pi / fs_jax.settings.nfp, n_phi_rt, endpoint=False)
# theta_mg, phi_mg = jnp.meshgrid(theta_rt, phi_rt, indexing='ij')

# positions_origins = fs_jax.cartesian_position(1.0, theta_mg, phi_mg)  # just to compile
# directions        = fs_jax.cartesian_position(2.0, theta_mg, phi_mg) - positions_origins  # just to compile
# total_triangles = positions_standard_ordering.reshape(-1,3)[vertices]
# t,u,v , mask = ray_triangle_intersect_single(positions_origins[0,0], directions[0,0], total_triangles)  # just to compile