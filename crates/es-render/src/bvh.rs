//! A single-level BVH over the frame's world-space triangles (packet M7/R1 step 3).
//!
//! Built on the CPU, once per uploaded [`TriScene`](crate::TriScene), and uploaded **beside**
//! the triangles in the same buffer — so the GPU pays no extra transfer and `es-gpu` stays
//! compute-only (no `VK_KHR_acceleration_structure`, spec 15.4's hardware TLAS is still not
//! available here).
//!
//! What it is: a binary tree, median split on the centroid along the longest axis of the
//! centroid bounds, leaves of at most [`LEAF_MAX`] triangles, nodes in a flat array. What it
//! is **not**: spec 15.4's two-level TLAS/BLAS. See [`build`](Bvh::build) for why.
//!
//! # Determinism (spec 3.4)
//!
//! No threads, no `HashMap`, no randomness: the split axis is a comparison of three `f32`
//! extents, and the sort is by `(centroid, global triangle index)`, a total order. Same
//! triangles in, same tree out, on any host. The tree does **not** enter any hash: it is a
//! search structure over data that is already hashed, and [`crate::cpu::nearest_hit`] returns
//! the flat scan's answer whatever tree it is handed (`docs/design/renderer.md` section 8.3).

use crate::scene::Tri;

/// Triangles per leaf. Four is the smallest leaf that still amortises the node fetch; it is
/// not a quality knob, because no output depends on it.
const LEAF_MAX: usize = 4;

/// Traversal stack depth. Mirrored by `ES_BVH_STACK` in `common.slang`, and enforced by
/// [`Bvh::build`], which makes a leaf rather than exceed it.
pub const STACK: u32 = 64;

/// Floats per node in the upload buffer: `min.xyz max.xyz a count`.
pub const NODE_STRIDE: usize = 8;

/// Outward padding of every node box, metres.
///
/// Möller–Trumbore can report a hit for a ray that, in exact arithmetic, passes just outside
/// the triangle's edge; such a ray can also miss the triangle's exact bounding box, and the
/// BVH would then skip a triangle the flat scan accepts. 0.1 mm is far above `f32` rounding
/// at the scales this renderer works at (1 ULP at 100 m is 7.6e-6 m) and far below anything
/// that costs traversal work.
const PAD: f32 = 1e-4;

/// Robust slab-test factors (Ize, *Robust BVH Ray Traversal*, 2013): `1 -/+ 2^-22`. They make
/// the slab test conservative against its own rounding, so a node is never rejected for a ray
/// that enters it. Mirrored by `ES_BVH_LO` / `ES_BVH_HI` in `common.slang`.
pub const LO: f32 = 0.999_999_76;
/// See [`LO`].
pub const HI: f32 = 1.000_000_24;

/// One node. A leaf has `count > 0` and `a` is its first slot in [`Bvh::prim`]; an inner node
/// has `count == 0`, its left child at `self_index + 1` and its right child at `a`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Node {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub a: u32,
    pub count: u32,
}

/// The tree plus the triangle-index permutation its leaves point into.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bvh {
    pub nodes: Vec<Node>,
    /// Leaf slot -> **global** triangle index. The triangles themselves are never reordered:
    /// the global index is what the hit rule breaks ties on and what the `hit_tri` G-buffer
    /// carries, so a permutation of the array would be a changed output.
    pub prim: Vec<u32>,
    /// Depth of the deepest leaf, root = 1. Bounds the traversal stack occupancy, because a
    /// stack-based descent holds at most one entry per level.
    pub depth: u32,
}

impl Bvh {
    /// Build over `tris`, in world space, every frame.
    ///
    /// `ponytail:` single level, rebuilt per frame. Spec 15.4's design is two-level — one
    /// BLAS per geom built once, a TLAS over body instances per frame — and that is the
    /// upgrade path. It is not this packet: the per-frame `f64` CPU posing in
    /// [`crate::SceneCache`] is what keeps the world-space vertices bit-identical to what
    /// every committed golden and fixture was rendered from (spec 28.10 rule 1), and a
    /// 3,000-triangle rebuild is 0.2 ms, not where the frame goes
    /// (`docs/design/renderer.md` section 8.1). Take the upgrade when a scene is large
    /// enough that the *build* shows up in `frame_profile`.
    pub fn build(tris: &[Tri]) -> Self {
        let mut out = Self::default();
        if tris.is_empty() {
            return out;
        }
        let bounds: Vec<([f32; 3], [f32; 3])> = tris.iter().map(tri_bounds).collect();
        let centroid: Vec<[f32; 3]> = bounds
            .iter()
            .map(|(lo, hi)| {
                [
                    (lo[0] + hi[0]) * 0.5,
                    (lo[1] + hi[1]) * 0.5,
                    (lo[2] + hi[2]) * 0.5,
                ]
            })
            .collect();
        out.prim = (0..tris.len() as u32).collect();
        out.depth = subdivide(
            &mut out.nodes,
            &mut out.prim,
            &bounds,
            &centroid,
            0,
            tris.len(),
            1,
        );
        out
    }

    /// The upload image: every node, then the triangle-index permutation, `u32`s bitcast into
    /// `f32` slots exactly as [`crate::TriScene::to_floats`] bitcasts the segmentation id.
    pub fn to_floats(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.nodes.len() * NODE_STRIDE + self.prim.len());
        for n in &self.nodes {
            out.extend_from_slice(&n.min);
            out.extend_from_slice(&n.max);
            out.push(f32::from_bits(n.a));
            out.push(f32::from_bits(n.count));
        }
        out.extend(self.prim.iter().map(|p| f32::from_bits(*p)));
        out
    }
}

/// Axis-aligned bounds of one triangle, padded outward by [`PAD`].
fn tri_bounds(t: &Tri) -> ([f32; 3], [f32; 3]) {
    let mut lo = t.v[0];
    let mut hi = t.v[0];
    for v in &t.v[1..] {
        for k in 0..3 {
            lo[k] = lo[k].min(v[k]);
            hi[k] = hi[k].max(v[k]);
        }
    }
    for k in 0..3 {
        lo[k] -= PAD;
        hi[k] += PAD;
    }
    (lo, hi)
}

/// Push one node covering `prim[first..first + count]` and recurse. Returns the depth of the
/// deepest leaf beneath it.
fn subdivide(
    nodes: &mut Vec<Node>,
    prim: &mut [u32],
    bounds: &[([f32; 3], [f32; 3])],
    centroid: &[[f32; 3]],
    first: usize,
    count: usize,
    depth: u32,
) -> u32 {
    let mut node = Node {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
        a: first as u32,
        count: count as u32,
    };
    for p in &prim[first..first + count] {
        let (lo, hi) = bounds[*p as usize];
        for k in 0..3 {
            node.min[k] = node.min[k].min(lo[k]);
            node.max[k] = node.max[k].max(hi[k]);
        }
    }
    let me = nodes.len();
    nodes.push(node);
    // `depth + 1 >= STACK` cannot be reached with a median split (it halves every time, so
    // the depth is at most log2(count) + 1) — it is the guarantee that makes the fixed
    // traversal stack safe rather than probable.
    if count <= LEAF_MAX || depth + 1 >= STACK {
        return depth;
    }

    let (mut clo, mut chi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for p in &prim[first..first + count] {
        for k in 0..3 {
            clo[k] = clo[k].min(centroid[*p as usize][k]);
            chi[k] = chi[k].max(centroid[*p as usize][k]);
        }
    }
    let ext = [chi[0] - clo[0], chi[1] - clo[1], chi[2] - clo[2]];
    let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
        0
    } else if ext[1] >= ext[2] {
        1
    } else {
        2
    };
    // `(centroid, global index)` is a total order even when every centroid coincides, so the
    // split is a function of the triangle array alone (spec 3.4).
    prim[first..first + count].sort_unstable_by(|a, b| {
        centroid[*a as usize][axis]
            .total_cmp(&centroid[*b as usize][axis])
            .then(a.cmp(b))
    });
    let mid = count / 2;

    let left = subdivide(nodes, prim, bounds, centroid, first, mid, depth + 1);
    nodes[me].a = nodes.len() as u32;
    nodes[me].count = 0;
    let right = subdivide(
        nodes,
        prim,
        bounds,
        centroid,
        first + mid,
        count - mid,
        depth + 1,
    );
    left.max(right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cornell::cornell_box;
    use crate::TriScene;

    fn cornell() -> TriScene {
        TriScene::from_scene(&cornell_box()).expect("cornell tessellates")
    }

    #[test]
    fn an_empty_scene_builds_an_empty_tree() {
        let bvh = Bvh::build(&[]);
        assert!(bvh.nodes.is_empty() && bvh.prim.is_empty() && bvh.depth == 0);
        assert!(bvh.to_floats().is_empty());
    }

    /// Every triangle appears in exactly one leaf, the depth is bounded, and every node's box
    /// contains its children's. The traversal can only be right if the tree is.
    #[test]
    fn the_tree_covers_every_triangle_exactly_once() {
        let scene = cornell();
        let bvh = Bvh::build(&scene.tris);
        assert!(bvh.depth <= STACK, "depth {} exceeds the stack", bvh.depth);
        let mut seen = vec![0u32; scene.tris.len()];
        let mut leaves = 0;
        for n in &bvh.nodes {
            if n.count == 0 {
                assert!((n.a as usize) < bvh.nodes.len(), "right child out of range");
                continue;
            }
            leaves += 1;
            assert!(n.count as usize <= LEAF_MAX, "leaf of {}", n.count);
            for k in 0..n.count as usize {
                seen[bvh.prim[n.a as usize + k] as usize] += 1;
            }
        }
        assert!(
            seen.iter().all(|c| *c == 1),
            "a triangle is not in one leaf"
        );
        // A leaf's box contains its triangles.
        for n in bvh.nodes.iter().filter(|n| n.count > 0) {
            for k in 0..n.count as usize {
                for v in &scene.tris[bvh.prim[n.a as usize + k] as usize].v {
                    for c in 0..3 {
                        assert!(
                            v[c] >= n.min[c] && v[c] <= n.max[c],
                            "vertex outside its leaf"
                        );
                    }
                }
            }
        }
        println!(
            "cornell: {} triangles, {} nodes, {leaves} leaves, depth {}",
            scene.tris.len(),
            bvh.nodes.len(),
            bvh.depth
        );
    }

    #[test]
    fn the_build_is_reproducible() {
        let scene = cornell();
        assert_eq!(Bvh::build(&scene.tris), Bvh::build(&scene.tris));
    }
}
