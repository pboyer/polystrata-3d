//! # 3D Planar B-Rep Boolean & Healing Kernel
//!
//! Standalone 3D Stratified Tolerance Complex (`polystrata`) for 3D Planar Boundary-Representation
//! (B-Rep) solids with zero external dependencies.
//!
//! ## 4-Stage Coupled 3D Stratified Tolerance Complex Architecture
//! 1. **Stage 0 & 1 — 3D Normalization, Carrier Plane Clustering, & Locus Injection**:
//!    Centers the 3D bounding box at `(0, 0, 0)`, clusters coplanar input faces into canonical
//!    carrier planes `π_c = (O_c, U_c, V_c, N_c, d_c)`, and injects all input face boundary loops
//!    plus all 1D plane-plane intersection chords `L_{ab} = π_a ∩ π_b` into a single global 3D
//!    `(0, 1)`-stratum graph `(Vert3D, Edge3D)`.
//! 2. **Stage 2 — Coupled 3D 0-Stratum & 1-Stratum Fixpoint Consolidation**:
//!    Iteratively merges overlapping 3D vertex balls (`r_max`-capped), contracts micro-edges,
//!    propagates affine plane incidence (`u, v ∈ π_c ⟹ (u, v) ∈ π_c`), splits 3D & co-carrier 2D
//!    T-junctions and segment intersections, and projects vertices onto their incident carrier
//!    planes via regularized 3×3 quadric minimization inside each vertex ball.
//! 3. **Stage 3 — Combinatorial 2-Stratum (`∂₂`) & 3-Stratum (`∂₃`) Cellular Extraction**:
//!    - **Stage 3a (`∂₂`)**: On each carrier plane `π_c`, wires consolidated 3D edges into planar
//!      half-edge cycles to extract minimal 2D `FacePatch3D` sheets with exact integer surface
//!      winding jumps `Δw_b(f_m)`.
//!    - **Stage 3b (`∂₃`)**: Strips open degree-1 flaps (`∂₂` 2-core) and radially sorts incident
//!      face patches around every 3D edge `e` by dihedral angle `θ` around `t_e`, uniting the
//!      two CoFaces bounding each dihedral wedge into closed 3D Volume Cells `S_v` (`∂₂ ∘ ∂₃ ≡ 0`).
//! 4. **Stage 4 — Volume Cell Winding Classification, `b₂ = ∂₃ · c₃`, & Option B Planarity**:
//!    Classifies each 3D Volume Cell `S_v` once, extracts surviving boundary patches where
//!    `keep(S_back) != keep(S_front)`, cancels internal coplanar seams, simplifies global 3D
//!    degree-2 collinear vertices, and splits any high-tolerance warped face along boundary-vertex
//!    diagonals into strictly planar triangles (Option B).

/// A 2D point in double-precision (`f64`) carrier-plane coordinates.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Winding rule used to classify the interior of each input solid from its signed winding number `w`:
/// - [`WindingRule::Positive`]: `w >= 1` (standard CAD/B-Rep convention where outward shell = +1, void = -1)
/// - [`WindingRule::NonZero`]: `w != 0`
/// - [`WindingRule::EvenOdd`]: `|w| % 2 == 1`
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum WindingRule {
    #[default]
    Positive = 0,
    NonZero = 1,
    EvenOdd = 2,
}

impl WindingRule {
    #[inline]
    pub const fn code(self) -> u32 {
        self as u32
    }

    #[inline]
    pub fn apply(self, w: f64) -> bool {
        match self {
            WindingRule::Positive => w >= 0.5,
            WindingRule::NonZero => w.abs() >= 0.5,
            WindingRule::EvenOdd => (w.round() as i64).abs() % 2 == 1,
        }
    }
}

/// Standard 2-body Boolean operation selector.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BooleanOp {
    /// Set union `A ∪ B`.
    Union = 0,
    /// Set intersection `A ∩ B`.
    Intersect = 1,
    /// Set difference `A \ B`.
    Difference = 2,
    /// Symmetric difference `A ⊕ B`.
    Xor = 3,
}

impl BooleanOp {
    #[inline]
    pub const fn code(self) -> u32 {
        self as u32
    }
}

/// Configuration options for the 3D Stratified Tolerance Complex kernel.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KernelOptions {
    /// Base snap tolerance `r₀`.
    /// When `None`, the kernel automatically selects a scale-aware tolerance
    /// (`1e-9 × bounding_box_diagonal`) and caps vertex tolerance ball growth at `r_max = 4 × r₀`.
    pub snap_tolerance: Option<f64>,
    /// Winding rule used to determine inside/outside for each input solid.
    /// Defaults to [`WindingRule::Positive`].
    pub winding_rule: WindingRule,
    /// Whether to collapse collinear degree-2 vertices along output boundary loops.
    /// Defaults to `true`.
    pub simplify_collinear: bool,
}

impl Default for KernelOptions {
    #[inline]
    fn default() -> Self {
        Self {
            snap_tolerance: None,
            winding_rule: WindingRule::Positive,
            simplify_collinear: true,
        }
    }
}

/// A 3D point in double-precision (`f64`) world coordinates.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3D {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    #[inline]
    pub fn normalized(self) -> Option<Self> {
        let len = self.norm();
        if len <= 1e-15 {
            None
        } else {
            Some(self.scale(1.0 / len))
        }
    }
}

impl From<[f64; 3]> for Point3D {
    #[inline]
    fn from(arr: [f64; 3]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }
}

impl From<Point3D> for [f64; 3] {
    #[inline]
    fn from(p: Point3D) -> Self {
        [p.x, p.y, p.z]
    }
}

/// A planar 3D face consisting of one counter-clockwise (`CCW`) outer loop (viewed from outside
/// along the outward `normal`) and zero or more clockwise (`CW`) interior hole loops.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Face3D {
    pub outer: Vec<Point3D>,
    pub holes: Vec<Vec<Point3D>>,
    pub normal: Point3D,
}

impl Face3D {
    /// Construct a simple planar face from an outer loop of 3D vertices, computing its outward
    /// unit normal via Newell's method.
    pub fn from_outer<P: Into<Point3D> + Copy>(outer_pts: &[P]) -> Self {
        let outer: Vec<Point3D> = outer_pts.iter().copied().map(Into::into).collect();
        let normal = compute_newell_normal(&outer).unwrap_or(Point3D::new(0.0, 0.0, 1.0));
        Self {
            outer,
            holes: Vec::new(),
            normal,
        }
    }

    /// Construct a planar face with holes, computing its outward unit normal from the outer loop.
    pub fn new(outer: Vec<Point3D>, holes: Vec<Vec<Point3D>>) -> Self {
        let normal = compute_newell_normal(&outer).unwrap_or(Point3D::new(0.0, 0.0, 1.0));
        Self {
            outer,
            holes,
            normal,
        }
    }

    /// Signed area of the face (outer shell area minus hole areas) projected onto its unit normal.
    pub fn area(&self) -> f64 {
        let mut total = loop_area_along_normal(&self.outer, self.normal);
        for hole in &self.holes {
            total += loop_area_along_normal(hole, self.normal);
        }
        total.abs()
    }
}

/// A 3D Planar B-Rep solid (or multi-shell collection) represented as a set of oriented planar faces.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Solid3D {
    pub faces: Vec<Face3D>,
}

impl Solid3D {
    pub fn new(faces: Vec<Face3D>) -> Self {
        Self { faces }
    }

    /// Construct an axis-aligned box `[min_x, max_x] × [min_y, max_y] × [min_z, max_z]` with 6 CCW outward planar faces.
    pub fn from_box(min: [f64; 3], max: [f64; 3]) -> Self {
        let (x0, y0, z0) = (min[0], min[1], min[2]);
        let (x1, y1, z1) = (max[0], max[1], max[2]);
        let p000 = Point3D::new(x0, y0, z0);
        let p100 = Point3D::new(x1, y0, z0);
        let p110 = Point3D::new(x1, y1, z0);
        let p010 = Point3D::new(x0, y1, z0);
        let p001 = Point3D::new(x0, y0, z1);
        let p101 = Point3D::new(x1, y0, z1);
        let p111 = Point3D::new(x1, y1, z1);
        let p011 = Point3D::new(x0, y1, z1);

        let faces = vec![
            // -Z bottom
            Face3D::from_outer(&[p000, p010, p110, p100]),
            // +Z top
            Face3D::from_outer(&[p001, p101, p111, p011]),
            // -Y front
            Face3D::from_outer(&[p000, p100, p101, p001]),
            // +Y back
            Face3D::from_outer(&[p110, p010, p011, p111]),
            // -X left
            Face3D::from_outer(&[p010, p000, p001, p011]),
            // +X right
            Face3D::from_outer(&[p100, p110, p111, p101]),
        ];
        Self { faces }
    }

    /// Construct a right prism by extruding a 2D polygon along the Z axis from `z_min` to `z_max`.
    pub fn from_extruded_polygon(
        outer_xy: &[[f64; 2]],
        holes_xy: &[Vec<[f64; 2]>],
        z_min: f64,
        z_max: f64,
    ) -> Self {
        let mut faces = Vec::new();
        if outer_xy.len() < 3 || z_max <= z_min {
            return Self { faces };
        }

        // Ensure outer_xy is CCW
        let mut outer_ccw: Vec<[f64; 2]> = outer_xy.to_vec();
        if signed_area_2d_arr(&outer_ccw) < 0.0 {
            outer_ccw.reverse();
        }

        let mut holes_cw: Vec<Vec<[f64; 2]>> = Vec::with_capacity(holes_xy.len());
        for h in holes_xy {
            if h.len() < 3 {
                continue;
            }
            let mut h_cw = h.clone();
            if signed_area_2d_arr(&h_cw) > 0.0 {
                h_cw.reverse();
            }
            holes_cw.push(h_cw);
        }

        // Top face (+Z): outer is CCW, holes are CW
        let top_outer: Vec<Point3D> = outer_ccw
            .iter()
            .map(|p| Point3D::new(p[0], p[1], z_max))
            .collect();
        let top_holes: Vec<Vec<Point3D>> = holes_cw
            .iter()
            .map(|h| h.iter().map(|p| Point3D::new(p[0], p[1], z_max)).collect())
            .collect();
        faces.push(Face3D {
            outer: top_outer,
            holes: top_holes,
            normal: Point3D::new(0.0, 0.0, 1.0),
        });

        // Bottom face (-Z): reverse orientation so outward normal is -Z
        let bot_outer: Vec<Point3D> = outer_ccw
            .iter()
            .rev()
            .map(|p| Point3D::new(p[0], p[1], z_min))
            .collect();
        let bot_holes: Vec<Vec<Point3D>> = holes_cw
            .iter()
            .map(|h| {
                h.iter()
                    .rev()
                    .map(|p| Point3D::new(p[0], p[1], z_min))
                    .collect()
            })
            .collect();
        faces.push(Face3D {
            outer: bot_outer,
            holes: bot_holes,
            normal: Point3D::new(0.0, 0.0, -1.0),
        });

        // Side walls for outer loop (CCW in XY -> outward normal points right of edge)
        append_extrusion_walls(&outer_ccw, z_min, z_max, &mut faces);
        // Side walls for holes (CW in XY -> outward normal points into the hole!)
        for h_cw in &holes_cw {
            append_extrusion_walls(h_cw, z_min, z_max, &mut faces);
        }

        Self { faces }
    }

    /// Compute total signed volume via the Divergence Theorem:
    /// `V = (1/6) ∑_{F} ∑_{ring ∈ F} ∑_{i} p_0 · (ring[i] × ring[i+1])`.
    pub fn volume(&self) -> f64 {
        let mut vol6 = 0.0;
        for face in &self.faces {
            if let Some(&p0) = face.outer.first() {
                let n = face.outer.len();
                for i in 1..n.saturating_sub(1) {
                    vol6 += p0.dot(face.outer[i].cross(face.outer[i + 1]));
                }
                for hole in &face.holes {
                    let hn = hole.len();
                    for i in 0..hn {
                        vol6 += p0.dot(hole[i].cross(hole[(i + 1) % hn]));
                    }
                }
            }
        }
        vol6 / 6.0
    }

    /// Compute total surface area of all faces in the solid.
    pub fn surface_area(&self) -> f64 {
        self.faces.iter().map(Face3D::area).sum()
    }
}

fn append_extrusion_walls(ring: &[[f64; 2]], z_min: f64, z_max: f64, faces: &mut Vec<Face3D>) {
    let n = ring.len();
    for i in 0..n {
        let a = ring[i];
        let b = ring[(i + 1) % n];
        if (b[0] - a[0]).hypot(b[1] - a[1]) <= 1e-12 {
            continue;
        }
        let p0 = Point3D::new(a[0], a[1], z_min);
        let p1 = Point3D::new(b[0], b[1], z_min);
        let p2 = Point3D::new(b[0], b[1], z_max);
        let p3 = Point3D::new(a[0], a[1], z_max);
        faces.push(Face3D::from_outer(&[p0, p1, p2, p3]));
    }
}

/// Result of a 3D Planar B-Rep Boolean or healing operation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BRepResult3D {
    pub solid: Solid3D,
    pub total_volume: f64,
    pub total_surface_area: f64,
    /// Number of consolidated 3D vertices on the output boundary.
    pub vertex_count: usize,
    /// Number of consolidated 3D edges on the output boundary.
    pub edge_count: usize,
    /// Number of unmatched half-edges (0 means strictly watertight 2-manifold boundary `∂₁ · ∂₂ = 0`).
    pub unmatched_half_edge_count: usize,
}

#[inline]
fn signed_area_2d_arr(pts: &[[f64; 2]]) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * sum
}

#[inline]
fn signed_area_2d_pts(pts: &[Point2D]) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        sum += a.x * b.y - b.x * a.y;
    }
    0.5 * sum
}

/// Compute the unit normal of a 3D polygon loop via Newell's method.
pub fn compute_newell_normal(pts: &[Point3D]) -> Option<Point3D> {
    let n = pts.len();
    if n < 3 {
        return None;
    }
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;
    for i in 0..n {
        let cur = pts[i];
        let nxt = pts[(i + 1) % n];
        nx += (cur.y - nxt.y) * (cur.z + nxt.z);
        ny += (cur.z - nxt.z) * (cur.x + nxt.x);
        nz += (cur.x - nxt.x) * (cur.y + nxt.y);
    }
    Point3D::new(nx, ny, nz).normalized()
}

fn loop_area_along_normal(pts: &[Point3D], normal: Point3D) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let p0 = pts[0];
    let mut sum = Point3D::default();
    for i in 1..(n - 1) {
        let u = pts[i].sub(p0);
        let v = pts[i + 1].sub(p0);
        sum = sum.add(u.cross(v));
    }
    0.5 * sum.dot(normal)
}

/// Canonicalizes a unit normal `n` and signed plane offset `d = n · x` so that the dominant coordinate
/// of `n` is positive. Returns `(canonical_n, canonical_d, orient_sign)`.
fn canonicalize_plane(n: Point3D, d: f64) -> (Point3D, f64, i32) {
    let ax = n.x.abs();
    let ay = n.y.abs();
    let az = n.z.abs();
    let sign = if az >= ax && az >= ay {
        if n.z >= 0.0 {
            1
        } else {
            -1
        }
    } else if ay >= ax {
        if n.y >= 0.0 {
            1
        } else {
            -1
        }
    } else if n.x >= 0.0 {
        1
    } else {
        -1
    };

    if sign == 1 {
        (n, d, 1)
    } else {
        (n.scale(-1.0), -d, -1)
    }
}

/// Build an orthonormal 2D tangent frame `(u_axis, v_axis)` for unit normal `n` such that `u × v = n`.
fn build_plane_frame(n: Point3D) -> (Point3D, Point3D) {
    let helper = if n.x.abs() < 0.8 {
        Point3D::new(1.0, 0.0, 0.0)
    } else {
        Point3D::new(0.0, 1.0, 0.0)
    };
    let u = n.cross(helper).normalized().unwrap();
    let v = n.cross(u).normalized().unwrap();
    (u, v)
}

struct CanonicalPlane {
    origin: Point3D,
    u_axis: Point3D,
    v_axis: Point3D,
    normal: Point3D,
    d: f64,
    min_pt: Point3D,
    max_pt: Point3D,
}

impl CanonicalPlane {
    #[inline]
    fn project_2d(&self, p: Point3D) -> Point2D {
        let rel = p.sub(self.origin);
        Point2D {
            x: rel.dot(self.u_axis),
            y: rel.dot(self.v_axis),
        }
    }

    #[inline]
    fn unproject_3d(&self, u: f64, v: f64) -> Point3D {
        self.origin
            .add(self.u_axis.scale(u))
            .add(self.v_axis.scale(v))
    }
}

struct NormalizedInputFace {
    body_index: usize,
    plane_id: usize,
    orient_sign: i32,
    area_2d: f64,
    loops_2d: Vec<Vec<Point2D>>,
    loops_3d: Vec<Vec<Point3D>>,
    min_pt: Point3D,
    max_pt: Point3D,
}

#[derive(Clone, Copy, Debug)]
struct IdxPoint2D {
    x: f64,
    y: f64,
    id: u32,
}

impl IdxPoint2D {
    #[inline]
    fn pt(self) -> Point2D {
        Point2D {
            x: self.x,
            y: self.y,
        }
    }
}

fn clean_indexed_ring_2d(pts: &[IdxPoint2D]) -> Vec<IdxPoint2D> {
    let mut out: Vec<IdxPoint2D> = Vec::with_capacity(pts.len());
    for &p in pts {
        if let Some(&last) = out.last() {
            if p.id == last.id || (p.x - last.x).hypot(p.y - last.y) <= 1e-11 {
                continue;
            }
        }
        out.push(p);
    }
    while out.len() >= 2 {
        let a = out[0];
        let b = *out.last().unwrap();
        if a.id == b.id || (a.x - b.x).hypot(a.y - b.y) <= 1e-11 {
            out.pop();
        } else {
            break;
        }
    }

    // Only strip consecutive duplicate vertices and retracted 2-edge hairpins (`prev == next`).
    // Never strip forward collinear boundary vertices (`prev != next`), so that all shared 3D
    // boundary half-edges are strictly preserved during face triangulation!
    let mut changed = true;
    while changed && out.len() >= 3 {
        changed = false;
        let n = out.len();
        let mut next_ring: Vec<IdxPoint2D> = Vec::with_capacity(n);
        for i in 0..n {
            let prev = if let Some(&last) = next_ring.last() {
                last
            } else {
                out[(i + n - 1) % n]
            };
            let curr = out[i];
            let next = out[(i + 1) % n];
            if curr.id == prev.id || (curr.x - prev.x).hypot(curr.y - prev.y) <= 1e-11 {
                changed = true;
                continue;
            }
            if prev.id == next.id || (prev.x - next.x).hypot(prev.y - next.y) <= 1e-11 {
                changed = true;
                continue;
            }
            next_ring.push(curr);
        }
        out = next_ring;
    }
    out
}

#[inline]
fn signed_area_2d_idx_pts(ring: &[IdxPoint2D]) -> f64 {
    let n = ring.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let a = ring[i];
        let b = ring[(i + 1) % n];
        s += a.x * b.y - b.x * a.y;
    }
    0.5 * s
}

/// Check if point `m` lies inside the interior CCW angular wedge at vertex `outer[idx]`.
#[inline]
fn point_in_vertex_cone_2d(prev: Point2D, v: Point2D, next: Point2D, m: Point2D) -> bool {
    let turn = cross_2d(prev, v, next);
    let c_prev = cross_2d(prev, v, m);
    let c_next = cross_2d(v, next, m);
    if turn >= -1e-12 {
        // Convex corner: m must be to the left of both (prev -> v) and (v -> next)
        c_prev >= -1e-11 && c_next >= -1e-11
    } else {
        // Reflex corner: m must not be strictly in the exterior CW cone
        c_prev >= -1e-11 || c_next >= -1e-11
    }
}

/// Split any figure-8 / self-pinching 2D ring (where a vertex occurs more than once)
/// into independent simple sub-loops before ear-clipping.
fn decompose_pinched_indexed_ring_2d(ring: &[IdxPoint2D]) -> Vec<Vec<IdxPoint2D>> {
    if ring.len() < 3 {
        return Vec::new();
    }
    let mut stack: Vec<IdxPoint2D> = Vec::with_capacity(ring.len());
    let mut sub_loops: Vec<Vec<IdxPoint2D>> = Vec::new();

    for &pt in ring {
        if let Some(pos) = stack
            .iter()
            .position(|p| p.id == pt.id || (p.x - pt.x).hypot(p.y - pt.y) <= 1e-11)
        {
            if stack.len() - pos >= 3 {
                let sub = clean_indexed_ring_2d(&stack[pos..]);
                if sub.len() >= 3 {
                    sub_loops.push(sub);
                }
            }
            stack.truncate(pos + 1);
        } else {
            stack.push(pt);
        }
    }

    if stack.len() >= 3 {
        let sub = clean_indexed_ring_2d(&stack);
        if sub.len() >= 3 {
            sub_loops.push(sub);
        }
    }

    sub_loops
}

/// Triangulate an indexed 2D polygon with holes `(outer_ccw, holes_cw)` via bridge-stitching + ear-clipping,
/// returning vertex ID triples `[id_a, id_b, id_c]` that strictly preserve all boundary half-edges.
fn triangulate_indexed_polygon_2d(
    outer: &[IdxPoint2D],
    holes: &[Vec<IdxPoint2D>],
) -> Vec<[u32; 3]> {
    let ring = clean_indexed_ring_2d(outer);
    if ring.len() < 3 {
        return Vec::new();
    }

    let mut simple_shells = decompose_pinched_indexed_ring_2d(&ring);
    if simple_shells.is_empty() {
        return Vec::new();
    }

    if !holes.is_empty() {
        let mut sorted_holes: Vec<Vec<IdxPoint2D>> = Vec::new();
        for h in holes {
            let ch = clean_indexed_ring_2d(h);
            if ch.len() < 3 {
                continue;
            }
            sorted_holes.push(ch);
        }

        sorted_holes.sort_by(|a, b| {
            let max_xa = a.iter().fold(f64::NEG_INFINITY, |acc, p| acc.max(p.x));
            let max_xb = b.iter().fold(f64::NEG_INFINITY, |acc, p| acc.max(p.x));
            max_xb
                .partial_cmp(&max_xa)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        for hole in sorted_holes {
            let mut target_idx = 0usize;
            if simple_shells.len() > 1 {
                let test_pt = hole[0].pt();
                let mut best_area = f64::INFINITY;
                for (s_idx, sh) in simple_shells.iter().enumerate() {
                    let sh_pts: Vec<Point2D> = sh.iter().map(|p| p.pt()).collect();
                    let s_area = signed_area_2d_pts(&sh_pts).abs();
                    if loops_winding_2d(test_pt, core::slice::from_ref(&sh_pts)) != 0
                        && s_area < best_area
                    {
                        best_area = s_area;
                        target_idx = s_idx;
                    }
                }
            }
            splice_indexed_hole_into_outer(&mut simple_shells[target_idx], &hole);
        }
    }

    let mut all_tris = Vec::new();
    for sh in &simple_shells {
        all_tris.extend(ear_clip_indexed_ring(sh));
    }
    all_tris
}

/// Triangulate a 2D polygon with holes `(outer_ccw, holes_cw)` via bridge-stitching + ear-clipping.
fn triangulate_polygon_2d(outer: &[Point2D], holes: &[Vec<Point2D>]) -> Vec<[Point2D; 3]> {
    let mut all_pts: Vec<Point2D> = Vec::new();
    let mut idx_outer: Vec<IdxPoint2D> = outer
        .iter()
        .map(|&p| {
            let id = all_pts.len() as u32;
            all_pts.push(p);
            IdxPoint2D { x: p.x, y: p.y, id }
        })
        .collect();
    if signed_area_2d_idx_pts(&idx_outer) < 0.0 {
        idx_outer.reverse();
    }
    let idx_holes: Vec<Vec<IdxPoint2D>> = holes
        .iter()
        .map(|h| {
            let mut ih: Vec<IdxPoint2D> = h
                .iter()
                .map(|&p| {
                    let id = all_pts.len() as u32;
                    all_pts.push(p);
                    IdxPoint2D { x: p.x, y: p.y, id }
                })
                .collect();
            if signed_area_2d_idx_pts(&ih) > 0.0 {
                ih.reverse();
            }
            ih
        })
        .collect();

    triangulate_indexed_polygon_2d(&idx_outer, &idx_holes)
        .into_iter()
        .map(|[a, b, c]| [all_pts[a as usize], all_pts[b as usize], all_pts[c as usize]])
        .collect()
}

fn splice_indexed_hole_into_outer(outer: &mut Vec<IdxPoint2D>, hole: &[IdxPoint2D]) {
    if hole.is_empty() || outer.len() < 3 {
        return;
    }

    // Check if `hole` shares a vertex with `outer`.
    // If so, find the occurrence of that vertex on `outer` whose interior angular cone contains
    // the outgoing hole edge `hole[(hi + 1) % hole.len()]`, and splice `hole` directly there!
    for (hi, &hp) in hole.iter().enumerate() {
        let h_next = hole[(hi + 1) % hole.len()].pt();
        let mut best_oi = None;
        for (oi, &op) in outer.iter().enumerate() {
            if hp.id == op.id || (hp.x - op.x).hypot(hp.y - op.y) <= 1e-11 {
                best_oi = Some(oi);
                let prev = outer[(oi + outer.len() - 1) % outer.len()].pt();
                let next = outer[(oi + 1) % outer.len()].pt();
                if point_in_vertex_cone_2d(prev, op.pt(), next, h_next) {
                    break;
                }
            }
        }
        if let Some(oi) = best_oi {
            let mut combined = Vec::with_capacity(outer.len() + hole.len() + 1);
            combined.extend_from_slice(&outer[..=oi]);
            for k in 1..=hole.len() {
                combined.push(hole[(hi + k) % hole.len()]);
            }
            if oi + 1 < outer.len() {
                combined.extend_from_slice(&outer[(oi + 1)..]);
            }
            *outer = combined;
            return;
        }
    }

    let mut hm_idx = 0usize;
    for i in 1..hole.len() {
        if hole[i].x > hole[hm_idx].x
            || ((hole[i].x - hole[hm_idx].x).abs() <= 1e-12 && hole[i].y < hole[hm_idx].y)
        {
            hm_idx = i;
        }
    }
    let m = hole[hm_idx];

    // Cast horizontal ray +X from `m` to find closest upward edge of `outer`
    let n = outer.len();
    let mut best_t = f64::INFINITY;
    let mut cand_idx = 0usize;

    for i in 0..n {
        let a = outer[i];
        let b = outer[(i + 1) % n];
        if !(a.y <= m.y && b.y > m.y) && !(a.y < m.y && b.y >= m.y) {
            continue;
        }
        let dy = b.y - a.y;
        if dy <= 1e-14 {
            continue;
        }
        let s = (m.y - a.y) / dy;
        let x_int = a.x + s * (b.x - a.x);
        if x_int >= m.x - 1e-11 {
            let dist = (x_int - m.x).max(0.0);
            if dist < best_t {
                best_t = dist;
                if s <= 1e-11 {
                    cand_idx = i;
                } else if s >= 1.0 - 1e-11 {
                    cand_idx = (i + 1) % n;
                } else if a.x > b.x + 1e-11 {
                    cand_idx = i;
                } else {
                    cand_idx = (i + 1) % n;
                }
            }
        }
    }

    if !best_t.is_finite() {
        let mut best_d_sq = f64::INFINITY;
        let mut best_pair = (hm_idx, 0usize);
        for (hi, &hp) in hole.iter().enumerate() {
            for (oi, &op) in outer.iter().enumerate() {
                let d_sq = (hp.x - op.x) * (hp.x - op.x) + (hp.y - op.y) * (hp.y - op.y);
                if d_sq < best_d_sq {
                    best_d_sq = d_sq;
                    best_pair = (hi, oi);
                }
            }
        }
        let (hi, oi) = best_pair;
        let mut combined = Vec::with_capacity(outer.len() + hole.len() + 2);
        combined.extend_from_slice(&outer[..=oi]);
        for k in 0..hole.len() {
            combined.push(hole[(hi + k) % hole.len()]);
        }
        combined.push(hole[hi]);
        combined.push(outer[oi]);
        if oi + 1 < outer.len() {
            combined.extend_from_slice(&outer[(oi + 1)..]);
        }
        *outer = combined;
        return;
    }

    let p = outer[cand_idx];
    let ix = m.x + best_t;
    let mut bridge_pt = p;

    if (p.y - m.y).abs() > 1e-11 && best_t.is_finite() {
        let i_pt = Point2D { x: ix, y: m.y };
        let (tri_a, tri_b, tri_c) = if p.y > m.y {
            (m.pt(), i_pt, p.pt())
        } else {
            (m.pt(), p.pt(), i_pt)
        };
        let mut best_angle = f64::INFINITY;
        let mut best_dist_sq = f64::INFINITY;

        for i in 0..n {
            let prev = outer[(i + n - 1) % n].pt();
            let v = outer[i];
            let next = outer[(i + 1) % n].pt();
            if cross_2d(prev, v.pt(), next) >= 1e-11 {
                continue;
            }
            if (v.x - m.x).hypot(v.y - m.y) <= 1e-11 || (v.x - p.x).hypot(v.y - p.y) <= 1e-11 {
                continue;
            }
            if point_in_triangle_2d(v.pt(), tri_a, tri_b, tri_c) {
                let dx = v.x - m.x;
                let dy = (v.y - m.y).abs();
                let ang = dy.atan2(dx);
                let d_sq = dx * dx + dy * dy;
                if ang < best_angle - 1e-11
                    || ((ang - best_angle).abs() <= 1e-11 && d_sq < best_dist_sq)
                {
                    best_angle = ang;
                    best_dist_sq = d_sq;
                    bridge_pt = v;
                }
            }
        }
    }

    let mut bridge_idx = cand_idx;
    for i in 0..n {
        let v = outer[i];
        if v.id == bridge_pt.id || (v.x - bridge_pt.x).hypot(v.y - bridge_pt.y) <= 1e-11 {
            bridge_idx = i;
            let prev = outer[(i + n - 1) % n].pt();
            let next = outer[(i + 1) % n].pt();
            if point_in_vertex_cone_2d(prev, v.pt(), next, m.pt()) {
                break;
            }
        }
    }

    let mut combined = Vec::with_capacity(outer.len() + hole.len() + 2);
    combined.extend_from_slice(&outer[..=bridge_idx]);
    for k in 0..hole.len() {
        combined.push(hole[(hm_idx + k) % hole.len()]);
    }
    combined.push(hole[hm_idx]);
    combined.push(outer[bridge_idx]);
    if bridge_idx + 1 < outer.len() {
        combined.extend_from_slice(&outer[(bridge_idx + 1)..]);
    }
    *outer = combined;
}

#[inline]
fn cross_2d(a: Point2D, b: Point2D, c: Point2D) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

#[inline]
fn point_in_triangle_2d(p: Point2D, a: Point2D, b: Point2D, c: Point2D) -> bool {
    let c1 = cross_2d(a, b, p);
    let c2 = cross_2d(b, c, p);
    let c3 = cross_2d(c, a, p);
    c1 >= -1e-11 && c2 >= -1e-11 && c3 >= -1e-11
}

fn ear_clip_indexed_ring(ring: &[IdxPoint2D]) -> Vec<[u32; 3]> {
    let mut idxs: Vec<usize> = (0..ring.len()).collect();
    let mut tris = Vec::with_capacity(ring.len().saturating_sub(2));
    let mut guard = ring.len() * ring.len() * 2 + 32;

    while idxs.len() > 3 && guard > 0 {
        guard -= 1;

        // Only strip adjacent duplicate vertices (`curr.id == prev.id`) or closed 2-edge
        // bridge hairpins (`prev.id == next.id`). Never strip forward collinear boundary vertices!
        let mut stripped = false;
        let m = idxs.len();
        for i in 0..m {
            let prev = ring[idxs[(i + m - 1) % m]];
            let curr = ring[idxs[i]];
            let next = ring[idxs[(i + 1) % m]];
            if curr.id == prev.id
                || (curr.x - prev.x).hypot(curr.y - prev.y) <= 1e-11
                || prev.id == next.id
                || (prev.x - next.x).hypot(prev.y - next.y) <= 1e-11
            {
                idxs.remove(i);
                stripped = true;
                break;
            }
        }
        if stripped {
            continue;
        }

        let m = idxs.len();
        if m <= 3 {
            break;
        }
        let mut best_ear: Option<(usize, f64)> = None;

        for pass in 0..2 {
            let tol = if pass == 0 { 1e-11 } else { 1e-7 };
            for i in 0..m {
                let prev = ring[idxs[(i + m - 1) % m]];
                let curr = ring[idxs[i]];
                let next = ring[idxs[(i + 1) % m]];

                let cross = cross_2d(prev.pt(), curr.pt(), next.pt());
                if cross <= 1e-12 {
                    continue;
                }
                let lu = (curr.x - prev.x).hypot(curr.y - prev.y);
                let lv = (next.x - curr.x).hypot(next.y - curr.y);
                let sin_theta = if lu * lv > 1e-20 {
                    cross / (lu * lv)
                } else {
                    0.0
                };

                let prev_prev = ring[idxs[(i + m - 2) % m]];
                let next_next = ring[idxs[(i + 2) % m]];

                // Reject if chord `prev -> next` folds 180° onto `prev -> prev_prev` or `next -> next_next`
                if m > 3 {
                    let l_pp = (prev_prev.x - prev.x).hypot(prev_prev.y - prev.y);
                    let l_nn = (next_next.x - next.x).hypot(next_next.y - next.y);
                    let l_pn = (next.x - prev.x).hypot(next.y - prev.y);
                    if prev_prev.id != next.id && l_pp * l_pn > 1e-20 {
                        let c_p = cross_2d(prev_prev.pt(), prev.pt(), next.pt()).abs();
                        let d_p = (prev_prev.x - prev.x) * (next.x - prev.x)
                            + (prev_prev.y - prev.y) * (next.y - prev.y);
                        if c_p <= 1e-8 * l_pp * l_pn && d_p > 0.0 {
                            continue;
                        }
                    }
                    if next_next.id != prev.id && l_nn * l_pn > 1e-20 {
                        let c_n = cross_2d(prev.pt(), next.pt(), next_next.pt()).abs();
                        let d_n = (prev.x - next.x) * (next_next.x - next.x)
                            + (prev.y - next.y) * (next_next.y - next.y);
                        if c_n <= 1e-8 * l_nn * l_pn && d_n > 0.0 {
                            continue;
                        }
                    }
                }

                if pass == 0
                    && (!point_in_vertex_cone_2d(
                        prev_prev.pt(),
                        prev.pt(),
                        curr.pt(),
                        next.pt(),
                    ) || !point_in_vertex_cone_2d(
                        curr.pt(),
                        next.pt(),
                        next_next.pt(),
                        prev.pt(),
                    ))
                {
                    continue;
                }

                let mut any_inside = false;
                for j in 0..m {
                    if j == (i + m - 1) % m || j == i || j == (i + 1) % m {
                        continue;
                    }
                    let pt = ring[idxs[j]];
                    if pt.id == prev.id
                        || pt.id == curr.id
                        || pt.id == next.id
                        || (pt.x - prev.x).hypot(pt.y - prev.y) <= tol
                        || (pt.x - curr.x).hypot(pt.y - curr.y) <= tol
                        || (pt.x - next.x).hypot(pt.y - next.y) <= tol
                    {
                        continue;
                    }
                    let pj = ring[idxs[(j + m - 1) % m]].pt();
                    let nj = ring[idxs[(j + 1) % m]].pt();
                    if cross_2d(pj, pt.pt(), nj) > tol {
                        continue;
                    }
                    if point_in_triangle_2d(pt.pt(), prev.pt(), curr.pt(), next.pt()) {
                        any_inside = true;
                        break;
                    }
                }

                if !any_inside {
                    if best_ear.map_or(true, |(_, best_s)| sin_theta > best_s) {
                        best_ear = Some((i, sin_theta));
                    }
                }
            }
            if let Some((best_i, _)) = best_ear {
                let prev = ring[idxs[(best_i + m - 1) % m]];
                let curr = ring[idxs[best_i]];
                let next = ring[idxs[(best_i + 1) % m]];
                if prev.id != curr.id && curr.id != next.id && next.id != prev.id {
                    tris.push([prev.id, curr.id, next.id]);
                }
                idxs.remove(best_i);
                break;
            }
        }

        if best_ear.is_none() {
            // Fallback: pick the vertex that contains the fewest reflex vertices and has the best corner angle.
            // Always emit `[prev.id, curr.id, next.id]` so no boundary half-edge is ever dropped!
            let mut best_i = 0usize;
            let mut best_inside_count = usize::MAX;
            let mut best_sin = f64::NEG_INFINITY;
            for i in 0..m {
                let prev = ring[idxs[(i + m - 1) % m]];
                let curr = ring[idxs[i]];
                let next = ring[idxs[(i + 1) % m]];
                let c = cross_2d(prev.pt(), curr.pt(), next.pt());
                if c <= -1e-9 {
                    continue;
                }
                let lu = (curr.x - prev.x).hypot(curr.y - prev.y);
                let lv = (next.x - curr.x).hypot(next.y - curr.y);
                let sin_t = if lu * lv > 1e-20 { c / (lu * lv) } else { 0.0 };
                let mut inside_count = 0usize;
                for j in 0..m {
                    if j == (i + m - 1) % m || j == i || j == (i + 1) % m {
                        continue;
                    }
                    let pt = ring[idxs[j]];
                    if point_in_triangle_2d(pt.pt(), prev.pt(), curr.pt(), next.pt()) {
                        inside_count += 1;
                    }
                }
                if inside_count < best_inside_count
                    || (inside_count == best_inside_count && sin_t > best_sin)
                {
                    best_inside_count = inside_count;
                    best_sin = sin_t;
                    best_i = i;
                }
            }
            let prev = ring[idxs[(best_i + m - 1) % m]];
            let curr = ring[idxs[best_i]];
            let next = ring[idxs[(best_i + 1) % m]];
            if prev.id != curr.id && curr.id != next.id && next.id != prev.id {
                tris.push([prev.id, curr.id, next.id]);
            }
            idxs.remove(best_i);
        }
    }

    if idxs.len() == 3 {
        let a = ring[idxs[0]];
        let b = ring[idxs[1]];
        let c = ring[idxs[2]];
        if a.id != b.id && b.id != c.id && c.id != a.id {
            tris.push([a.id, b.id, c.id]);
        }
    }

    tris
}

#[inline]
fn quad_choose_diagonal_02(p0: Point3D, p1: Point3D, p2: Point3D, p3: Point3D, normal: Point3D) -> bool {
    let c0 = p1.sub(p0).cross(p2.sub(p0)).dot(normal);
    let c2 = p2.sub(p0).cross(p3.sub(p0)).dot(normal);
    let c1 = p2.sub(p1).cross(p3.sub(p1)).dot(normal);
    let c3 = p3.sub(p1).cross(p0.sub(p1)).dot(normal);
    c0.min(c2) >= c1.min(c3)
}

/// Triangulate a 3D face specified by vertex IDs `(outer_vids, holes_vids)` and 3D coordinates `(outer_3d, holes_3d)`
/// into vertex-ID triangles `[vid_a, vid_b, vid_c]` that strictly preserve all boundary half-edges.
fn triangulate_face_vids_3d(
    outer_vids: &[u32],
    holes_vids: &[Vec<u32>],
    outer_3d: &[Point3D],
    holes_3d: &[Vec<Point3D>],
    normal: Point3D,
) -> Vec<[u32; 3]> {
    if outer_vids.len() < 3 {
        return Vec::new();
    }
    if outer_vids.len() == 3 && holes_vids.is_empty() {
        return vec![[outer_vids[0], outer_vids[1], outer_vids[2]]];
    }
    if outer_vids.len() == 4
        && holes_vids.is_empty()
        && outer_vids[0] != outer_vids[1]
        && outer_vids[1] != outer_vids[2]
        && outer_vids[2] != outer_vids[3]
        && outer_vids[3] != outer_vids[0]
        && outer_vids[0] != outer_vids[2]
        && outer_vids[1] != outer_vids[3]
    {
        if quad_choose_diagonal_02(outer_3d[0], outer_3d[1], outer_3d[2], outer_3d[3], normal) {
            return vec![
                [outer_vids[0], outer_vids[1], outer_vids[2]],
                [outer_vids[0], outer_vids[2], outer_vids[3]],
            ];
        } else {
            return vec![
                [outer_vids[0], outer_vids[1], outer_vids[3]],
                [outer_vids[1], outer_vids[2], outer_vids[3]],
            ];
        }
    }
    let origin = outer_3d[0];
    let (u_axis, v_axis) = build_plane_frame(normal);
    let project = |p: Point3D, id: u32| -> IdxPoint2D {
        let rel = p.sub(origin);
        IdxPoint2D {
            x: rel.dot(u_axis),
            y: rel.dot(v_axis),
            id,
        }
    };

    let outer_2d: Vec<IdxPoint2D> = outer_3d
        .iter()
        .zip(outer_vids.iter())
        .map(|(&p, &vid)| project(p, vid))
        .collect();
    let holes_2d: Vec<Vec<IdxPoint2D>> = holes_3d
        .iter()
        .zip(holes_vids.iter())
        .map(|(h_pts, h_vids)| {
            h_pts
                .iter()
                .zip(h_vids.iter())
                .map(|(&p, &vid)| project(p, vid))
                .collect()
        })
        .collect();

    triangulate_indexed_polygon_2d(&outer_2d, &holes_2d)
}

/// Triangulate a 3D planar face (`Face3D`) into oriented 3D triangles `[a, b, c]` matching `face.normal`.
pub fn triangulate_face_3d(face: &Face3D) -> Vec<[Point3D; 3]> {
    if face.outer.len() < 3 {
        return Vec::new();
    }
    if face.outer.len() == 3 && face.holes.is_empty() {
        return vec![[face.outer[0], face.outer[1], face.outer[2]]];
    }
    if face.outer.len() == 4 && face.holes.is_empty() {
        let p0 = face.outer[0];
        let p1 = face.outer[1];
        let p2 = face.outer[2];
        let p3 = face.outer[3];
        if quad_choose_diagonal_02(p0, p1, p2, p3, face.normal) {
            return vec![[p0, p1, p2], [p0, p2, p3]];
        } else {
            return vec![[p0, p1, p3], [p1, p2, p3]];
        }
    }

    let mut orig_3d: Vec<Point3D> = Vec::with_capacity(face.outer.len() + face.holes.len() * 4);
    let mut find_or_push_pt = |p: Point3D| -> u32 {
        for (i, &q) in orig_3d.iter().enumerate() {
            if p.sub(q).norm_sq() <= 1e-22 {
                return i as u32;
            }
        }
        let id = orig_3d.len() as u32;
        orig_3d.push(p);
        id
    };

    let outer_vids: Vec<u32> = face.outer.iter().map(|&p| find_or_push_pt(p)).collect();
    let holes_vids: Vec<Vec<u32>> = face
        .holes
        .iter()
        .map(|h| h.iter().map(|&p| find_or_push_pt(p)).collect())
        .collect();

    triangulate_face_vids_3d(&outer_vids, &holes_vids, &face.outer, &face.holes, face.normal)
        .into_iter()
        .map(|[a, b, c]| [orig_3d[a as usize], orig_3d[b as usize], orig_3d[c as usize]])
        .collect()
}

/// Exact Van Oosterom–Strackee solid angle `Ω_T(p) / (4π)` of oriented triangle `[a, b, c]` at point `p`.
#[inline]
fn triangle_solid_angle_turns(p: Point3D, a: Point3D, b: Point3D, c: Point3D) -> f64 {
    let ra = a.sub(p);
    let rb = b.sub(p);
    let rc = c.sub(p);
    let la = ra.norm();
    let lb = rb.norm();
    let lc = rc.norm();
    if la <= 1e-14 || lb <= 1e-14 || lc <= 1e-14 {
        return 0.0;
    }
    let num = ra.dot(rb.cross(rc));
    let den = la * lb * lc + ra.dot(rb) * lc + rb.dot(rc) * la + rc.dot(ra) * lb;
    let half_omega = num.atan2(den);
    half_omega / (2.0 * core::f64::consts::PI)
}

/// Evaluate the exact solid angle (in turns `Ω / 4π`) subtended by a planar face `nf` on plane `pl`
/// from 3D query point `p`. Uses direct spherical triangle evaluation for triangles/quads (1 `atan2`
/// for triangles, 2 for quads) and orthogonal projection `p_perp` as the fan apex for general polygons
/// or faces with holes.
fn face_solid_angle_turns(p: Point3D, nf: &NormalizedInputFace, pl: &CanonicalPlane) -> f64 {
    let h = pl.normal.dot(p) - pl.d;
    // If `p` lies on `pl` (outside `nf`), the solid angle subtended by `nf` is identically 0.
    if h.abs() <= 1e-12 {
        return 0.0;
    }
    let mut half_omega_sum = 0.0;
    if nf.loops_3d.len() == 1 && nf.loops_3d[0].len() <= 4 {
        let ring = &nf.loops_3d[0];
        let n = ring.len();
        if n >= 3 {
            let ra = ring[0].sub(p);
            let la = ra.norm();
            let mut rb = ring[1].sub(p);
            let mut lb = rb.norm();
            for i in 1..(n - 1) {
                let rc = ring[i + 1].sub(p);
                let lc = rc.norm();
                if la > 1e-14 && lb > 1e-14 && lc > 1e-14 {
                    let num = ra.dot(rb.cross(rc));
                    let den = la * lb * lc + ra.dot(rb) * lc + rb.dot(rc) * la + rc.dot(ra) * lb;
                    half_omega_sum += num.atan2(den);
                }
                rb = rc;
                lb = lc;
            }
        }
    } else {
        let p_perp = p.sub(pl.normal.scale(h));
        let ra = p_perp.sub(p);
        let la = h.abs();
        for ring in &nf.loops_3d {
            let n = ring.len();
            if n < 3 {
                continue;
            }
            let mut rb = ring[0].sub(p);
            let mut lb = rb.norm();
            for i in 0..n {
                let rc = ring[(i + 1) % n].sub(p);
                let lc = rc.norm();
                if lb > 1e-14 && lc > 1e-14 {
                    let num = ra.dot(rb.cross(rc));
                    let den = la * lb * lc + ra.dot(rb) * lc + rb.dot(rc) * la + rc.dot(ra) * lb;
                    half_omega_sum += num.atan2(den);
                }
                rb = rc;
                lb = lc;
            }
        }
    }
    let turns = half_omega_sum * (1.0 / (2.0 * core::f64::consts::PI));
    if nf.orient_sign >= 0 {
        turns
    } else {
        -turns
    }
}

/// Evaluate the 3D generalized winding number `w(p)` of a `Solid3D` at query point `p`.
pub fn solid_winding_number(p: Point3D, solid: &Solid3D) -> f64 {
    let mut turns = 0.0;
    for face in &solid.faces {
        if face.outer.len() < 3 {
            continue;
        }
        let d = face.normal.dot(face.outer[0]);
        let h = face.normal.dot(p) - d;
        if h.abs() <= 1e-12 {
            continue;
        }
        let p_perp = p.sub(face.normal.scale(h));
        let n = face.outer.len();
        for i in 0..n {
            turns += triangle_solid_angle_turns(p, p_perp, face.outer[i], face.outer[(i + 1) % n]);
        }
        for hole in &face.holes {
            let hn = hole.len();
            for i in 0..hn {
                turns += triangle_solid_angle_turns(p, p_perp, hole[i], hole[(i + 1) % hn]);
            }
        }
    }
    turns
}

#[inline]
fn aabb_overlap_3d(min_a: Point3D, max_a: Point3D, min_b: Point3D, max_b: Point3D, tol: f64) -> bool {
    !(max_a.x + tol < min_b.x
        || min_a.x - tol > max_b.x
        || max_a.y + tol < min_b.y
        || min_a.y - tol > max_b.y
        || max_a.z + tol < min_b.z
        || min_a.z - tol > max_b.z)
}

#[inline]
fn plane_straddles_aabb(n: Point3D, d: f64, min_p: Point3D, max_p: Point3D, tol: f64) -> bool {
    let px_min = if n.x >= 0.0 { min_p.x } else { max_p.x };
    let py_min = if n.y >= 0.0 { min_p.y } else { max_p.y };
    let pz_min = if n.z >= 0.0 { min_p.z } else { max_p.z };
    let d_min = n.x * px_min + n.y * py_min + n.z * pz_min - d;
    if d_min > tol {
        return false;
    }
    let px_max = if n.x >= 0.0 { max_p.x } else { min_p.x };
    let py_max = if n.y >= 0.0 { max_p.y } else { min_p.y };
    let pz_max = if n.z >= 0.0 { max_p.z } else { min_p.z };
    let d_max = n.x * px_max + n.y * py_max + n.z * pz_max - d;
    d_max >= -tol
}

/// Compute the 2D signed winding number of `loops` around `pt`.
fn loops_winding_2d(pt: Point2D, loops: &[Vec<Point2D>]) -> i32 {
    let mut w = 0i32;
    for loop_pts in loops {
        let n = loop_pts.len();
        for i in 0..n {
            let a = loop_pts[i];
            let b = loop_pts[(i + 1) % n];
            if a.y <= pt.y {
                if b.y > pt.y && cross_2d(a, b, pt) > 0.0 {
                    w += 1;
                }
            } else if b.y <= pt.y && cross_2d(a, b, pt) < 0.0 {
                w -= 1;
            }
        }
    }
    w
}

// ============================================================================
// Standalone 3D Stratified Tolerance Complex (0, 1, 2, 3-Strata)
// ============================================================================

#[derive(Clone, Debug)]
struct Vert3D {
    pos: Point3D,
    rep: Point3D,
    radius: f64,
    weight: u32,
    parent: u32,
    planes: Vec<usize>,
    dirty_proj: bool,
    alive: bool,
}

#[derive(Clone, Debug)]
struct Edge3D {
    v0: u32,
    v1: u32,
    planes: Vec<usize>,
    /// Directed 2D boundary winding contributions `(nf_idx, delta)` along `v0 -> v1` (`v0 < v1`).
    face_windings: Vec<(usize, i32)>,
    alive: bool,
}

#[inline]
fn merge_plane_tags(dst: &mut Vec<usize>, src: &[usize]) -> bool {
    let mut changed = false;
    for &p in src {
        if !dst.contains(&p) {
            dst.push(p);
            changed = true;
        }
    }
    changed
}

#[inline]
fn merge_face_windings(dst: &mut Vec<(usize, i32)>, src: &[(usize, i32)], sign: i32) {
    for &(nf_idx, d) in src {
        let sd = d * sign;
        if sd == 0 {
            continue;
        }
        if let Some(pos) = dst.iter().position(|&(id, _)| id == nf_idx) {
            dst[pos].1 += sd;
            if dst[pos].1 == 0 {
                dst.swap_remove(pos);
            }
        } else {
            dst.push((nf_idx, sd));
        }
    }
}

#[inline]
fn resolve_vert_3d(verts: &mut [Vert3D], mut v: u32) -> u32 {
    while verts[v as usize].parent != v {
        let p = verts[v as usize].parent;
        verts[v as usize].parent = verts[p as usize].parent;
        v = p;
    }
    v
}

fn find_or_create_vert_3d(
    verts: &mut Vec<Vert3D>,
    p: Point3D,
    r0: f64,
    r_init: f64,
    plane_tags: &[usize],
) -> u32 {
    let mut best_v = None;
    let mut best_d_sq = f64::INFINITY;
    for i in 0..verts.len() {
        if !verts[i].alive {
            continue;
        }
        let d_sq = verts[i].pos.sub(p).norm_sq();
        let rad = verts[i].radius.max(r0);
        if d_sq <= rad * rad && d_sq < best_d_sq {
            best_d_sq = d_sq;
            best_v = Some(i as u32);
        }
    }
    if let Some(vid) = best_v {
        let root = resolve_vert_3d(verts, vid);
        if merge_plane_tags(&mut verts[root as usize].planes, plane_tags) {
            verts[root as usize].dirty_proj = true;
        }
        return root;
    }
    let id = verts.len() as u32;
    let mut planes = Vec::with_capacity(4);
    merge_plane_tags(&mut planes, plane_tags);
    verts.push(Vert3D {
        pos: p,
        rep: p,
        radius: r_init,
        weight: 1,
        parent: id,
        planes,
        dirty_proj: true,
        alive: true,
    });
    id
}

fn find_or_create_vert_3d_bucketed(
    verts: &mut Vec<Vert3D>,
    vert_xy_buckets: &mut [Vec<u32>; 256],
    vert_adj: &mut Vec<Vec<usize>>,
    p: Point3D,
    r0: f64,
    r_init: f64,
    plane_tags: &[usize],
    x_low: f64,
    y_low: f64,
    x_scale: f64,
    y_scale: f64,
) -> u32 {
    let bx0 = (((p.x - r0 - x_low) * x_scale).floor() as isize).clamp(0, 15) as usize;
    let bx1 = (((p.x + r0 - x_low) * x_scale).floor() as isize).clamp(0, 15) as usize;
    let by0 = (((p.y - r0 - y_low) * y_scale).floor() as isize).clamp(0, 15) as usize;
    let by1 = (((p.y + r0 - y_low) * y_scale).floor() as isize).clamp(0, 15) as usize;
    let mut best_v = None;
    let mut best_d_sq = f64::INFINITY;
    for bx in bx0..=bx1 {
        let row = bx << 4;
        for by in by0..=by1 {
            for &cand in &vert_xy_buckets[row | by] {
                let i = cand as usize;
                if !verts[i].alive {
                    continue;
                }
                let d_sq = verts[i].pos.sub(p).norm_sq();
                let rad = verts[i].radius.max(r0);
                if d_sq <= rad * rad && (d_sq < best_d_sq || (d_sq == best_d_sq && Some(cand) < best_v)) {
                    best_d_sq = d_sq;
                    best_v = Some(cand);
                }
            }
        }
    }
    if let Some(vid) = best_v {
        let root = resolve_vert_3d(verts, vid);
        if merge_plane_tags(&mut verts[root as usize].planes, plane_tags) {
            verts[root as usize].dirty_proj = true;
        }
        return root;
    }
    let id = verts.len() as u32;
    let mut planes = Vec::with_capacity(4);
    merge_plane_tags(&mut planes, plane_tags);
    verts.push(Vert3D {
        pos: p,
        rep: p,
        radius: r_init,
        weight: 1,
        parent: id,
        planes,
        dirty_proj: true,
        alive: true,
    });
    let bx = (((p.x - x_low) * x_scale).floor() as isize).clamp(0, 15) as usize;
    let by = (((p.y - y_low) * y_scale).floor() as isize).clamp(0, 15) as usize;
    vert_xy_buckets[(bx << 4) | by].push(id);
    vert_adj.push(Vec::new());
    id
}

fn inject_edge_3d_adj(
    edges: &mut Vec<Edge3D>,
    vert_adj: &mut [Vec<usize>],
    v0: u32,
    v1: u32,
    plane_tags: &[usize],
    face_windings: &[(usize, i32)],
) {
    if v0 == v1 {
        return;
    }
    let (a, b, sign) = if v0 < v1 {
        (v0, v1, 1)
    } else {
        (v1, v0, -1)
    };
    for &e_id in &vert_adj[a as usize] {
        let e = &mut edges[e_id];
        if e.alive && e.v0 == a && e.v1 == b {
            merge_plane_tags(&mut e.planes, plane_tags);
            merge_face_windings(&mut e.face_windings, face_windings, sign);
            return;
        }
    }
    let mut planes = Vec::with_capacity(4);
    merge_plane_tags(&mut planes, plane_tags);
    let mut fw = Vec::with_capacity(face_windings.len());
    merge_face_windings(&mut fw, face_windings, sign);
    let e_id = edges.len();
    edges.push(Edge3D {
        v0: a,
        v1: b,
        planes,
        face_windings: fw,
        alive: true,
    });
    vert_adj[a as usize].push(e_id);
}

/// Check whether a 2D point `pt` lies inside `loops_2d` under `winding_rule`, or within `tol`
/// of any boundary segment of `loops_2d`.
fn point_in_or_near_face_2d(
    pt: Point2D,
    loops_2d: &[Vec<Point2D>],
    winding_rule: WindingRule,
    tol: f64,
) -> bool {
    let w = loops_winding_2d(pt, loops_2d);
    if w != 0 && winding_rule.apply(w as f64) {
        return true;
    }
    let tol_sq = tol * tol;
    for ring in loops_2d {
        let n = ring.len();
        for i in 0..n {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let l_sq = dx * dx + dy * dy;
            let t = if l_sq > 1e-24 {
                (((pt.x - a.x) * dx + (pt.y - a.y) * dy) / l_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let px = a.x + t * dx;
            let py = a.y + t * dy;
            let d_sq = (pt.x - px) * (pt.x - px) + (pt.y - py) * (pt.y - py);
            if d_sq <= tol_sq {
                return true;
            }
        }
    }
    false
}

/// Extract exact 1D intersection chords where planar face `other_nf` (on plane `other_pl`)
/// intersects faces `target_plane_faces` on carrier plane `target_pl`, and inject them directly
/// into the global 3D `(verts, edges)` complex tagged with `[other_nf.plane_id, target_pl_id]`.
fn inject_face_plane_chords_3d(
    other_nf: &NormalizedInputFace,
    other_pl: &CanonicalPlane,
    target_pl: &CanonicalPlane,
    target_pl_id: usize,
    target_plane_faces: &[usize],
    norm_faces: &[NormalizedInputFace],
    r0: f64,
    r_init: f64,
    winding_rule: WindingRule,
    verts: &mut Vec<Vert3D>,
    edges: &mut Vec<Edge3D>,
    vert_xy_buckets: &mut [Vec<u32>; 256],
    vert_adj: &mut Vec<Vec<usize>>,
    x_low: f64,
    y_low: f64,
    x_scale: f64,
    y_scale: f64,
) -> bool {
    let s0 = target_pl.normal.dot(other_pl.origin) - target_pl.d;
    let alpha = target_pl.normal.dot(other_pl.u_axis);
    let beta = target_pl.normal.dot(other_pl.v_axis);
    let grad_norm = alpha.hypot(beta);
    if grad_norm <= 1e-11 {
        return false;
    }
    let inv_grad = 1.0 / grad_norm;
    let h0 = s0 * inv_grad;
    let nx = alpha * inv_grad;
    let ny = beta * inv_grad;
    let tx = -ny;
    let ty = nx;
    let snap_tol = 1e-11;
    let both_planes = [other_nf.plane_id, target_pl_id];
    let near_tol = r0.max(1e-7);

    let mut line_pts: Vec<(f64, Point2D)> = Vec::with_capacity(16);

    for loop_pts in &other_nf.loops_2d {
        let n = loop_pts.len();
        for i in 0..n {
            let a = loop_pts[i];
            let b = loop_pts[(i + 1) % n];
            let mut ha = h0 + nx * a.x + ny * a.y;
            let mut hb = h0 + nx * b.x + ny * b.y;
            if ha.abs() <= snap_tol {
                ha = 0.0;
            }
            if hb.abs() <= snap_tol {
                hb = 0.0;
            }

            if ha == 0.0 && hb == 0.0 {
                let ta = tx * a.x + ty * a.y;
                let tb = tx * b.x + ty * b.y;
                line_pts.push((ta, a));
                line_pts.push((tb, b));
            } else if ha == 0.0 {
                line_pts.push((tx * a.x + ty * a.y, a));
            } else if hb == 0.0 {
                line_pts.push((tx * b.x + ty * b.y, b));
            } else if (ha < 0.0 && hb > 0.0) || (ha > 0.0 && hb < 0.0) {
                let s = ha / (ha - hb);
                let p = Point2D {
                    x: a.x + s * (b.x - a.x),
                    y: a.y + s * (b.y - a.y),
                };
                line_pts.push((tx * p.x + ty * p.y, p));
            }
        }
    }

    if line_pts.len() < 2 {
        return false;
    }

    let mut t_other_min = f64::INFINITY;
    let mut t_other_max = f64::NEG_INFINITY;
    for &(t, _) in &line_pts {
        t_other_min = t_other_min.min(t);
        t_other_max = t_other_max.max(t);
    }
    if t_other_max - t_other_min <= snap_tol {
        return false;
    }

    // Also subdivide the 1D intersection line at crossing points of `target_plane_faces` on `target_pl`
    let mut any_tf_near = false;
    let mut t_tf_min = f64::INFINITY;
    let mut t_tf_max = f64::NEG_INFINITY;
    let t_clip_min = t_other_min - near_tol;
    let t_clip_max = t_other_max + near_tol;
    for &tf_idx in target_plane_faces {
        let tf = &norm_faces[tf_idx];
        if !aabb_overlap_3d(other_nf.min_pt, other_nf.max_pt, tf.min_pt, tf.max_pt, r0) {
            continue;
        }
        any_tf_near = true;
        for loop_3d in &tf.loops_3d {
            let n = loop_3d.len();
            for i in 0..n {
                let a3 = loop_3d[i];
                let b3 = loop_3d[(i + 1) % n];
                let mut ha = other_pl.normal.dot(a3) - other_pl.d;
                let mut hb = other_pl.normal.dot(b3) - other_pl.d;
                if ha.abs() <= snap_tol {
                    ha = 0.0;
                }
                if hb.abs() <= snap_tol {
                    hb = 0.0;
                }
                if ha == 0.0 && hb == 0.0 {
                    let pa = other_pl.project_2d(a3);
                    let pb = other_pl.project_2d(b3);
                    let ta = tx * pa.x + ty * pa.y;
                    let tb = tx * pb.x + ty * pb.y;
                    t_tf_min = t_tf_min.min(ta).min(tb);
                    t_tf_max = t_tf_max.max(ta).max(tb);
                    if ta >= t_clip_min && ta <= t_clip_max {
                        line_pts.push((ta, pa));
                    }
                    if tb >= t_clip_min && tb <= t_clip_max {
                        line_pts.push((tb, pb));
                    }
                } else if ha == 0.0 {
                    let pa = other_pl.project_2d(a3);
                    let ta = tx * pa.x + ty * pa.y;
                    t_tf_min = t_tf_min.min(ta);
                    t_tf_max = t_tf_max.max(ta);
                    if ta >= t_clip_min && ta <= t_clip_max {
                        line_pts.push((ta, pa));
                    }
                } else if hb == 0.0 {
                    let pb = other_pl.project_2d(b3);
                    let tb = tx * pb.x + ty * pb.y;
                    t_tf_min = t_tf_min.min(tb);
                    t_tf_max = t_tf_max.max(tb);
                    if tb >= t_clip_min && tb <= t_clip_max {
                        line_pts.push((tb, pb));
                    }
                } else if (ha < 0.0 && hb > 0.0) || (ha > 0.0 && hb < 0.0) {
                    let s = ha / (ha - hb);
                    let p3 = a3.add(b3.sub(a3).scale(s));
                    let p2 = other_pl.project_2d(p3);
                    let tp = tx * p2.x + ty * p2.y;
                    t_tf_min = t_tf_min.min(tp);
                    t_tf_max = t_tf_max.max(tp);
                    if tp >= t_clip_min && tp <= t_clip_max {
                        line_pts.push((tp, p2));
                    }
                }
            }
        }
    }
    if !any_tf_near {
        return false;
    }
    if r0 <= 1e-4 && t_tf_min <= t_tf_max {
        let overlap_min = t_other_min.max(t_tf_min - near_tol);
        let overlap_max = t_other_max.min(t_tf_max + near_tol);
        if overlap_max - overlap_min <= snap_tol {
            return false;
        }
    }

    line_pts.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(core::cmp::Ordering::Equal));

    let mut unique_pts: Vec<(f64, Point2D)> = Vec::with_capacity(line_pts.len());
    for item in line_pts {
        if let Some(last) = unique_pts.last_mut() {
            if (item.0 - last.0).abs() <= snap_tol {
                last.1 = Point2D {
                    x: 0.5 * (last.1.x + item.1.x),
                    y: 0.5 * (last.1.y + item.1.y),
                };
                last.0 = tx * last.1.x + ty * last.1.y;
                continue;
            }
        }
        unique_pts.push(item);
    }

    if unique_pts.len() < 2 {
        return false;
    }

    let mut active_start: Option<Point2D> = None;
    let mut active_end: Point2D = unique_pts[0].1;
    let mut injected_any = false;

    for k in 0..(unique_pts.len() - 1) {
        let p0 = unique_pts[k].1;
        let p1 = unique_pts[k + 1].1;
        let mid = Point2D {
            x: 0.5 * (p0.x + p1.x),
            y: 0.5 * (p0.y + p1.y),
        };
        let in_other = point_in_or_near_face_2d(mid, &other_nf.loops_2d, winding_rule, 1e-7);
        let mid_3d = other_pl.unproject_3d(mid.x, mid.y);
        let mid_target_2d = target_pl.project_2d(mid_3d);
        let in_target = target_plane_faces.iter().any(|&tf_idx| {
            let tf = &norm_faces[tf_idx];
            aabb_overlap_3d(other_nf.min_pt, other_nf.max_pt, tf.min_pt, tf.max_pt, r0)
                && point_in_or_near_face_2d(mid_target_2d, &tf.loops_2d, winding_rule, near_tol)
        });
        let inside = in_other && in_target;

        if inside {
            if active_start.is_none() {
                active_start = Some(p0);
            }
            active_end = p1;
        } else if let Some(start_pt) = active_start.take() {
            let p0_3d = other_pl.unproject_3d(start_pt.x, start_pt.y);
            let p1_3d = other_pl.unproject_3d(active_end.x, active_end.y);
            let v0 = find_or_create_vert_3d_bucketed(
                verts,
                vert_xy_buckets,
                vert_adj,
                p0_3d,
                r0,
                r_init,
                &both_planes,
                x_low,
                y_low,
                x_scale,
                y_scale,
            );
            let v1 = find_or_create_vert_3d_bucketed(
                verts,
                vert_xy_buckets,
                vert_adj,
                p1_3d,
                r0,
                r_init,
                &both_planes,
                x_low,
                y_low,
                x_scale,
                y_scale,
            );
            inject_edge_3d_adj(edges, vert_adj, v0, v1, &both_planes, &[]);
            injected_any = true;
        }
    }

    if let Some(start_pt) = active_start {
        let p0_3d = other_pl.unproject_3d(start_pt.x, start_pt.y);
        let p1_3d = other_pl.unproject_3d(active_end.x, active_end.y);
        let v0 = find_or_create_vert_3d_bucketed(
            verts,
            vert_xy_buckets,
            vert_adj,
            p0_3d,
            r0,
            r_init,
            &both_planes,
            x_low,
            y_low,
            x_scale,
            y_scale,
        );
        let v1 = find_or_create_vert_3d_bucketed(
            verts,
            vert_xy_buckets,
            vert_adj,
            p1_3d,
            r0,
            r_init,
            &both_planes,
            x_low,
            y_low,
            x_scale,
            y_scale,
        );
        inject_edge_3d_adj(edges, vert_adj, v0, v1, &both_planes, &[]);
        injected_any = true;
    }
    injected_any
}

fn merge_two_verts_3d(verts: &mut [Vert3D], u: u32, v: u32, r_max: f64) -> bool {
    let ru = resolve_vert_3d(verts, u) as usize;
    let rv = resolve_vert_3d(verts, v) as usize;
    if ru == rv || !verts[ru].alive || !verts[rv].alive {
        return false;
    }
    let pu = verts[ru].pos;
    let pv = verts[rv].pos;
    let d = pu.sub(pv).norm();
    let rad_u = verts[ru].radius;
    let rad_v = verts[rv].radius;
    let wu = verts[ru].weight as f64;
    let wv = verts[rv].weight as f64;
    let w_tot = wu + wv;
    let new_pos = pu.scale(wu / w_tot).add(pv.scale(wv / w_tot));
    let new_rad = r_max.min(rad_u.max(rad_v).max(0.5 * (d + rad_u + rad_v)));

    let (keep, kill) = if ru < rv { (ru, rv) } else { (rv, ru) };
    let kill_planes = core::mem::take(&mut verts[kill].planes);
    verts[keep].pos = new_pos;
    verts[keep].rep = new_pos;
    verts[keep].radius = new_rad;
    verts[keep].weight = verts[ru].weight.saturating_add(verts[rv].weight);
    verts[keep].dirty_proj = true;
    merge_plane_tags(&mut verts[keep].planes, &kill_planes);
    verts[kill].alive = false;
    verts[kill].parent = keep as u32;
    true
}

/// Solve 3×3 symmetric positive-definite linear system `A x = b` via Cramer's rule.
fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<Point3D> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() <= 1e-18 {
        return None;
    }
    let inv_det = 1.0 / det;
    let dx = (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]))
        * inv_det;
    let dy = (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]))
        * inv_det;
    let dz = (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        * inv_det;
    Some(Point3D::new(dx, dy, dz))
}

/// Closest approach between two 3D segments `(p0, p1)` and `(q0, q1)`.
/// Returns `(s, t, dist_sq, mid_pt)` with `s, t ∈ [0, 1]`.
fn closest_approach_segments_3d(
    p0: Point3D,
    p1: Point3D,
    q0: Point3D,
    q1: Point3D,
) -> (f64, f64, f64, Point3D) {
    let u = p1.sub(p0);
    let v = q1.sub(q0);
    let w = p0.sub(q0);
    let a = u.dot(u);
    let b = u.dot(v);
    let c = v.dot(v);
    let d = u.dot(w);
    let e = v.dot(w);
    let denom = a * c - b * b;

    if denom <= 1e-16 * a * c || a <= 1e-22 || c <= 1e-22 {
        return (0.0, 0.0, f64::INFINITY, p0);
    }

    let s = ((b * e - c * d) / denom).clamp(0.0, 1.0);
    let t = ((a * e - b * d) / denom).clamp(0.0, 1.0);
    let pa = p0.add(u.scale(s));
    let qb = q0.add(v.scale(t));
    let dist_sq = pa.sub(qb).norm_sq();
    let mid = pa.add(qb).scale(0.5);
    (s, t, dist_sq, mid)
}

/// Project each consolidated 3D vertex `v` onto the intersection of its incident carrier planes
/// via Levenberg-Marquardt trust-region 3×3 quadric minimization clamped inside `B(v.pos, cap)`.
fn project_vertices_to_carrier_planes_3d(
    verts: &mut [Vert3D],
    planes: &[CanonicalPlane],
    r_max: f64,
) {
    let base_lambda = 1e-9;
    for v in verts.iter_mut() {
        if !v.alive || !v.dirty_proj {
            continue;
        }
        v.dirty_proj = false;
        if v.planes.is_empty() {
            v.rep = v.pos;
            continue;
        }
        let mut ata = [[0.0; 3]; 3];
        let mut rhs = [0.0; 3];
        let mut max_res = 0.0_f64;
        for &p_id in &v.planes {
            let pl = &planes[p_id];
            let n = pl.normal;
            let res = pl.d - n.dot(v.pos);
            max_res = max_res.max(res.abs());
            let n_arr = [n.x, n.y, n.z];
            for r in 0..3 {
                rhs[r] += res * n_arr[r];
                for c in 0..3 {
                    ata[r][c] += n_arr[r] * n_arr[c];
                }
            }
        }
        if max_res <= 1e-13 {
            v.rep = v.pos;
            continue;
        }
        let cap = v.radius.max(0.75 * r_max);
        let solve_with_lambda = |lam: f64| -> Option<Point3D> {
            let mut m = ata;
            m[0][0] += lam;
            m[1][1] += lam;
            m[2][2] += lam;
            solve_3x3(m, rhs)
        };

        if let Some(delta0) = solve_with_lambda(base_lambda) {
            if delta0.norm() <= cap {
                v.rep = v.pos.add(delta0);
            } else {
                // Glancing planes whose exact intersection line lies outside the trust ball `cap`:
                // increase Tikhonov damping `lambda` so ill-conditioned tangential slide is suppressed
                // while preserving orthogonal projection onto the carrier planes.
                let k_scale = (v.planes.len() as f64).max(1.0);
                let mut lo = 1e-6 * k_scale;
                let mut hi = 100.0 * k_scale;
                let mut best = delta0.scale(cap / delta0.norm());
                for _ in 0..12 {
                    let mid = (lo * hi).sqrt();
                    if let Some(d_mid) = solve_with_lambda(mid) {
                        let len = d_mid.norm();
                        if len <= cap {
                            best = d_mid;
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    } else {
                        lo = mid;
                    }
                }
                v.rep = v.pos.add(best);
            }
        } else {
            v.rep = v.pos;
        }
    }
}

/// Stage 2: Coupled 3D 0-Stratum & 1-Stratum Fixpoint Consolidation.
fn consolidate_3d_complex(
    verts: &mut Vec<Vert3D>,
    edges: &mut Vec<Edge3D>,
    planes: &[CanonicalPlane],
    r0: f64,
    r_init: f64,
    r_max: f64,
    vert_active: &mut Vec<bool>,
    vert_xy_buckets: &mut [Vec<u32>; 256],
    vert_adj: &mut Vec<Vec<usize>>,
    x_low: f64,
    y_low: f64,
    x_scale: f64,
    y_scale: f64,
) {
    let mut edge_splits: Vec<Vec<(f64, u32)>> = Vec::new();
    let mut sorted_v_step1: Vec<(f64, f64, usize)> = Vec::new();
    let mut sorted_e_step3: Vec<(u32, u32, usize)> = Vec::new();
    let mut sorted_v_step4: Vec<(f64, usize)> = Vec::new();
    let mut sorted_e_step5: Vec<(f64, usize)> = Vec::new();

    for iter in 0..32 {
        let mut changed = false;
        vert_active.resize(verts.len(), true);
        if vert_adj.len() < verts.len() {
            vert_adj.resize_with(verts.len(), Vec::new);
        }

        // 0. Project vertex representative coordinates onto incident carrier planes
        project_vertices_to_carrier_planes_3d(verts, planes, r_max);
        let rep_collapse_tol = (1e-3 * r0).max(1e-7);
        let rep_collapse_sq = rep_collapse_tol * rep_collapse_tol;

        let mut next_active = vec![false; verts.len()];
        let mark_neighborhood_active =
            |u: usize, next_active: &mut [bool], vert_adj: &[Vec<usize>], edges: &[Edge3D]| {
                if u < next_active.len() {
                    next_active[u] = true;
                }
                if u < vert_adj.len() {
                    for &eid in &vert_adj[u] {
                        let e = &edges[eid];
                        if e.alive {
                            let va = e.v0 as usize;
                            let vb = e.v1 as usize;
                            if va < next_active.len() {
                                next_active[va] = true;
                            }
                            if vb < next_active.len() {
                                next_active[vb] = true;
                            }
                        }
                    }
                }
            };

        // 1. 0-Stratum Ball Merge (1D X-Sorted Sweep-and-Prune preserving exact `(i, j)` order)
        let nv = verts.len();
        let max_search_r = r_max.max(r0).max(rep_collapse_tol);
        let search_pos_rx = 2.0 * r_max + max_search_r;
        sorted_v_step1.clear();
        for idx in 0..nv {
            if verts[idx].alive && vert_active[idx] {
                sorted_v_step1.push((verts[idx].pos.x, 0.0, idx));
            }
        }
        sorted_v_step1.sort_unstable_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal)
        });

        let mut cand_js: Vec<usize> = Vec::with_capacity(16);
        for i in 0..nv {
            if !verts[i].alive || !vert_active[i] {
                continue;
            }
            let min_px = verts[i].pos.x - search_pos_rx;
            let max_px = verts[i].pos.x + search_pos_rx;
            let start = sorted_v_step1.partition_point(|&(px, _, _)| px < min_px);
            cand_js.clear();
            for &(px, _, j) in &sorted_v_step1[start..] {
                if px > max_px {
                    break;
                }
                if j > i && verts[j].alive {
                    cand_js.push(j);
                }
            }
            cand_js.sort_unstable();

            let pi = verts[i].pos;
            let ri = verts[i].rep;
            let rad_i = verts[i].radius;
            for &j in &cand_js {
                if !verts[j].alive {
                    continue;
                }
                let rj = verts[j].rep;
                let pj = verts[j].pos;
                if (ri.x - rj.x).abs() > max_search_r && (pi.x - pj.x).abs() > max_search_r {
                    continue;
                }
                if (ri.y - rj.y).abs() > max_search_r && (pi.y - pj.y).abs() > max_search_r {
                    continue;
                }
                if (ri.z - rj.z).abs() > max_search_r && (pi.z - pj.z).abs() > max_search_r {
                    continue;
                }
                let max_r = (rad_i + verts[j].radius).min(r_max);
                let pos_close = (pi.x - pj.x).abs() <= max_r
                    && (pi.y - pj.y).abs() <= max_r
                    && (pi.z - pj.z).abs() <= max_r
                    && pi.sub(pj).norm_sq() <= max_r * max_r;
                let rep_coincident = (ri.x - rj.x).abs() <= rep_collapse_tol
                    && (ri.y - rj.y).abs() <= rep_collapse_tol
                    && (ri.z - rj.z).abs() <= rep_collapse_tol
                    && ri.sub(rj).norm_sq() <= rep_collapse_sq;
                let rep_close = (ri.x - rj.x).abs() <= r0
                    && (ri.y - rj.y).abs() <= r0
                    && (ri.z - rj.z).abs() <= r0
                    && ri.sub(rj).norm_sq() <= r0 * r0
                    && pi.sub(pj).norm_sq() <= 4.0 * r_max * r_max;
                if pos_close || rep_coincident || rep_close {
                    if merge_two_verts_3d(verts, i as u32, j as u32, r_max) {
                        changed = true;
                        let root = resolve_vert_3d(verts, i as u32) as usize;
                        mark_neighborhood_active(i, &mut next_active, vert_adj, edges);
                        mark_neighborhood_active(j, &mut next_active, vert_adj, edges);
                        mark_neighborhood_active(root, &mut next_active, vert_adj, edges);
                    }
                }
            }
        }

        // 2. Micro-Edge Contraction & Endpoint Remap
        let ne = edges.len();
        for e_idx in 0..ne {
            if !edges[e_idx].alive {
                continue;
            }
            let u0 = edges[e_idx].v0 as usize;
            let v0 = edges[e_idx].v1 as usize;
            if !vert_active[u0] && !vert_active[v0] {
                continue;
            }
            let u = resolve_vert_3d(verts, edges[e_idx].v0);
            let v = resolve_vert_3d(verts, edges[e_idx].v1);
            if u == v {
                edges[e_idx].alive = false;
                changed = true;
                mark_neighborhood_active(u as usize, &mut next_active, vert_adj, edges);
                continue;
            }
            let (a, b, flipped) = if u < v { (u, v, false) } else { (v, u, true) };
            if edges[e_idx].v0 != a || edges[e_idx].v1 != b {
                edges[e_idx].v0 = a;
                edges[e_idx].v1 = b;
                if !vert_adj[a as usize].contains(&e_idx) {
                    vert_adj[a as usize].push(e_idx);
                }
            }
            if flipped {
                for fw in &mut edges[e_idx].face_windings {
                    fw.1 = -fw.1;
                }
            }

            let pa = verts[a as usize].pos;
            let pb = verts[b as usize].pos;
            let ra = verts[a as usize].rep;
            let rb = verts[b as usize].rep;
            let contract_r = (verts[a as usize].radius + verts[b as usize].radius).min(r_max);
            if pa.sub(pb).norm_sq() <= contract_r * contract_r
                || ra.sub(rb).norm_sq() <= rep_collapse_sq
            {
                merge_two_verts_3d(verts, a, b, r_max);
                let root = resolve_vert_3d(verts, a);
                let e_planes = core::mem::take(&mut edges[e_idx].planes);
                if merge_plane_tags(&mut verts[root as usize].planes, &e_planes) {
                    verts[root as usize].dirty_proj = true;
                }
                edges[e_idx].alive = false;
                changed = true;
                mark_neighborhood_active(a as usize, &mut next_active, vert_adj, edges);
                mark_neighborhood_active(b as usize, &mut next_active, vert_adj, edges);
                mark_neighborhood_active(root as usize, &mut next_active, vert_adj, edges);
                continue;
            }

            // Propagate edge carrier planes to its endpoints without heap allocations
            if merge_plane_tags(&mut verts[a as usize].planes, &edges[e_idx].planes) {
                verts[a as usize].dirty_proj = true;
                mark_neighborhood_active(a as usize, &mut next_active, vert_adj, edges);
            }
            if merge_plane_tags(&mut verts[b as usize].planes, &edges[e_idx].planes) {
                verts[b as usize].dirty_proj = true;
                mark_neighborhood_active(b as usize, &mut next_active, vert_adj, edges);
            }
        }

        // 3. Deduplicate Coincident Undirected Edges `{v0, v1}` in O(E log E)
        sorted_e_step3.clear();
        for i in 0..edges.len() {
            if !edges[i].alive {
                continue;
            }
            let u0 = edges[i].v0 as usize;
            let v0 = edges[i].v1 as usize;
            if !vert_active[u0] && !vert_active[v0] {
                continue;
            }
            let u = resolve_vert_3d(verts, edges[i].v0);
            let v = resolve_vert_3d(verts, edges[i].v1);
            if u == v {
                edges[i].alive = false;
                changed = true;
                mark_neighborhood_active(u as usize, &mut next_active, vert_adj, edges);
                continue;
            }
            let (a, b, flip_i) = if u < v { (u, v, false) } else { (v, u, true) };
            if edges[i].v0 != a {
                if !vert_adj[a as usize].contains(&i) {
                    vert_adj[a as usize].push(i);
                }
            }
            edges[i].v0 = a;
            edges[i].v1 = b;
            if flip_i {
                for fw in &mut edges[i].face_windings {
                    fw.1 = -fw.1;
                }
            }
            sorted_e_step3.push((a, b, i));
        }
        sorted_e_step3.sort_unstable();
        let mut idx_e = 0;
        while idx_e < sorted_e_step3.len() {
            let (a, b, i) = sorted_e_step3[idx_e];
            idx_e += 1;
            while idx_e < sorted_e_step3.len()
                && sorted_e_step3[idx_e].0 == a
                && sorted_e_step3[idx_e].1 == b
            {
                let j = sorted_e_step3[idx_e].2;
                let j_planes = core::mem::take(&mut edges[j].planes);
                let j_fw = core::mem::take(&mut edges[j].face_windings);
                merge_plane_tags(&mut edges[i].planes, &j_planes);
                merge_face_windings(&mut edges[i].face_windings, &j_fw, 1);
                edges[j].alive = false;
                changed = true;
                mark_neighborhood_active(a as usize, &mut next_active, vert_adj, edges);
                mark_neighborhood_active(b as usize, &mut next_active, vert_adj, edges);
                idx_e += 1;
            }
        }

        // Update vertex representative coordinates on carrier planes before checking T-junctions & crossings
        project_vertices_to_carrier_planes_3d(verts, planes, r_max);

        edge_splits.clear();
        edge_splits.resize(edges.len(), Vec::new());

        // 4. Ball-Tube 3D & Co-Carrier 2D T-Junction Splits (1D X-Sorted Sweep-and-Prune)
        let cur_nv = verts.len();
        let cur_ne = edges.len();
        sorted_v_step4.clear();
        for w in 0..cur_nv {
            if verts[w].alive && vert_active[w] {
                sorted_v_step4.push((verts[w].rep.x, w));
            }
        }
        sorted_v_step4.sort_unstable_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal)
        });

        for e_idx in 0..cur_ne {
            if !edges[e_idx].alive {
                continue;
            }
            let u = edges[e_idx].v0 as usize;
            let v = edges[e_idx].v1 as usize;
            if !vert_active[u] && !vert_active[v] {
                continue;
            }
            let pu = verts[u].rep;
            let pv = verts[v].rep;
            let ab = pv.sub(pu);
            let len_sq = ab.norm_sq();
            if len_sq <= 1e-20 {
                continue;
            }
            let min_x = pu.x.min(pv.x) - r_max;
            let max_x = pu.x.max(pv.x) + r_max;
            let min_y = pu.y.min(pv.y) - r_max;
            let max_y = pu.y.max(pv.y) + r_max;
            let min_z = pu.z.min(pv.z) - r_max;
            let max_z = pu.z.max(pv.z) + r_max;

            let start_idx = sorted_v_step4.partition_point(|&(vx, _)| vx < min_x);
            for &(vx, w) in &sorted_v_step4[start_idx..] {
                if vx > max_x {
                    break;
                }
                if w == u || w == v {
                    continue;
                }
                let pw = verts[w].rep;
                if pw.y < min_y || pw.y > max_y || pw.z < min_z || pw.z > max_z {
                    continue;
                }

                let tol = verts[w].radius.max(r0);
                let tol_sq = tol * tol;
                if pw.sub(pu).norm_sq() <= tol_sq || pw.sub(pv).norm_sq() <= tol_sq {
                    continue;
                }

                let t = pw.sub(pu).dot(ab) / len_sq;
                let mut split_t = None;
                if t > 1e-5 && t < 1.0 - 1e-5 {
                    let proj = pu.add(ab.scale(t));
                    let d_sq = pw.sub(proj).norm_sq();
                    if d_sq <= tol_sq {
                        split_t = Some(t);
                    }
                }

                // Also check 2D projection on any shared carrier plane `p_id`
                if split_t.is_none() {
                    for &p_id in &edges[e_idx].planes {
                        if !verts[w].planes.contains(&p_id) {
                            continue;
                        }
                        let pl = &planes[p_id];
                        let a2 = pl.project_2d(pu);
                        let b2 = pl.project_2d(pv);
                        let w2 = pl.project_2d(pw);
                        let da_sq = (w2.x - a2.x) * (w2.x - a2.x) + (w2.y - a2.y) * (w2.y - a2.y);
                        let db_sq = (w2.x - b2.x) * (w2.x - b2.x) + (w2.y - b2.y) * (w2.y - b2.y);
                        if da_sq <= tol_sq || db_sq <= tol_sq {
                            continue;
                        }
                        let dx = b2.x - a2.x;
                        let dy = b2.y - a2.y;
                        let l2 = dx * dx + dy * dy;
                        if l2 <= 1e-20 {
                            continue;
                        }
                        let t2 = ((w2.x - a2.x) * dx + (w2.y - a2.y) * dy) / l2;
                        if t2 > 1e-5 && t2 < 1.0 - 1e-5 {
                            let px = a2.x + t2 * dx;
                            let py = a2.y + t2 * dy;
                            let d2_sq = (w2.x - px) * (w2.x - px) + (w2.y - py) * (w2.y - py);
                            if d2_sq <= tol_sq {
                                split_t = Some(t2);
                                break;
                            }
                        }
                    }
                }

                if let Some(st) = split_t {
                    if merge_plane_tags(&mut verts[w].planes, &edges[e_idx].planes) {
                        verts[w].dirty_proj = true;
                    }
                    edge_splits[e_idx].push((st, w as u32));
                }
            }
        }

        // 5. Tube-Tube 3D & Co-Carrier 2D Segment Intersection (1D X-Sorted Sweep-and-Prune)
        let mut edge_aabb: Vec<[f64; 6]> = Vec::with_capacity(cur_ne);
        sorted_e_step5.clear();
        for e_idx in 0..cur_ne {
            let u = edges[e_idx].v0 as usize;
            let v = edges[e_idx].v1 as usize;
            if !edges[e_idx].alive || (!vert_active[u] && !vert_active[v]) {
                edge_aabb.push([0.0; 6]);
                continue;
            }
            let p0 = verts[u].rep;
            let p1 = verts[v].rep;
            let min_x = p0.x.min(p1.x) - r0;
            edge_aabb.push([
                min_x,
                p0.x.max(p1.x) + r0,
                p0.y.min(p1.y) - r0,
                p0.y.max(p1.y) + r0,
                p0.z.min(p1.z) - r0,
                p0.z.max(p1.z) + r0,
            ]);
            sorted_e_step5.push((min_x, e_idx));
        }
        sorted_e_step5.sort_unstable_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal)
        });

        for idx_a in 0..sorted_e_step5.len() {
            let e1 = sorted_e_step5[idx_a].1;
            let max1_x_bound = edge_aabb[e1][1];

            for idx_b in (idx_a + 1)..sorted_e_step5.len() {
                if sorted_e_step5[idx_b].0 > max1_x_bound {
                    break;
                }
                let e2 = sorted_e_step5[idx_b].1;
                let (i, j) = if e1 < e2 { (e1, e2) } else { (e2, e1) };

                let [min1_x, max1_x, min1_y, max1_y, min1_z, max1_z] = edge_aabb[i];
                let [min2_x, max2_x, min2_y, max2_y, min2_z, max2_z] = edge_aabb[j];
                if max2_x < min1_x
                    || min2_x > max1_x
                    || max2_y < min1_y
                    || min2_y > max1_y
                    || max2_z < min1_z
                    || min2_z > max1_z
                {
                    continue;
                }
                let u1 = edges[i].v0 as usize;
                let v1 = edges[i].v1 as usize;
                let u2 = edges[j].v0 as usize;
                let v2 = edges[j].v1 as usize;
                if u1 == u2 || u1 == v2 || v1 == u2 || v1 == v2 {
                    continue;
                }
                let p0 = verts[u1].rep;
                let p1 = verts[v1].rep;
                let q0 = verts[u2].rep;
                let q1 = verts[v2].rep;

                let mut found_cross: Option<(f64, f64, Point3D)> = None;

                // 5a. Check 3D closest approach
                let (s3, t3, d3_sq, mid3) = closest_approach_segments_3d(p0, p1, q0, q1);
                if s3 > 1e-5 && s3 < 1.0 - 1e-5 && t3 > 1e-5 && t3 < 1.0 - 1e-5 && d3_sq <= r0 * r0
                {
                    found_cross = Some((s3, t3, mid3));
                }

                // 5b. Check 2D intersection on any shared carrier plane
                if found_cross.is_none() {
                    for &p_id in &edges[i].planes {
                        if !edges[j].planes.contains(&p_id) {
                            continue;
                        }
                        let pl = &planes[p_id];
                        let a2 = pl.project_2d(p0);
                        let b2 = pl.project_2d(p1);
                        let c2 = pl.project_2d(q0);
                        let d2 = pl.project_2d(q1);
                        let dx1 = b2.x - a2.x;
                        let dy1 = b2.y - a2.y;
                        let dx2 = d2.x - c2.x;
                        let dy2 = d2.y - c2.y;
                        let denom = dx1 * dy2 - dy1 * dx2;
                        if denom.abs() > 1e-14 {
                            let s2 = ((c2.x - a2.x) * dy2 - (c2.y - a2.y) * dx2) / denom;
                            let t2 = ((c2.x - a2.x) * dy1 - (c2.y - a2.y) * dx1) / denom;
                            if s2 > 1e-5 && s2 < 1.0 - 1e-5 && t2 > 1e-5 && t2 < 1.0 - 1e-5 {
                                let ix = a2.x + s2 * dx1;
                                let iy = a2.y + s2 * dy1;
                                found_cross = Some((s2, t2, pl.unproject_3d(ix, iy)));
                                break;
                            }
                        }
                    }
                }

                if let Some((s, t, mid)) = found_cross {
                    let mut union_planes = edges[i].planes.clone();
                    merge_plane_tags(&mut union_planes, &edges[j].planes);
                    let v_int = find_or_create_vert_3d_bucketed(
                        verts,
                        vert_xy_buckets,
                        vert_adj,
                        mid,
                        r0,
                        r_init,
                        &union_planes,
                        x_low,
                        y_low,
                        x_scale,
                        y_scale,
                    );
                    let r_int = resolve_vert_3d(verts, v_int) as usize;
                    if r_int >= next_active.len() {
                        next_active.resize(r_int + 1, false);
                    }
                    next_active[r_int] = true;
                    if r_int != u1 && r_int != v1 {
                        edge_splits[i].push((s, r_int as u32));
                    }
                    if r_int != u2 && r_int != v2 {
                        edge_splits[j].push((t, r_int as u32));
                    }
                }
            }
        }

        // 6. Apply Recorded Edge Splits in Sorted `t`-Order
        for e_idx in 0..cur_ne {
            if !edges[e_idx].alive || edge_splits[e_idx].is_empty() {
                continue;
            }
            let raw_u = edges[e_idx].v0;
            let raw_v = edges[e_idx].v1;
            let u = resolve_vert_3d(verts, raw_u);
            let v = resolve_vert_3d(verts, raw_v);
            let e_planes = edges[e_idx].planes.clone();
            let mut e_fw = edges[e_idx].face_windings.clone();
            if raw_u > raw_v {
                for fw in &mut e_fw {
                    fw.1 = -fw.1;
                }
            }

            let splits = &mut edge_splits[e_idx];
            splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

            let mut chain: Vec<u32> = Vec::with_capacity(splits.len() + 2);
            chain.push(u);
            for &(_, sv) in splits.iter() {
                let rsv = resolve_vert_3d(verts, sv);
                if rsv != v && !chain.contains(&rsv) {
                    chain.push(rsv);
                }
            }
            if chain.last() != Some(&v) {
                chain.push(v);
            }

            if chain.len() > 2 {
                edges[e_idx].alive = false;
                changed = true;
                for &cv in &chain {
                    let c_idx = cv as usize;
                    if c_idx >= next_active.len() {
                        next_active.resize(c_idx + 1, false);
                    }
                    mark_neighborhood_active(c_idx, &mut next_active, vert_adj, edges);
                }
                for k in 0..(chain.len() - 1) {
                    inject_edge_3d_adj(edges, vert_adj, chain[k], chain[k + 1], &e_planes, &e_fw);
                }
            }
        }

        if !changed {
            break;
        }
        if iter >= 0 {
            next_active.resize(verts.len(), true);
            // Expand next_active by 1 hop along alive edges
            let seed = next_active.clone();
            for e in edges.iter() {
                if !e.alive {
                    continue;
                }
                let u = e.v0 as usize;
                let v = e.v1 as usize;
                if seed[u] || seed[v] {
                    next_active[u] = true;
                    next_active[v] = true;
                }
            }
            *vert_active = next_active;
        }
    }

    // Final quadric representative projection onto carrier planes
    project_vertices_to_carrier_planes_3d(verts, planes, r_max);
}

#[derive(Clone, Debug)]
struct FacePatch3D {
    plane_id: usize,
    merged_planes: Vec<usize>,
    /// Directed edges `(edge_id, dir_sign)` where `dir_sign = +1` for `v0 -> v1` and `-1` for `v1 -> v0`.
    directed_edges: Vec<(usize, i32)>,
    outer_vids: Vec<u32>,
    holes_vids: Vec<Vec<u32>>,
    sample_2d: Point2D,
    area_2d: f64,
    delta_w: Vec<i32>,
    alive: bool,
}

struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut i: usize) -> usize {
        while self.parent[i] != i {
            let p = self.parent[i];
            self.parent[i] = self.parent[p];
            i = p;
        }
        i
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            if ra < rb {
                self.parent[rb] = ra;
            } else {
                self.parent[ra] = rb;
            }
        }
    }
}

/// Evaluate the consolidated 2D winding number of input face `nf_idx` at query point `pt`
/// using the subtended angle of consolidated 3D edges carrying `nf_idx`.
fn consolidated_face_winding_2d(
    pt: Point2D,
    nf_idx: usize,
    plane_edges: &[usize],
    edges: &[Edge3D],
    verts: &[Vert3D],
    pl: &CanonicalPlane,
) -> i32 {
    let mut total_angle = 0.0;
    let mut found_any = false;
    for &e_id in plane_edges {
        let e = &edges[e_id];
        if !e.alive {
            continue;
        }
        let mut dw = 0i32;
        for &(id, d) in &e.face_windings {
            if id == nf_idx {
                dw = d;
                break;
            }
        }
        if dw == 0 {
            continue;
        }
        found_any = true;
        let a2 = pl.project_2d(verts[e.v0 as usize].rep);
        let b2 = pl.project_2d(verts[e.v1 as usize].rep);
        let ax = a2.x - pt.x;
        let ay = a2.y - pt.y;
        let bx = b2.x - pt.x;
        let by = b2.y - pt.y;
        let d_theta = (ax * by - ay * bx).atan2(ax * bx + ay * by);
        total_angle += (dw as f64) * d_theta;
    }
    if !found_any {
        return 0;
    }
    (total_angle / (2.0 * core::f64::consts::PI)).round() as i32
}

/// Compute a guaranteed interior 2D sample point inside `(outer_vids, holes_vids)` on plane `pl`.
fn compute_patch_interior_sample_2d(
    outer_vids: &[u32],
    holes_vids: &[Vec<u32>],
    verts: &[Vert3D],
    pl: &CanonicalPlane,
) -> Point2D {
    let n = outer_vids.len();
    if holes_vids.is_empty() {
        if n == 3 {
            let a = pl.project_2d(verts[outer_vids[0] as usize].rep);
            let b = pl.project_2d(verts[outer_vids[1] as usize].rep);
            let c = pl.project_2d(verts[outer_vids[2] as usize].rep);
            let la = (b.x - c.x).hypot(b.y - c.y);
            let lb = (c.x - a.x).hypot(c.y - a.y);
            let lc = (a.x - b.x).hypot(a.y - b.y);
            let perim = la + lb + lc;
            if perim > 1e-12 {
                return Point2D {
                    x: (la * a.x + lb * b.x + lc * c.x) / perim,
                    y: (la * a.y + lb * b.y + lc * c.y) / perim,
                };
            }
            return Point2D {
                x: (a.x + b.x + c.x) * (1.0 / 3.0),
                y: (a.y + b.y + c.y) * (1.0 / 3.0),
            };
        } else if n == 4 {
            let p0 = pl.project_2d(verts[outer_vids[0] as usize].rep);
            let p1 = pl.project_2d(verts[outer_vids[1] as usize].rep);
            let p2 = pl.project_2d(verts[outer_vids[2] as usize].rep);
            let p3 = pl.project_2d(verts[outer_vids[3] as usize].rep);
            let a012 = cross_2d(p0, p1, p2);
            let a023 = cross_2d(p0, p2, p3);
            let a013 = cross_2d(p0, p1, p3);
            let a123 = cross_2d(p1, p2, p3);
            let tri = if a012 > 1e-11 && a023 > 1e-11 {
                if a012 >= a023 {
                    Some((p0, p1, p2))
                } else {
                    Some((p0, p2, p3))
                }
            } else if a013 > 1e-11 && a123 > 1e-11 {
                if a013 >= a123 {
                    Some((p0, p1, p3))
                } else {
                    Some((p1, p2, p3))
                }
            } else {
                None
            };
            if let Some((a, b, c)) = tri {
                let la = (b.x - c.x).hypot(b.y - c.y);
                let lb = (c.x - a.x).hypot(c.y - a.y);
                let lc = (a.x - b.x).hypot(a.y - b.y);
                let perim = la + lb + lc;
                if perim > 1e-12 {
                    return Point2D {
                        x: (la * a.x + lb * b.x + lc * c.x) / perim,
                        y: (la * a.y + lb * b.y + lc * c.y) / perim,
                    };
                }
                return Point2D {
                    x: (a.x + b.x + c.x) * (1.0 / 3.0),
                    y: (a.y + b.y + c.y) * (1.0 / 3.0),
                };
            }
        }
    }

    let outer_2d: Vec<Point2D> = outer_vids
        .iter()
        .map(|&v| pl.project_2d(verts[v as usize].rep))
        .collect();
    if holes_vids.is_empty() && n >= 5 {
        for i in 0..n {
            let a = outer_2d[(i + n - 1) % n];
            let b = outer_2d[i];
            let c = outer_2d[(i + 1) % n];
            if cross_2d(a, b, c) <= 1e-11 {
                continue;
            }
            let mut ear_ok = true;
            for j in 0..n {
                if j == i || j == (i + n - 1) % n || j == (i + 1) % n {
                    continue;
                }
                if point_in_triangle_2d(outer_2d[j], a, b, c) {
                    ear_ok = false;
                    break;
                }
            }
            if ear_ok {
                let la = (b.x - c.x).hypot(b.y - c.y);
                let lb = (c.x - a.x).hypot(c.y - a.y);
                let lc = (a.x - b.x).hypot(a.y - b.y);
                let perim = la + lb + lc;
                if perim > 1e-12 {
                    return Point2D {
                        x: (la * a.x + lb * b.x + lc * c.x) / perim,
                        y: (la * a.y + lb * b.y + lc * c.y) / perim,
                    };
                }
                return Point2D {
                    x: (a.x + b.x + c.x) * (1.0 / 3.0),
                    y: (a.y + b.y + c.y) * (1.0 / 3.0),
                };
            }
        }
    }
    let holes_2d: Vec<Vec<Point2D>> = holes_vids
        .iter()
        .map(|h| {
            h.iter()
                .map(|&v| pl.project_2d(verts[v as usize].rep))
                .collect()
        })
        .collect();

    let tris = triangulate_polygon_2d(&outer_2d, &holes_2d);
    let mut best_area = -1.0f64;
    let mut best_pt = outer_2d.first().copied().unwrap_or(Point2D { x: 0.0, y: 0.0 });
    for [a, b, c] in tris {
        let area = cross_2d(a, b, c).abs();
        if area > best_area {
            best_area = area;
            let la = (b.x - c.x).hypot(b.y - c.y);
            let lb = (c.x - a.x).hypot(c.y - a.y);
            let lc = (a.x - b.x).hypot(a.y - b.y);
            let perim = la + lb + lc;
            if perim > 1e-12 {
                best_pt = Point2D {
                    x: (la * a.x + lb * b.x + lc * c.x) / perim,
                    y: (la * a.y + lb * b.y + lc * c.y) / perim,
                };
            } else {
                best_pt = Point2D {
                    x: (a.x + b.x + c.x) / 3.0,
                    y: (a.y + b.y + c.y) / 3.0,
                };
            }
        }
    }
    best_pt
}

/// Stage 3a: Extract minimal 2D `FacePatch3D` sheets on carrier plane `p_id` from consolidated 3D edges.
fn extract_plane_face_patches_3d(
    p_id: usize,
    pl: &CanonicalPlane,
    plane_edges: &[usize],
    on_plane_faces: &[usize],
    norm_faces: &[NormalizedInputFace],
    verts: &[Vert3D],
    edges: &[Edge3D],
    body_count: usize,
    winding_rule: WindingRule,
    out_patches: &mut Vec<FacePatch3D>,
) {
    if plane_edges.len() < 3 {
        return;
    }

    let mut local_vids: Vec<u32> = Vec::with_capacity(plane_edges.len() * 2);
    for &e_id in plane_edges {
        local_vids.push(edges[e_id].v0);
        local_vids.push(edges[e_id].v1);
    }
    local_vids.sort_unstable();
    local_vids.dedup();
    let num_lv = local_vids.len();
    if num_lv < 3 {
        return;
    }
    let to_lv = |vid: u32| -> usize { local_vids.binary_search(&vid).unwrap() };

    // Iteratively prune degree-1 vertices (2D whiskers) on `p_id`
    let mut active_pe: Vec<bool> = vec![true; plane_edges.len()];
    let mut deg: Vec<u32> = vec![0; num_lv];
    loop {
        deg.fill(0);
        for (idx, &e_id) in plane_edges.iter().enumerate() {
            if active_pe[idx] {
                deg[to_lv(edges[e_id].v0)] += 1;
                deg[to_lv(edges[e_id].v1)] += 1;
            }
        }
        let mut pruned = false;
        for (idx, &e_id) in plane_edges.iter().enumerate() {
            if active_pe[idx]
                && (deg[to_lv(edges[e_id].v0)] < 2 || deg[to_lv(edges[e_id].v1)] < 2)
            {
                active_pe[idx] = false;
                pruned = true;
            }
        }
        if !pruned {
            break;
        }
    }

    let surviving_edges: Vec<usize> = plane_edges
        .iter()
        .enumerate()
        .filter_map(|(i, &e_id)| if active_pe[i] { Some(e_id) } else { None })
        .collect();
    if surviving_edges.len() < 3 {
        return;
    }

    // Build planar half-edges: `he = 2 * k` (`v0 -> v1`), `he = 2 * k + 1` (`v1 -> v0`)
    let num_se = surviving_edges.len();
    let num_he = 2 * num_se;
    let mut he_orig: Vec<u32> = Vec::with_capacity(num_he);
    let mut he_orig_lv: Vec<usize> = Vec::with_capacity(num_he);
    let mut he_dest_lv: Vec<usize> = Vec::with_capacity(num_he);
    let mut he_edge: Vec<usize> = Vec::with_capacity(num_he);
    let mut he_sign: Vec<i32> = Vec::with_capacity(num_he);

    let mut vert_out: Vec<Vec<usize>> = vec![Vec::new(); num_lv];
    for (k, &e_id) in surviving_edges.iter().enumerate() {
        let u = edges[e_id].v0;
        let v = edges[e_id].v1;
        let lu = to_lv(u);
        let lv = to_lv(v);
        let h0 = 2 * k;
        let h1 = 2 * k + 1;
        he_orig.push(u);
        he_orig_lv.push(lu);
        he_dest_lv.push(lv);
        he_edge.push(e_id);
        he_sign.push(1);
        vert_out[lu].push(h0);

        he_orig.push(v);
        he_orig_lv.push(lv);
        he_dest_lv.push(lu);
        he_edge.push(e_id);
        he_sign.push(-1);
        vert_out[lv].push(h1);
    }

    let v_2d: Vec<Point2D> = local_vids
        .iter()
        .map(|&vid| pl.project_2d(verts[vid as usize].rep))
        .collect();

    // Sort outgoing half-edges CCW by 2D angle at each vertex
    let mut he_next: Vec<usize> = vec![0; num_he];
    for lv_id in 0..num_lv {
        let spokes = &mut vert_out[lv_id];
        if spokes.len() <= 2 {
            continue;
        }
        let pv = v_2d[lv_id];
        spokes.sort_by(|&ha, &hb| {
            let da = v_2d[he_dest_lv[ha]];
            let db = v_2d[he_dest_lv[hb]];
            let ang_a = (da.y - pv.y).atan2(da.x - pv.x);
            let ang_b = (db.y - pv.y).atan2(db.x - pv.x);
            ang_a
                .partial_cmp(&ang_b)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
    }

    // Wire `next(h) = prev_ccw_spoke(twin(h))`
    for h in 0..num_he {
        let twin = h ^ 1;
        let target_lv = he_dest_lv[h];
        let spokes = &vert_out[target_lv];
        let pos = spokes.iter().position(|&sh| sh == twin).unwrap();
        let prev_idx = (pos + spokes.len() - 1) % spokes.len();
        he_next[h] = spokes[prev_idx];
    }

    // Trace closed half-edge cycles and decompose pinched figure-8 vertices into simple vertex cycles
    let mut visited = vec![false; num_he];
    let mut outer_cycles: Vec<(Vec<u32>, Vec<(usize, i32)>, f64)> = Vec::new();
    let mut hole_cycles: Vec<(Vec<u32>, Vec<(usize, i32)>, f64)> = Vec::new();

    for h_start in 0..num_he {
        if visited[h_start] {
            continue;
        }
        let mut raw_he_seq: Vec<usize> = Vec::new();
        let mut cur = h_start;
        while !visited[cur] {
            visited[cur] = true;
            raw_he_seq.push(cur);
            cur = he_next[cur];
        }
        if cur != h_start || raw_he_seq.len() < 3 {
            continue;
        }

        // Decompose at repeated vertices or spikes
        let mut v_stack: Vec<u32> = Vec::with_capacity(raw_he_seq.len());
        let mut e_stack: Vec<(usize, i32)> = Vec::with_capacity(raw_he_seq.len());
        let mut simple_loops: Vec<(Vec<u32>, Vec<(usize, i32)>)> = Vec::new();

        for &h in &raw_he_seq {
            let u = he_orig[h];
            let e_pair = (he_edge[h], he_sign[h]);
            if let Some(pos) = v_stack.iter().position(|&sv| sv == u) {
                let sub_v = v_stack[pos..].to_vec();
                let sub_e = e_stack[pos..].to_vec();
                if sub_v.len() >= 3 {
                    simple_loops.push((sub_v, sub_e));
                }
                v_stack.truncate(pos);
                e_stack.truncate(pos);
            }
            v_stack.push(u);
            e_stack.push(e_pair);
        }
        if v_stack.len() >= 3 {
            simple_loops.push((v_stack, e_stack));
        }

        for (mut sv, mut se) in simple_loops {
            // Strip any self-cancelling adjacent spike `(e, +s)` followed by `(e, -s)`
            let mut chg = true;
            while chg && sv.len() >= 3 {
                chg = false;
                let n = sv.len();
                for i in 0..n {
                    let j = (i + 1) % n;
                    if se[i].0 == se[j].0 && se[i].1 == -se[j].1 {
                        if i < j {
                            sv.remove(j);
                            se.remove(j);
                            sv.remove(i);
                            se.remove(i);
                        } else {
                            sv.remove(i);
                            se.remove(i);
                            sv.remove(j);
                            se.remove(j);
                        }
                        chg = true;
                        break;
                    }
                }
            }
            if sv.len() < 3 {
                continue;
            }
            let pts_2d: Vec<Point2D> = sv.iter().map(|&vid| v_2d[to_lv(vid)]).collect();
            let area = signed_area_2d_pts(&pts_2d);
            if area > 1e-11 {
                outer_cycles.push((sv, se, area));
            } else if area < -1e-11 {
                hole_cycles.push((sv, se, area.abs()));
            }
        }
    }

    if outer_cycles.is_empty() {
        return;
    }

    // Pair each negative-area interior hole cycle with its smallest enclosing positive-area outer cycle
    let mut patch_holes_vids: Vec<Vec<Vec<u32>>> = vec![Vec::new(); outer_cycles.len()];
    let mut patch_holes_edges: Vec<Vec<(usize, i32)>> = vec![Vec::new(); outer_cycles.len()];
    let outer_2d_loops: Vec<Vec<Point2D>> = outer_cycles
        .iter()
        .map(|(ov, _, _)| ov.iter().map(|&vid| v_2d[to_lv(vid)]).collect())
        .collect();

    for (hv, he, h_area) in hole_cycles {
        let test_pt = hv
            .iter()
            .map(|&vid| v_2d[to_lv(vid)])
            .find(|hp| {
                outer_2d_loops.iter().all(|sh| {
                    sh.iter()
                        .all(|sp| (sp.x - hp.x).hypot(sp.y - hp.y) > 1e-8)
                })
            })
            .unwrap_or_else(|| {
                let sum_x: f64 = hv.iter().map(|&vid| v_2d[to_lv(vid)].x).sum();
                let sum_y: f64 = hv.iter().map(|&vid| v_2d[to_lv(vid)].y).sum();
                Point2D {
                    x: sum_x / (hv.len() as f64),
                    y: sum_y / (hv.len() as f64),
                }
            });

        let mut best_idx = None;
        let mut best_area = f64::INFINITY;
        for (o_idx, (_, _, o_area)) in outer_cycles.iter().enumerate() {
            if *o_area <= h_area + 1e-11 {
                continue;
            }
            if loops_winding_2d(test_pt, core::slice::from_ref(&outer_2d_loops[o_idx])) != 0
                && *o_area < best_area
            {
                best_area = *o_area;
                best_idx = Some(o_idx);
            }
        }
        if let Some(o_idx) = best_idx {
            patch_holes_vids[o_idx].push(hv);
            patch_holes_edges[o_idx].extend(he);
        }
    }

    for (o_idx, (ov, mut oe, o_area)) in outer_cycles.into_iter().enumerate() {
        let hv = core::mem::take(&mut patch_holes_vids[o_idx]);
        oe.extend(core::mem::take(&mut patch_holes_edges[o_idx]));

        let mut net_area = o_area;
        for h_ring in &hv {
            let h_pts: Vec<Point2D> = h_ring.iter().map(|&vid| v_2d[to_lv(vid)]).collect();
            net_area -= signed_area_2d_pts(&h_pts).abs();
        }
        if net_area <= 1e-11 {
            continue;
        }

        let sample_2d = compute_patch_interior_sample_2d(&ov, &hv, verts, pl);

        let mut delta_w = vec![0i32; body_count];
        if on_plane_faces.len() == 1 && winding_rule.apply(1.0) {
            let nf = &norm_faces[on_plane_faces[0]];
            let all_interior_or_ccw = oe.iter().all(|&(e_id, dir_sign)| {
                let fw = &edges[e_id].face_windings;
                fw.is_empty() || (fw.len() == 1 && fw[0].1 == dir_sign)
            });
            if all_interior_or_ccw {
                delta_w[nf.body_index] = nf.orient_sign;
            } else {
                let w2d = consolidated_face_winding_2d(
                    sample_2d,
                    on_plane_faces[0],
                    &surviving_edges,
                    edges,
                    verts,
                    pl,
                );
                if w2d != 0 && winding_rule.apply(w2d as f64) {
                    delta_w[nf.body_index] += nf.orient_sign;
                }
            }
        } else {
            for &nf_idx in on_plane_faces {
                let nf = &norm_faces[nf_idx];
                let w2d = consolidated_face_winding_2d(
                    sample_2d,
                    nf_idx,
                    &surviving_edges,
                    edges,
                    verts,
                    pl,
                );
                if w2d != 0 && winding_rule.apply(w2d as f64) {
                    delta_w[nf.body_index] += nf.orient_sign;
                }
            }
        }

        if delta_w.iter().any(|&dw| dw != 0) {
            out_patches.push(FacePatch3D {
                plane_id: p_id,
                merged_planes: vec![p_id],
                directed_edges: oe,
                outer_vids: ov,
                holes_vids: hv,
                sample_2d,
                area_2d: net_area,
                delta_w,
                alive: true,
            });
        }
    }
}

/// Core 3D Planar B-Rep Boolean and healing execution engine.
pub fn execute_brep_3d<F>(
    bodies: &[&Solid3D],
    mut predicate: F,
    options: KernelOptions,
) -> BRepResult3D
where
    F: FnMut(&[f64]) -> bool,
{
    let body_count = bodies.len();
    if body_count == 0 {
        return BRepResult3D::default();
    }

    // Stage 0: Compute global 3D bounding box for coordinate centering & scale-aware tolerance
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut min_z = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    let mut any_pt = false;

    for solid in bodies {
        for face in &solid.faces {
            for pt in &face.outer {
                any_pt = true;
                min_x = min_x.min(pt.x);
                min_y = min_y.min(pt.y);
                min_z = min_z.min(pt.z);
                max_x = max_x.max(pt.x);
                max_y = max_y.max(pt.y);
                max_z = max_z.max(pt.z);
            }
        }
    }

    if !any_pt {
        return BRepResult3D::default();
    }

    let center = Point3D::new(
        0.5 * (min_x + max_x),
        0.5 * (min_y + max_y),
        0.5 * (min_z + max_z),
    );
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let dz = max_z - min_z;
    let diagonal = (dx * dx + dy * dy + dz * dz).sqrt().max(1.0);
    let r0 = options
        .snap_tolerance
        .unwrap_or(1e-8 * diagonal)
        .max(1e-9 * diagonal);
    let plane_dist_tol = r0;
    let plane_cos_tol = 1.0 - 1e-9;

    // Cluster input faces into Canonical Carrier Planes
    let mut planes: Vec<CanonicalPlane> = Vec::new();
    let mut plane_buckets: [Vec<usize>; 256] = std::array::from_fn(|_| Vec::new());
    let use_plane_grid = plane_dist_tol <= 1e-4 * diagonal;
    let mut norm_faces: Vec<NormalizedInputFace> = Vec::new();

    for (b_idx, solid) in bodies.iter().enumerate() {
        for face in &solid.faces {
            if face.outer.len() < 3 {
                continue;
            }
            let mut shifted_outer: Vec<Point3D> =
                face.outer.iter().map(|p| p.sub(center)).collect();
            let raw_normal = match compute_newell_normal(&shifted_outer) {
                Some(n) => n,
                None => continue,
            };

            let mut f_min = Point3D::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
            let mut f_max = Point3D::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
            let mut avg = Point3D::default();
            for &p in &shifted_outer {
                avg = avg.add(p);
                f_min.x = f_min.x.min(p.x);
                f_min.y = f_min.y.min(p.y);
                f_min.z = f_min.z.min(p.z);
                f_max.x = f_max.x.max(p.x);
                f_max.y = f_max.y.max(p.y);
                f_max.z = f_max.z.max(p.z);
            }
            avg = avg.scale(1.0 / (shifted_outer.len() as f64));
            let raw_d = raw_normal.dot(avg);
            let (can_n, can_d, orient_sign) = canonicalize_plane(raw_normal, raw_d);

            let mut found_plane = None;
            if use_plane_grid {
                let bx0 = (((can_n.x + 1.0) * 7.5 - 1e-3).floor() as isize).clamp(0, 15) as usize;
                let bx1 = (((can_n.x + 1.0) * 7.5 + 1e-3).ceil() as isize).clamp(0, 15) as usize;
                let by0 = (((can_n.y + 1.0) * 7.5 - 1e-3).floor() as isize).clamp(0, 15) as usize;
                let by1 = (((can_n.y + 1.0) * 7.5 + 1e-3).ceil() as isize).clamp(0, 15) as usize;
                'search_bucket: for bx in bx0..=bx1 {
                    for by in by0..=by1 {
                        for &p_idx in &plane_buckets[(bx << 4) | by] {
                            let pl = &mut planes[p_idx];
                            if (pl.d - can_d).abs() <= plane_dist_tol
                                && pl.normal.dot(can_n) >= plane_cos_tol
                            {
                                pl.min_pt.x = pl.min_pt.x.min(f_min.x);
                                pl.min_pt.y = pl.min_pt.y.min(f_min.y);
                                pl.min_pt.z = pl.min_pt.z.min(f_min.z);
                                pl.max_pt.x = pl.max_pt.x.max(f_max.x);
                                pl.max_pt.y = pl.max_pt.y.max(f_max.y);
                                pl.max_pt.z = pl.max_pt.z.max(f_max.z);
                                found_plane = Some(p_idx);
                                break 'search_bucket;
                            }
                        }
                    }
                }
            } else {
                for (p_idx, pl) in planes.iter_mut().enumerate() {
                    if (pl.d - can_d).abs() <= plane_dist_tol
                        && pl.normal.dot(can_n) >= plane_cos_tol
                    {
                        pl.min_pt.x = pl.min_pt.x.min(f_min.x);
                        pl.min_pt.y = pl.min_pt.y.min(f_min.y);
                        pl.min_pt.z = pl.min_pt.z.min(f_min.z);
                        pl.max_pt.x = pl.max_pt.x.max(f_max.x);
                        pl.max_pt.y = pl.max_pt.y.max(f_max.y);
                        pl.max_pt.z = pl.max_pt.z.max(f_max.z);
                        found_plane = Some(p_idx);
                        break;
                    }
                }
            }

            let plane_id = match found_plane {
                Some(idx) => idx,
                None => {
                    let (u_axis, v_axis) = build_plane_frame(can_n);
                    let origin = can_n.scale(can_d);
                    let idx = planes.len();
                    planes.push(CanonicalPlane {
                        origin,
                        u_axis,
                        v_axis,
                        normal: can_n,
                        d: can_d,
                        min_pt: f_min,
                        max_pt: f_max,
                    });
                    if use_plane_grid {
                        let bx = (((can_n.x + 1.0) * 7.5).floor() as isize).clamp(0, 15) as usize;
                        let by = (((can_n.y + 1.0) * 7.5).floor() as isize).clamp(0, 15) as usize;
                        plane_buckets[(bx << 4) | by].push(idx);
                    }
                    idx
                }
            };

            let pl = &planes[plane_id];
            let mut outer_2d: Vec<Point2D> =
                shifted_outer.iter().map(|&p| pl.project_2d(p)).collect();
            let raw_area = signed_area_2d_pts(&outer_2d);
            if raw_area < 0.0 {
                outer_2d.reverse();
                shifted_outer.reverse();
            }
            let area_2d = raw_area.abs();

            let mut holes_2d: Vec<Vec<Point2D>> = Vec::with_capacity(face.holes.len());
            let mut holes_3d_loops: Vec<Vec<Point3D>> = Vec::with_capacity(face.holes.len());
            for hole in &face.holes {
                if hole.len() < 3 {
                    continue;
                }
                let mut shifted_h: Vec<Point3D> = hole.iter().map(|p| p.sub(center)).collect();
                let mut h_2d: Vec<Point2D> =
                    shifted_h.iter().map(|&p| pl.project_2d(p)).collect();
                if signed_area_2d_pts(&h_2d) > 0.0 {
                    h_2d.reverse();
                    shifted_h.reverse();
                }
                holes_2d.push(h_2d);
                holes_3d_loops.push(shifted_h);
            }

            let mut loops_2d = Vec::with_capacity(1 + holes_2d.len());
            loops_2d.push(outer_2d);
            loops_2d.extend(holes_2d);

            let mut loops_3d = Vec::with_capacity(1 + holes_3d_loops.len());
            loops_3d.push(shifted_outer);
            loops_3d.extend(holes_3d_loops);

            norm_faces.push(NormalizedInputFace {
                body_index: b_idx,
                plane_id,
                orient_sign,
                area_2d,
                loops_2d,
                loops_3d,
                min_pt: f_min,
                max_pt: f_max,
            });
        }
    }

    let mut faces_on_plane: Vec<Vec<usize>> = vec![Vec::new(); planes.len()];
    let mut faces_by_body: Vec<Vec<usize>> = vec![Vec::new(); body_count];
    let mut body_min = vec![
        Point3D::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        body_count
    ];
    let mut body_max = vec![
        Point3D::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        body_count
    ];
    for (f_idx, nf) in norm_faces.iter().enumerate() {
        faces_on_plane[nf.plane_id].push(f_idx);
        let b = nf.body_index;
        faces_by_body[b].push(f_idx);
        body_min[b].x = body_min[b].x.min(nf.min_pt.x);
        body_min[b].y = body_min[b].y.min(nf.min_pt.y);
        body_min[b].z = body_min[b].z.min(nf.min_pt.z);
        body_max[b].x = body_max[b].x.max(nf.max_pt.x);
        body_max[b].y = body_max[b].y.max(nf.max_pt.y);
        body_max[b].z = body_max[b].z.max(nf.max_pt.z);
    }

    let r_init = 0.5 * r0;
    let r_max = 2.0 * r0;

    // Stage 1: Unconstrained 3D Locus Injection (`verts`, `edges`)
    let mut verts: Vec<Vert3D> = Vec::new();
    let mut edges: Vec<Edge3D> = Vec::new();
    let mut vert_adj: Vec<Vec<usize>> = Vec::new();
    let mut nf_outer_vids: Vec<Vec<u32>> = vec![Vec::new(); norm_faces.len()];
    let mut nf_all_vids: Vec<Vec<u32>> = vec![Vec::new(); norm_faces.len()];

    // 1a. Inject all input face boundary loops into the global 3D complex
    let x_span_inv = if dx > 1e-9 { 15.0 / dx } else { 0.0 };
    let y_span_inv = if dy > 1e-9 { 15.0 / dy } else { 0.0 };
    let x_low = min_x - center.x;
    let y_low = min_y - center.y;
    let mut vert_xy_buckets: [Vec<u32>; 256] = std::array::from_fn(|_| Vec::new());

    for (nf_idx, nf) in norm_faces.iter().enumerate() {
        let p_tag = [nf.plane_id];
        let fw_tag = [(nf_idx, 1i32)];
        for (loop_idx, ring_3d) in nf.loops_3d.iter().enumerate() {
            let n = ring_3d.len();
            if n < 3 {
                continue;
            }
            let mut ring_vids: Vec<u32> = Vec::with_capacity(n);
            for &pt in ring_3d {
                let vid = if r0 <= 1e-4 * diagonal && dx > 1e-9 && dy > 1e-9 {
                    let bx_min = (((pt.x - r0 - x_low) * x_span_inv).floor() as isize).clamp(0, 15) as usize;
                    let bx_max = (((pt.x + r0 - x_low) * x_span_inv).ceil() as isize).clamp(0, 15) as usize;
                    let by_min = (((pt.y - r0 - y_low) * y_span_inv).floor() as isize).clamp(0, 15) as usize;
                    let by_max = (((pt.y + r0 - y_low) * y_span_inv).ceil() as isize).clamp(0, 15) as usize;
                    let mut best_v = None;
                    let mut best_d_sq = r0 * r0;
                    for bx in bx_min..=bx_max {
                        for by in by_min..=by_max {
                            for &cand_id in &vert_xy_buckets[(bx << 4) | by] {
                                let v_cand = &verts[cand_id as usize];
                                if !v_cand.alive {
                                    continue;
                                }
                                let d_sq = v_cand.pos.sub(pt).norm_sq();
                                if d_sq <= best_d_sq {
                                    best_d_sq = d_sq;
                                    best_v = Some(cand_id);
                                }
                            }
                        }
                    }
                    if let Some(found_id) = best_v {
                        let root = resolve_vert_3d(&mut verts, found_id);
                        if merge_plane_tags(&mut verts[root as usize].planes, &p_tag) {
                            verts[root as usize].dirty_proj = true;
                        }
                        root
                    } else {
                        let id = verts.len() as u32;
                        let mut planes_v = Vec::with_capacity(4);
                        merge_plane_tags(&mut planes_v, &p_tag);
                        verts.push(Vert3D {
                            pos: pt,
                            rep: pt,
                            radius: r_init,
                            weight: 1,
                            parent: id,
                            planes: planes_v,
                            dirty_proj: true,
                            alive: true,
                        });
                        let bx_cur = (((pt.x - x_low) * x_span_inv).floor() as isize).clamp(0, 15) as usize;
                        let by_cur = (((pt.y - y_low) * y_span_inv).floor() as isize).clamp(0, 15) as usize;
                        vert_xy_buckets[(bx_cur << 4) | by_cur].push(id);
                        id
                    }
                } else {
                    find_or_create_vert_3d(&mut verts, pt, r0, r_init, &p_tag)
                };
                if vert_adj.len() < verts.len() {
                    vert_adj.resize_with(verts.len(), Vec::new);
                }
                ring_vids.push(vid);
            }
            for i in 0..n {
                let u = resolve_vert_3d(&mut verts, ring_vids[i]);
                let v = resolve_vert_3d(&mut verts, ring_vids[(i + 1) % n]);
                ring_vids[i] = u;
                if u == v {
                    continue;
                }
                let (a, b, sign) = if u < v { (u, v, 1) } else { (v, u, -1) };
                let mut found_e = false;
                for &e_id in &vert_adj[a as usize] {
                    let e = &mut edges[e_id];
                    if e.alive && e.v0 == a && e.v1 == b {
                        merge_plane_tags(&mut e.planes, &p_tag);
                        merge_face_windings(&mut e.face_windings, &fw_tag, sign);
                        found_e = true;
                        break;
                    }
                }
                if !found_e {
                    let e_id = edges.len();
                    let mut planes_vec = Vec::with_capacity(4);
                    merge_plane_tags(&mut planes_vec, &p_tag);
                    let mut fw = Vec::with_capacity(2);
                    merge_face_windings(&mut fw, &fw_tag, sign);
                    edges.push(Edge3D {
                        v0: a,
                        v1: b,
                        planes: planes_vec,
                        face_windings: fw,
                        alive: true,
                    });
                    vert_adj[a as usize].push(e_id);
                }
            }
            if loop_idx == 0 {
                nf_outer_vids[nf_idx] = ring_vids.clone();
            }
            nf_all_vids[nf_idx].extend_from_slice(&ring_vids);
        }
    }

    let num_stage1a_verts = verts.len();
    let num_stage1a_edges = edges.len();
    let all_bodies_closed = edges[..num_stage1a_edges].iter().all(|e| {
        if !e.alive {
            return true;
        }
        if e.face_windings.len() != 2 {
            return false;
        }
        let (f0, s0) = e.face_windings[0];
        let (f1, s1) = e.face_windings[1];
        norm_faces[f0].body_index == norm_faces[f1].body_index
            && s0 * norm_faces[f0].orient_sign + s1 * norm_faces[f1].orient_sign == 0
    });
    let mut plane_active_1b: Vec<bool> = vec![false; planes.len()];

    // 1b. Inject all 1D polygon-plane intersection chords where faces on `π_b ≠ π_c` cross `π_c`
    let skip_closed_same_body = all_bodies_closed && r0 <= 1e-4 * diagonal && body_count >= 2;
    let z_low = min_z - center.z;
    let vox_x_inv = if dx > 1e-9 { 31.0 / dx } else { 0.0 };
    let vox_y_inv = if dy > 1e-9 { 31.0 / dy } else { 0.0 };
    let vox_z_inv = if dz > 1e-9 { 31.0 / dz } else { 0.0 };
    let vox_pad = 4.0 * r_max + plane_dist_tol;

    let mut body_vox: Vec<[u32; 1024]> = vec![[0u32; 1024]; body_count];
    if body_count >= 2 && r0 <= 1e-4 * diagonal {
        for nf in &norm_faces {
            let b = nf.body_index;
            let vx0 = (((nf.min_pt.x - vox_pad - x_low) * vox_x_inv).floor() as isize).clamp(0, 31) as usize;
            let vx1 = (((nf.max_pt.x + vox_pad - x_low) * vox_x_inv).floor() as isize).clamp(0, 31) as usize;
            let vy0 = (((nf.min_pt.y - vox_pad - y_low) * vox_y_inv).floor() as isize).clamp(0, 31) as usize;
            let vy1 = (((nf.max_pt.y + vox_pad - y_low) * vox_y_inv).floor() as isize).clamp(0, 31) as usize;
            let vz0 = (((nf.min_pt.z - vox_pad - z_low) * vox_z_inv).floor() as isize).clamp(0, 31) as u32;
            let vz1 = (((nf.max_pt.z + vox_pad - z_low) * vox_z_inv).floor() as isize).clamp(0, 31) as u32;
            let width = vz1 - vz0 + 1;
            let z_mask = if width >= 32 {
                u32::MAX
            } else {
                ((1u32 << width) - 1) << vz0
            };
            let vox = &mut body_vox[b];
            for vx in vx0..=vx1 {
                let row = vx << 5;
                for vy in vy0..=vy1 {
                    vox[row | vy] |= z_mask;
                }
            }
        }
    }

    let y64_inv = if dy > 1e-9 { 63.0 / dy } else { 0.0 };
    let z64_inv = if dz > 1e-9 { 63.0 / dz } else { 0.0 };
    let range_mask_64 = |lo: f64, hi: f64, base: f64, inv: f64| -> u64 {
        if inv == 0.0 {
            return u64::MAX;
        }
        let b0 = (((lo - base) * inv).floor() as isize).clamp(0, 63) as u32;
        let b1 = (((hi - base) * inv).floor() as isize).clamp(0, 63) as u32;
        let width = b1 - b0 + 1;
        if width >= 64 {
            u64::MAX
        } else {
            ((1u64 << width) - 1) << b0
        }
    };

    let mut face_near_other_vox: Vec<bool> = vec![true; norm_faces.len()];
    if body_count >= 2 && r0 <= 1e-4 * diagonal {
        let mut vox_cands: Vec<(f64, u64, u64, usize)> = Vec::new();
        for (nf_idx, nf) in norm_faces.iter().enumerate() {
            let b0 = nf.body_index;
            let vx0 = (((nf.min_pt.x - x_low) * vox_x_inv).floor() as isize).clamp(0, 31) as usize;
            let vx1 = (((nf.max_pt.x - x_low) * vox_x_inv).floor() as isize).clamp(0, 31) as usize;
            let vy0 = (((nf.min_pt.y - y_low) * vox_y_inv).floor() as isize).clamp(0, 31) as usize;
            let vy1 = (((nf.max_pt.y - y_low) * vox_y_inv).floor() as isize).clamp(0, 31) as usize;
            let vz0 = (((nf.min_pt.z - z_low) * vox_z_inv).floor() as isize).clamp(0, 31) as u32;
            let vz1 = (((nf.max_pt.z - z_low) * vox_z_inv).floor() as isize).clamp(0, 31) as u32;
            let width = vz1 - vz0 + 1;
            let z_mask = if width >= 32 {
                u32::MAX
            } else {
                ((1u32 << width) - 1) << vz0
            };
            let mut near_any = false;
            'body_loop: for b in 0..body_count {
                if b == b0
                    || !aabb_overlap_3d(nf.min_pt, nf.max_pt, body_min[b], body_max[b], vox_pad)
                {
                    continue;
                }
                let vox = &body_vox[b];
                for vx in vx0..=vx1 {
                    let row = vx << 5;
                    for vy in vy0..=vy1 {
                        if (vox[row | vy] & z_mask) != 0 {
                            near_any = true;
                            break 'body_loop;
                        }
                    }
                }
            }
            face_near_other_vox[nf_idx] = false;
            if near_any {
                let my = range_mask_64(nf.min_pt.y - vox_pad, nf.max_pt.y + vox_pad, y_low, y64_inv);
                let mz = range_mask_64(nf.min_pt.z - vox_pad, nf.max_pt.z + vox_pad, z_low, z64_inv);
                vox_cands.push((nf.min_pt.x, my, mz, nf_idx));
            }
        }
        // Refine voxel candidates with exact padded AABB overlap between different bodies
        vox_cands.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        for i in 0..vox_cands.len() {
            let (_, my_i, mz_i, fi) = vox_cands[i];
            let nf_i = &norm_faces[fi];
            let max_x_i = nf_i.max_pt.x + vox_pad;
            for j in (i + 1)..vox_cands.len() {
                let (min_x_j, my_j, mz_j, fj) = vox_cands[j];
                if min_x_j > max_x_i {
                    break;
                }
                if (my_i & my_j) == 0 || (mz_i & mz_j) == 0 {
                    continue;
                }
                let nf_j = &norm_faces[fj];
                if nf_i.body_index != nf_j.body_index
                    && aabb_overlap_3d(nf_i.min_pt, nf_i.max_pt, nf_j.min_pt, nf_j.max_pt, vox_pad)
                {
                    face_near_other_vox[fi] = true;
                    face_near_other_vox[fj] = true;
                }
            }
        }
    }

    let all_planes_single_body = skip_closed_same_body
        && faces_on_plane.iter().all(|op| {
            op.len() <= 1
                || op
                    .iter()
                    .all(|&fi| norm_faces[fi].body_index == norm_faces[op[0]].body_index)
        });

    let mut faces_by_min_x: Vec<(f64, u64, u64, usize)> =
        Vec::with_capacity(norm_faces.len());
    for (idx, nf) in norm_faces.iter().enumerate() {
        if all_planes_single_body && !face_near_other_vox[idx] {
            continue;
        }
        let my = range_mask_64(nf.min_pt.y, nf.max_pt.y, y_low, y64_inv);
        let mz = range_mask_64(nf.min_pt.z, nf.max_pt.z, z_low, z64_inv);
        faces_by_min_x.push((nf.min_pt.x, my, mz, idx));
    }
    faces_by_min_x
        .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

    let mut on_plane_vids_buf: Vec<u32> = Vec::with_capacity(8);
    for (p_id, pl) in planes.iter().enumerate() {
        let on_plane_faces = &faces_on_plane[p_id];
        if on_plane_faces.is_empty() {
            continue;
        }
        let single_body_on_plane = if skip_closed_same_body {
            let b0 = norm_faces[on_plane_faces[0]].body_index;
            if on_plane_faces.iter().all(|&fi| norm_faces[fi].body_index == b0) {
                Some(b0)
            } else {
                None
            }
        } else {
            None
        };
        if single_body_on_plane.is_some()
            && !on_plane_faces.iter().any(|&fi| face_near_other_vox[fi])
        {
            continue;
        }
        let pl_mask_y = range_mask_64(
            pl.min_pt.y - plane_dist_tol,
            pl.max_pt.y + plane_dist_tol,
            y_low,
            y64_inv,
        );
        let pl_mask_z = range_mask_64(
            pl.min_pt.z - plane_dist_tol,
            pl.max_pt.z + plane_dist_tol,
            z_low,
            z64_inv,
        );
        let max_px = pl.max_pt.x + plane_dist_tol;
        for &(other_min_x, nf_mask_y, nf_mask_z, other_idx) in &faces_by_min_x {
            if other_min_x > max_px {
                break;
            }
            if (pl_mask_y & nf_mask_y) == 0 || (pl_mask_z & nf_mask_z) == 0 {
                continue;
            }
            let other_nf = &norm_faces[other_idx];
            if other_nf.plane_id == p_id {
                continue;
            }
            if single_body_on_plane == Some(other_nf.body_index) {
                continue;
            }
            if !aabb_overlap_3d(
                pl.min_pt,
                pl.max_pt,
                other_nf.min_pt,
                other_nf.max_pt,
                plane_dist_tol,
            ) {
                continue;
            }
            if !plane_straddles_aabb(
                pl.normal,
                pl.d,
                other_nf.min_pt,
                other_nf.max_pt,
                plane_dist_tol,
            ) {
                continue;
            }
            let other_pl = &planes[other_nf.plane_id];
            if !plane_straddles_aabb(
                other_pl.normal,
                other_pl.d,
                pl.min_pt,
                pl.max_pt,
                plane_dist_tol,
            ) {
                continue;
            }

            // Fast Stage 1a vertex-sharing check for simple triangle/quad faces:
            // Two simple triangles/quads on distinct planes `pl != other_pl` that share a Stage 1a
            // edge (`shared_vids >= 2`) already have their unique line of intersection `pl ∩ other_pl`
            // injected as that shared Stage 1a edge.
            if r0 <= 1e-4 * diagonal && on_plane_faces.len() == 1 {
                let tf_idx = on_plane_faces[0];
                let tf = &norm_faces[tf_idx];
                if tf.loops_3d.len() == 1
                    && tf.loops_3d[0].len() <= 4
                    && other_nf.loops_3d.len() == 1
                    && other_nf.loops_3d[0].len() <= 4
                {
                    let tf_vids = &nf_all_vids[tf_idx];
                    let other_vids = &nf_all_vids[other_idx];
                    let mut shared_vids = 0usize;
                    for &uv in other_vids {
                        if tf_vids.contains(&uv) {
                            shared_vids += 1;
                        }
                    }
                    if shared_vids >= 2 {
                        continue;
                    }
                }
            }

            // Fast polygon-vs-plane check using `other_nf`'s actual vertices on `other_pl`:
            // Skip adjacent mesh faces that do not straddle `pl` beyond `r0` and only touch `pl`
            // at <= 1 vertex or along an already-shared Stage 1a mesh edge.
            let mut pos_count = 0usize;
            let mut neg_count = 0usize;
            on_plane_vids_buf.clear();
            let mut v_cursor = 0usize;
            for ring_3d in &other_nf.loops_3d {
                for &pt in ring_3d {
                    let vid = nf_all_vids[other_idx][v_cursor];
                    v_cursor += 1;
                    let h = pl.normal.dot(pt) - pl.d;
                    if h > r0 {
                        pos_count += 1;
                    } else if h < -r0 {
                        neg_count += 1;
                    } else if !on_plane_vids_buf.contains(&vid) {
                        on_plane_vids_buf.push(vid);
                    }
                }
            }
            let strictly_straddles = pos_count > 0 && neg_count > 0;
            if !strictly_straddles {
                if on_plane_vids_buf.len() <= 1 {
                    continue;
                }
                if on_plane_vids_buf.len() == 2 && r0 <= 1e-4 * diagonal {
                    let va = on_plane_vids_buf[0].min(on_plane_vids_buf[1]);
                    let vb = on_plane_vids_buf[0].max(on_plane_vids_buf[1]);
                    if (va as usize) < vert_adj.len() {
                        let already_shared_edge = vert_adj[va as usize].iter().any(|&e_id| {
                            let e = &edges[e_id];
                            e.alive && e.v0 == va && e.v1 == vb && e.planes.contains(&p_id)
                        });
                        if already_shared_edge {
                            continue;
                        }
                    }
                }
            }

            // Also verify at least one target face on `p_id` overlaps `other_nf`'s AABB
            // and has vertices straddling or touching `other_pl` within `near_tol`
            let near_tol = r0.max(1e-7);
            let any_target_active = on_plane_faces.iter().any(|&tf_idx| {
                let tf = &norm_faces[tf_idx];
                if !aabb_overlap_3d(
                    tf.min_pt,
                    tf.max_pt,
                    other_nf.min_pt,
                    other_nf.max_pt,
                    r0,
                ) {
                    return false;
                }
                let mut has_pos = false;
                let mut has_neg = false;
                for ring_3d in &tf.loops_3d {
                    for &pt in ring_3d {
                        let h = other_pl.normal.dot(pt) - other_pl.d;
                        if h >= -near_tol {
                            has_pos = true;
                        }
                        if h <= near_tol {
                            has_neg = true;
                        }
                        if has_pos && has_neg {
                            return true;
                        }
                    }
                }
                false
            });
            if !any_target_active {
                continue;
            }

            if inject_face_plane_chords_3d(
                other_nf,
                other_pl,
                pl,
                p_id,
                on_plane_faces,
                &norm_faces,
                r0,
                r_init,
                options.winding_rule,
                &mut verts,
                &mut edges,
                &mut vert_xy_buckets,
                &mut vert_adj,
                x_low,
                y_low,
                x_span_inv,
                y_span_inv,
            ) {
                plane_active_1b[p_id] = true;
                plane_active_1b[other_nf.plane_id] = true;
            }
        }
    }

    // Active-zone localization for Stage 2 (`consolidate_3d_complex`):
    // Prevent healing on sections of the input that are clean 2-manifold and uninvolved in the op.
    let mut vert_active: Vec<bool> = if r0 > 1e-4 * diagonal || body_count < 2 {
        vec![true; verts.len()]
    } else {
        let mut active = vec![false; verts.len()];
        for v_idx in num_stage1a_verts..verts.len() {
            active[v_idx] = true;
        }
        for (p_id, on_plane) in faces_on_plane.iter().enumerate() {
            let multi_body_or_unclosed = on_plane.len() > 1
                && (!all_bodies_closed
                    || on_plane
                        .iter()
                        .any(|&fi| norm_faces[fi].body_index != norm_faces[on_plane[0]].body_index));
            if plane_active_1b[p_id] || multi_body_or_unclosed {
                for &nf_idx in on_plane {
                    for &vid in &nf_all_vids[nf_idx] {
                        let r = resolve_vert_3d(&mut verts, vid) as usize;
                        active[r] = true;
                    }
                }
            }
        }
        for (nf_idx, &near_other) in face_near_other_vox.iter().enumerate() {
            if near_other {
                for &vid in &nf_all_vids[nf_idx] {
                    let r = resolve_vert_3d(&mut verts, vid) as usize;
                    active[r] = true;
                }
            }
        }
        let min_clean_len_sq = 4.0 * r_max * r_max;
        for (e_idx, e) in edges.iter().enumerate() {
            if !e.alive {
                continue;
            }
            let u = e.v0 as usize;
            let v = e.v1 as usize;
            if e_idx >= num_stage1a_edges {
                active[u] = true;
                active[v] = true;
                continue;
            }
            let is_clean_2manifold = if e.face_windings.len() == 2 {
                let (f0, s0) = e.face_windings[0];
                let (f1, s1) = e.face_windings[1];
                norm_faces[f0].body_index == norm_faces[f1].body_index
                    && s0 * norm_faces[f0].orient_sign + s1 * norm_faces[f1].orient_sign == 0
                    && verts[u].pos.sub(verts[v].pos).norm_sq() > min_clean_len_sq
            } else {
                false
            };
            if !is_clean_2manifold {
                active[u] = true;
                active[v] = true;
            }
        }
        // Expand active set by 1 hop along Stage 1a edges
        let seed_active = active.clone();
        for e in &edges[..num_stage1a_edges] {
            if !e.alive {
                continue;
            }
            let u = e.v0 as usize;
            let v = e.v1 as usize;
            if seed_active[u] || seed_active[v] {
                active[u] = true;
                active[v] = true;
            }
        }
        for (v_idx, &is_act) in active.iter().enumerate() {
            if !is_act {
                verts[v_idx].dirty_proj = false;
            }
        }
        active
    };

    // Stage 2: Coupled 3D 0-Stratum & 1-Stratum Fixpoint Consolidation
    consolidate_3d_complex(
        &mut verts,
        &mut edges,
        &planes,
        r0,
        r_init,
        r_max,
        &mut vert_active,
        &mut vert_xy_buckets,
        &mut vert_adj,
        x_low,
        y_low,
        x_span_inv,
        y_span_inv,
    );

    // Resolve Stage 1a outer vertex IDs after Stage 2 consolidation
    for ring in &mut nf_outer_vids {
        for vid in ring.iter_mut() {
            *vid = resolve_vert_3d(&mut verts, *vid);
        }
    }

    // Stage 3a: Planar 2D Face Patch Extraction (`∂₂`) on each canonical carrier plane
    let mut edges_by_plane: Vec<Vec<usize>> = vec![Vec::new(); planes.len()];
    for (e_id, e) in edges.iter().enumerate() {
        if !e.alive || e.v0 == e.v1 {
            continue;
        }
        for &p_id in &e.planes {
            edges_by_plane[p_id].push(e_id);
        }
    }

    let mut patches: Vec<FacePatch3D> = Vec::new();
    let mut fast_dir_edges: Vec<(usize, i32)> = Vec::with_capacity(8);
    for (p_id, pl) in planes.iter().enumerate() {
        let on_plane = &faces_on_plane[p_id];
        let plane_edges = &edges_by_plane[p_id];
        if on_plane.is_empty() || plane_edges.len() < 3 {
            continue;
        }

        // Passive Plane Fast Path:
        // If `p_id` has 1 simple face (or multiple simple faces of a single closed body on an
        // uninvolved plane `!plane_active_1b[p_id]`) and all boundary vertices and edges are unsplit
        // by Stage 1b / Stage 2, emit each `FacePatch3D` directly without 2D arrangement extraction.
        let can_try_passive = if on_plane.len() == 1 {
            let nf_idx = on_plane[0];
            let nf = &norm_faces[nf_idx];
            let k = nf_outer_vids[nf_idx].len();
            nf.loops_3d.len() == 1 && (k == 3 || k == 4) && plane_edges.len() == k && nf.area_2d > 1e-11
        } else {
            all_bodies_closed
                && !plane_active_1b[p_id]
                && r0 <= 1e-4 * diagonal
                && on_plane.iter().all(|&fi| {
                    let nf = &norm_faces[fi];
                    let k = nf_outer_vids[fi].len();
                    nf.body_index == norm_faces[on_plane[0]].body_index
                        && nf.loops_3d.len() == 1
                        && (k == 3 || k == 4)
                        && nf.area_2d > 1e-11
                })
                && {
                    let expected_he: usize = on_plane.iter().map(|&fi| nf_outer_vids[fi].len()).sum();
                    let mut actual_he = 0usize;
                    let mut all_from_plane = true;
                    for &e_id in plane_edges {
                        let c = edges[e_id]
                            .face_windings
                            .iter()
                            .filter(|&&(fi, _)| norm_faces[fi].plane_id == p_id)
                            .count();
                        if c == 0 {
                            all_from_plane = false;
                            break;
                        }
                        actual_he += c;
                    }
                    all_from_plane && actual_he == expected_he
                }
        };

        if can_try_passive && options.winding_rule.apply(1.0) {
            let start_patches_len = patches.len();
            let mut all_faces_ok = true;
            for &nf_idx in on_plane {
                let nf = &norm_faces[nf_idx];
                if nf.orient_sign == 0 {
                    all_faces_ok = false;
                    break;
                }
                let orig_vids = &nf_outer_vids[nf_idx];
                let k = orig_vids.len();
                let mut distinct_alive = true;
                for i in 0..k {
                    let vi = orig_vids[i] as usize;
                    if !verts[vi].alive {
                        distinct_alive = false;
                        break;
                    }
                    for j in (i + 1)..k {
                        if orig_vids[i] == orig_vids[j] {
                            distinct_alive = false;
                            break;
                        }
                    }
                    if !distinct_alive {
                        break;
                    }
                }
                if !distinct_alive {
                    all_faces_ok = false;
                    break;
                }

                fast_dir_edges.clear();
                for i in 0..k {
                    let u = orig_vids[i];
                    let v = orig_vids[(i + 1) % k];
                    let (a, b, sign) = if u < v { (u, v, 1) } else { (v, u, -1) };
                    let mut matched_eid = None;
                    for &e_id in plane_edges {
                        let e = &edges[e_id];
                        if e.v0 == a && e.v1 == b && e.face_windings.contains(&(nf_idx, sign)) {
                            matched_eid = Some(e_id);
                            break;
                        }
                    }
                    if let Some(e_id) = matched_eid {
                        fast_dir_edges.push((e_id, sign));
                    } else {
                        all_faces_ok = false;
                        break;
                    }
                }
                if !all_faces_ok {
                    break;
                }

                if k == 3 {
                    let p0 = pl.project_2d(verts[orig_vids[0] as usize].rep);
                    let p1 = pl.project_2d(verts[orig_vids[1] as usize].rep);
                    let p2 = pl.project_2d(verts[orig_vids[2] as usize].rep);
                    let area = 0.5 * cross_2d(p0, p1, p2);
                    if area > 1e-11 {
                        let sample_2d = Point2D {
                            x: (p0.x + p1.x + p2.x) * (1.0 / 3.0),
                            y: (p0.y + p1.y + p2.y) * (1.0 / 3.0),
                        };
                        let mut delta_w = vec![0i32; body_count];
                        delta_w[nf.body_index] = nf.orient_sign;
                        patches.push(FacePatch3D {
                            plane_id: p_id,
                            merged_planes: vec![p_id],
                            directed_edges: fast_dir_edges.clone(),
                            outer_vids: orig_vids.clone(),
                            holes_vids: Vec::new(),
                            sample_2d,
                            area_2d: area,
                            delta_w,
                            alive: true,
                        });
                    } else {
                        all_faces_ok = false;
                        break;
                    }
                } else {
                    let p0 = pl.project_2d(verts[orig_vids[0] as usize].rep);
                    let p1 = pl.project_2d(verts[orig_vids[1] as usize].rep);
                    let p2 = pl.project_2d(verts[orig_vids[2] as usize].rep);
                    let p3 = pl.project_2d(verts[orig_vids[3] as usize].rep);
                    let a012 = 0.5 * cross_2d(p0, p1, p2);
                    let a023 = 0.5 * cross_2d(p0, p2, p3);
                    let a013 = 0.5 * cross_2d(p0, p1, p3);
                    let a123 = 0.5 * cross_2d(p1, p2, p3);
                    let sample_and_area = if a012 > 1e-11 && a023 > 1e-11 {
                        let sample = if a012 >= a023 {
                            Point2D {
                                x: (p0.x + p1.x + p2.x) * (1.0 / 3.0),
                                y: (p0.y + p1.y + p2.y) * (1.0 / 3.0),
                            }
                        } else {
                            Point2D {
                                x: (p0.x + p2.x + p3.x) * (1.0 / 3.0),
                                y: (p0.y + p2.y + p3.y) * (1.0 / 3.0),
                            }
                        };
                        Some((sample, a012 + a023))
                    } else if a013 > 1e-11 && a123 > 1e-11 {
                        let sample = if a013 >= a123 {
                            Point2D {
                                x: (p0.x + p1.x + p3.x) * (1.0 / 3.0),
                                y: (p0.y + p1.y + p3.y) * (1.0 / 3.0),
                            }
                        } else {
                            Point2D {
                                x: (p1.x + p2.x + p3.x) * (1.0 / 3.0),
                                y: (p1.y + p2.y + p3.y) * (1.0 / 3.0),
                            }
                        };
                        Some((sample, a013 + a123))
                    } else {
                        None
                    };
                    if let Some((sample_2d, area)) = sample_and_area {
                        let mut delta_w = vec![0i32; body_count];
                        delta_w[nf.body_index] = nf.orient_sign;
                        patches.push(FacePatch3D {
                            plane_id: p_id,
                            merged_planes: vec![p_id],
                            directed_edges: fast_dir_edges.clone(),
                            outer_vids: orig_vids.clone(),
                            holes_vids: Vec::new(),
                            sample_2d,
                            area_2d: area,
                            delta_w,
                            alive: true,
                        });
                    } else {
                        all_faces_ok = false;
                        break;
                    }
                }
            }
            if all_faces_ok {
                continue;
            }
            patches.truncate(start_patches_len);
        }

        extract_plane_face_patches_3d(
            p_id,
            pl,
            plane_edges,
            on_plane,
            &norm_faces,
            &verts,
            &edges,
            body_count,
            options.winding_rule,
            &mut patches,
        );
    }

    if patches.is_empty() {
        return BRepResult3D::default();
    }

    // Stage 3b-0: 2-Stratum Coincident Patch Deduplication in O(P) via `min_edge_id` buckets
    let mut patches_by_min_edge: Vec<Vec<usize>> = vec![Vec::new(); edges.len()];
    for (i, patch) in patches.iter().enumerate() {
        if !patch.alive || patch.directed_edges.is_empty() {
            continue;
        }
        if let Some(min_e) = patch.directed_edges.iter().map(|&(e, _)| e).min() {
            patches_by_min_edge[min_e].push(i);
        }
    }
    for bucket in &patches_by_min_edge {
        if bucket.len() < 2 {
            continue;
        }
        for idx_a in 0..bucket.len() {
            let i = bucket[idx_a];
            if !patches[i].alive {
                continue;
            }
            for idx_b in (idx_a + 1)..bucket.len() {
                let j = bucket[idx_b];
                if !patches[j].alive
                    || patches[j].directed_edges.len() != patches[i].directed_edges.len()
                {
                    continue;
                }
                let (e0, s_i0) = patches[i].directed_edges[0];
                let Some(&(_, s_j0)) =
                    patches[j].directed_edges.iter().find(|&&(e, _)| e == e0)
                else {
                    continue;
                };
                let tau = s_i0 * s_j0;
                let same_edges = patches[i].directed_edges.iter().all(|&(e_id, s_i)| {
                    patches[j]
                        .directed_edges
                        .iter()
                        .any(|&(ej, s_j)| ej == e_id && s_i * s_j == tau)
                });
                if same_edges {
                    let j_planes = core::mem::take(&mut patches[j].merged_planes);
                    let j_dw = core::mem::take(&mut patches[j].delta_w);
                    merge_plane_tags(&mut patches[i].merged_planes, &j_planes);
                    for b in 0..body_count {
                        patches[i].delta_w[b] += tau * j_dw[b];
                    }
                    patches[j].alive = false;
                }
            }
            if patches[i].delta_w.iter().all(|&dw| dw == 0) {
                patches[i].alive = false;
            }
        }
    }

    // Stage 3b: Prune open degree-1 flaps (`∂₂` 2-core) & Radial CoFace Sorting (`∂₃`)
    loop {
        let mut edge_patch_deg: Vec<u32> = vec![0; edges.len()];
        for patch in &patches {
            if !patch.alive {
                continue;
            }
            for &(e_id, _) in &patch.directed_edges {
                edge_patch_deg[e_id] += 1;
            }
        }
        let mut pruned_any = false;
        for patch in &mut patches {
            if !patch.alive {
                continue;
            }
            if patch
                .directed_edges
                .iter()
                .any(|&(e_id, _)| edge_patch_deg[e_id] < 2)
            {
                patch.alive = false;
                pruned_any = true;
            }
        }
        if !pruned_any {
            break;
        }
    }

    let m_patches = patches.len();
    if m_patches == 0 {
        return BRepResult3D::default();
    }
    let mut uf = DisjointSet::new(2 * m_patches);
    let mut edge_spokes: Vec<Vec<(usize, i32)>> = vec![Vec::new(); edges.len()];
    for (f_idx, patch) in patches.iter().enumerate() {
        if !patch.alive {
            continue;
        }
        for &(e_id, sign) in &patch.directed_edges {
            edge_spokes[e_id].push((f_idx, sign));
        }
    }

    let patch_sample_3d: Vec<Point3D> = patches
        .iter()
        .map(|p| planes[p.plane_id].unproject_3d(p.sample_2d.x, p.sample_2d.y))
        .collect();

    // Lazily evaluated 3D solid-angle winding `(wb, wf)` per patch
    let mut patch_wb: Vec<Vec<f64>> = vec![vec![0.0; body_count]; m_patches];
    let mut patch_wf: Vec<Vec<f64>> = vec![vec![0.0; body_count]; m_patches];
    let mut patch_ib: Vec<Vec<i32>> = vec![vec![0; body_count]; m_patches];
    let mut patch_if: Vec<Vec<i32>> = vec![vec![0; body_count]; m_patches];
    let mut patch_evaluated: Vec<bool> = vec![false; m_patches];

    let eval_patch_winding = |f_idx: usize,
                                  patch_evaluated: &mut [bool],
                                  patch_wb: &mut [Vec<f64>],
                                  patch_wf: &mut [Vec<f64>],
                                  patch_ib: &mut [Vec<i32>],
                                  patch_if: &mut [Vec<i32>]| {
        if patch_evaluated[f_idx] {
            return;
        }
        patch_evaluated[f_idx] = true;
        let patch = &patches[f_idx];
        if !patch.alive {
            return;
        }
        let p_3d = patch_sample_3d[f_idx];
        for b in 0..body_count {
            let dw = patch.delta_w[b];
            if dw == 0
                && (p_3d.x < body_min[b].x - plane_dist_tol
                    || p_3d.x > body_max[b].x + plane_dist_tol
                    || p_3d.y < body_min[b].y - plane_dist_tol
                    || p_3d.y > body_max[b].y + plane_dist_tol
                    || p_3d.z < body_min[b].z - plane_dist_tol
                    || p_3d.z > body_max[b].z + plane_dist_tol)
            {
                continue;
            }
            let mut w_np = 0.0;
            for &nf_idx in &faces_by_body[b] {
                let other_nf = &norm_faces[nf_idx];
                if patch.merged_planes.contains(&other_nf.plane_id) {
                    continue;
                }
                let other_pl = &planes[other_nf.plane_id];
                w_np += face_solid_angle_turns(p_3d, other_nf, other_pl);
            }
            let dw_f = dw as f64;
            let wb = w_np + 0.5 * dw_f;
            let wf = w_np - 0.5 * dw_f;
            patch_wb[f_idx][b] = wb;
            patch_wf[f_idx][b] = wf;
            patch_ib[f_idx][b] = wb.round() as i32;
            patch_if[f_idx][b] = wf.round() as i32;
        }
    };

    for (e_id, spokes) in edge_spokes.iter_mut().enumerate() {
        let k = spokes.len();
        if k == 0 {
            continue;
        }
        if k == 1 {
            let (f_idx, _) = spokes[0];
            uf.union(2 * f_idx, 2 * f_idx + 1);
            continue;
        }
        // Fast path for manifold edges (`k == 2`): a 2-element cycle has a unique cyclic order,
        // so no frame construction, cross products, or `atan2` calls are needed!
        if k == 2 {
            let (f0, sig0) = spokes[0];
            let (f1, sig1) = spokes[1];
            let cf_ccw0 = if sig0 == 1 { 2 * f0 } else { 2 * f0 + 1 };
            let cf_cw0 = if sig0 == 1 { 2 * f0 + 1 } else { 2 * f0 };
            let cf_ccw1 = if sig1 == 1 { 2 * f1 } else { 2 * f1 + 1 };
            let cf_cw1 = if sig1 == 1 { 2 * f1 + 1 } else { 2 * f1 };
            uf.union(cf_ccw0, cf_cw1);
            uf.union(cf_ccw1, cf_cw0);
            continue;
        }

        let u = edges[e_id].v0 as usize;
        let v = edges[e_id].v1 as usize;
        let edge_vec = verts[v].rep.sub(verts[u].rep);
        let edge_len = edge_vec.norm();
        if edge_len <= 1e-15 {
            continue;
        }
        let t_vec = edge_vec.scale(1.0 / edge_len);
        let (e1, e2) = build_plane_frame(t_vec);

        let mut radial: Vec<(f64, usize, i32, Point3D)> = Vec::with_capacity(k);
        for &(f_idx, sigma) in spokes.iter() {
            let n_p = planes[patches[f_idx].plane_id].normal;
            // Inward sheet half-plane direction perpendicular to `t_vec`
            let b_dir = n_p.cross(t_vec).scale(sigma as f64);
            let b_unit = b_dir.normalized().unwrap_or(b_dir);
            let theta = b_dir.dot(e2).atan2(b_dir.dot(e1));
            radial.push((theta, f_idx, sigma, b_unit));
        }
        radial.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

        // Refine cyclic ordering of nearly-coplanar glancing sheets (`b_a · b_b > 0.99`)
        // when `e_id` is NOT the actual transverse intersection chord of `(π_a, π_b)`.
        let edge_mid = verts[u].rep.add(verts[v].rep).scale(0.5);
        for _pass in 0..k {
            let mut swapped = false;
            for idx in 0..k {
                let nxt = (idx + 1) % k;
                let (_, f_a, sig_a, b_a) = radial[idx];
                let (_, f_b, sig_b, b_b) = radial[nxt];
                if f_a == f_b || patches[f_a].plane_id == patches[f_b].plane_id {
                    continue;
                }
                let pl_a = &planes[patches[f_a].plane_id];
                let pl_b = &planes[patches[f_b].plane_id];
                let gap_at_edge =
                    ((pl_a.normal.dot(edge_mid) - pl_a.d) - (pl_b.normal.dot(edge_mid) - pl_b.d)).abs();
                if gap_at_edge > 1e-3 * r_max
                    && b_a.dot(b_b) > 0.99
                    && pl_a.normal.dot(pl_b.normal).abs() > 0.99
                {
                    let s_a = patch_sample_3d[f_a];
                    let s_b = patch_sample_3d[f_b];
                    let h_b_vs_a = (sig_a as f64) * (pl_a.normal.dot(s_b) - pl_a.d);
                    let h_a_vs_b = (sig_b as f64) * (pl_b.normal.dot(s_a) - pl_b.d);
                    let delta_h = h_b_vs_a - h_a_vs_b;
                    if delta_h < -1e-9 {
                        radial.swap(idx, nxt);
                        swapped = true;
                    }
                }
            }
            if !swapped {
                break;
            }
        }

        // Only evaluate 3D solid-angle winding on `k >= 3` edges when the radial spoke order
        // is geometrically ambiguous (coarse `r0`, open/non-manifold input, or collapsed edge).
        let mut ambiguous_spokes =
            !all_bodies_closed || r0 > 1e-4 * diagonal || edge_len <= 2.0 * r_max;
        if !ambiguous_spokes {
            for idx in 0..k {
                let nxt = (idx + 1) % k;
                if radial[idx].3.dot(radial[nxt].3) > 0.995 {
                    let pa = patches[radial[idx].1].plane_id;
                    let pb = patches[radial[nxt].1].plane_id;
                    let pl_a = &planes[pa];
                    let pl_b = &planes[pb];
                    let dev = (pl_a.normal.dot(edge_mid) - pl_a.d)
                        .abs()
                        .max((pl_b.normal.dot(edge_mid) - pl_b.d).abs());
                    if pa == pb || dev > 1e-3 * r_max {
                        ambiguous_spokes = true;
                        break;
                    }
                }
            }
        }

        if ambiguous_spokes {
            for &(_, f_idx, _, _) in &radial {
                eval_patch_winding(
                    f_idx,
                    &mut patch_evaluated,
                    &mut patch_wb,
                    &mut patch_wf,
                    &mut patch_ib,
                    &mut patch_if,
                );
            }

            let spoke_w_cw = |f_idx: usize, sig: i32| -> &[i32] {
                if sig == 1 {
                    &patch_ib[f_idx]
                } else {
                    &patch_if[f_idx]
                }
            };
            let spoke_w_ccw = |f_idx: usize, sig: i32| -> &[i32] {
                if sig == 1 {
                    &patch_if[f_idx]
                } else {
                    &patch_ib[f_idx]
                }
            };

            let all_spokes_quantized = radial.iter().all(|&(_, fi, _, _)| {
                (0..body_count).all(|b| {
                    (patch_wb[fi][b] - (patch_ib[fi][b] as f64)).abs() < 0.15
                        && (patch_wf[fi][b] - (patch_if[fi][b] as f64)).abs() < 0.15
                })
            });

            let count_mismatches = |order: &[(f64, usize, i32, Point3D)]| -> usize {
                let mut m = 0usize;
                for idx in 0..k {
                    let (_, f_cur, sig_cur, _) = order[idx];
                    let (_, f_nxt, sig_nxt, _) = order[(idx + 1) % k];
                    if spoke_w_ccw(f_cur, sig_cur) != spoke_w_cw(f_nxt, sig_nxt) {
                        m += 1;
                    }
                }
                m
            };

            let init_mismatches = if all_spokes_quantized {
                count_mismatches(&radial)
            } else {
                0
            };
            if init_mismatches > 0 && k <= 7 {
                let orig = radial.clone();
                let mut perm: Vec<usize> = (0..k).collect();
                let mut best_perm = perm.clone();
                let mut best_score = (init_mismatches, 0usize);

                fn search_cyclic_perms<'a>(
                    pos: usize,
                    k: usize,
                    perm: &mut Vec<usize>,
                    orig: &[(f64, usize, i32, Point3D)],
                    spoke_w_cw: &dyn Fn(usize, i32) -> &'a [i32],
                    spoke_w_ccw: &dyn Fn(usize, i32) -> &'a [i32],
                    best_perm: &mut Vec<usize>,
                    best_score: &mut (usize, usize),
                ) {
                    if pos == k {
                        let mut m = 0usize;
                        for idx in 0..k {
                            let (_, f_cur, sig_cur, _) = orig[perm[idx]];
                            let (_, f_nxt, sig_nxt, _) = orig[perm[(idx + 1) % k]];
                            if spoke_w_ccw(f_cur, sig_cur) != spoke_w_cw(f_nxt, sig_nxt) {
                                m += 1;
                            }
                        }
                        if m > best_score.0 {
                            return;
                        }
                        let mut inv = 0usize;
                        for a in 1..k {
                            for b in (a + 1)..k {
                                if perm[a] > perm[b] {
                                    inv += 1;
                                }
                            }
                        }
                        let score = (m, inv);
                        if score < *best_score {
                            *best_score = score;
                            *best_perm = perm.clone();
                        }
                        return;
                    }
                    for i in pos..k {
                        perm.swap(pos, i);
                        search_cyclic_perms(
                            pos + 1,
                            k,
                            perm,
                            orig,
                            spoke_w_cw,
                            spoke_w_ccw,
                            best_perm,
                            best_score,
                        );
                        perm.swap(pos, i);
                    }
                }

                search_cyclic_perms(
                    1,
                    k,
                    &mut perm,
                    &orig,
                    &spoke_w_cw,
                    &spoke_w_ccw,
                    &mut best_perm,
                    &mut best_score,
                );
                if best_score < (init_mismatches, 0) {
                    for idx in 0..k {
                        radial[idx] = orig[best_perm[idx]];
                    }
                }
            }
        }

        for idx in 0..k {
            let (_, f_cur, sig_cur, _) = radial[idx];
            let (_, f_nxt, sig_nxt, _) = radial[(idx + 1) % k];
            let cf_ccw = if sig_cur == 1 {
                2 * f_cur
            } else {
                2 * f_cur + 1
            };
            let cf_cw = if sig_nxt == 1 {
                2 * f_nxt + 1
            } else {
                2 * f_nxt
            };
            uf.union(cf_ccw, cf_cw);
        }
    }

    // Stage 4a: 3D Volume Cell Winding Classification (`S_v ∈ C₃`)
    let num_cf = 2 * m_patches;
    let mut patch_roots: Vec<(usize, usize)> = vec![(0, 0); m_patches];
    let mut cf_patches: Vec<Vec<(bool, f64, usize)>> = vec![Vec::new(); num_cf];
    for (f_idx, patch) in patches.iter().enumerate() {
        if !patch.alive {
            continue;
        }
        let r_front = uf.find(2 * f_idx);
        let r_back = uf.find(2 * f_idx + 1);
        patch_roots[f_idx] = (r_front, r_back);
        let w_weight = patch.area_2d.max(1e-12);
        let p_3d = patch_sample_3d[f_idx];
        // Prefer patches whose sample point lies outside other bodies' AABBs so solid angle skips them
        let mut outside_other = true;
        for b in 0..body_count {
            if patch.delta_w[b] == 0
                && p_3d.x >= body_min[b].x - plane_dist_tol
                && p_3d.x <= body_max[b].x + plane_dist_tol
                && p_3d.y >= body_min[b].y - plane_dist_tol
                && p_3d.y <= body_max[b].y + plane_dist_tol
                && p_3d.z >= body_min[b].z - plane_dist_tol
                && p_3d.z <= body_max[b].z + plane_dist_tol
            {
                outside_other = false;
                break;
            }
        }
        cf_patches[r_front].push((outside_other, w_weight, f_idx));
        if r_back != r_front {
            cf_patches[r_back].push((outside_other, w_weight, f_idx));
        }
    }

    // Sort candidate patches inside each CoFace root so `outside_other == true` comes first
    let mut cf_order: Vec<usize> = Vec::new();
    for (cf, list) in cf_patches.iter_mut().enumerate() {
        if list.is_empty() {
            continue;
        }
        list.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal))
        });
        cf_order.push(cf);
    }
    // Process CoFace roots that have an `outside_other == true` patch first so BFS seeds from exterior patches
    cf_order.sort_by(|&a, &b| cf_patches[b][0].0.cmp(&cf_patches[a][0].0));

    let allow_early_stop = all_bodies_closed && r0 <= 1e-4 * diagonal;
    let no_leaked_patches = allow_early_stop
        && patches
            .iter()
            .enumerate()
            .all(|(fi, p)| !p.alive || patch_roots[fi].0 != patch_roots[fi].1);

    let mut cf_has_int_w: Vec<bool> = vec![false; num_cf];
    let mut cf_int_w: Vec<Vec<i32>> = if no_leaked_patches {
        vec![vec![0i32; body_count]; num_cf]
    } else {
        Vec::new()
    };
    let mut bfs_consistent = no_leaked_patches;

    let propagate_coface_bfs = |seed_cf: usize,
                                    cf_has_int_w: &mut [bool],
                                    cf_int_w: &mut [Vec<i32>],
                                    bfs_consistent: &mut bool| {
        if !*bfs_consistent {
            return;
        }
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        queue.push_back(seed_cf);
        let mut next_w = vec![0i32; body_count];
        while let Some(curr_cf) = queue.pop_front() {
            for &(_, _, f_idx) in &cf_patches[curr_cf] {
                let patch = &patches[f_idx];
                if !patch.alive {
                    continue;
                }
                let (r_front, r_back) = patch_roots[f_idx];
                if curr_cf == r_front {
                    for b in 0..body_count {
                        next_w[b] = cf_int_w[r_front][b] + patch.delta_w[b];
                    }
                    if !cf_has_int_w[r_back] {
                        cf_int_w[r_back].copy_from_slice(&next_w);
                        cf_has_int_w[r_back] = true;
                        queue.push_back(r_back);
                    } else if cf_int_w[r_back] != next_w {
                        *bfs_consistent = false;
                        return;
                    }
                } else if curr_cf == r_back {
                    for b in 0..body_count {
                        next_w[b] = cf_int_w[r_back][b] - patch.delta_w[b];
                    }
                    if !cf_has_int_w[r_front] {
                        cf_int_w[r_front].copy_from_slice(&next_w);
                        cf_has_int_w[r_front] = true;
                        queue.push_back(r_front);
                    } else if cf_int_w[r_front] != next_w {
                        *bfs_consistent = false;
                        return;
                    }
                }
            }
        }
    };

    for &cf in &cf_order {
        if bfs_consistent && cf_has_int_w[cf] {
            continue;
        }
        let list = &cf_patches[cf];
        if allow_early_stop {
            if let Some(&(_, _, f_idx)) = list.iter().find(|&&(_, _, f_idx)| {
                patch_evaluated[f_idx]
                    && (0..body_count).all(|b| {
                        (patch_wb[f_idx][b] - (patch_ib[f_idx][b] as f64)).abs() < 0.02
                            && (patch_wf[f_idx][b] - (patch_if[f_idx][b] as f64)).abs() < 0.02
                    })
            }) {
                if bfs_consistent {
                    let (r_front, r_back) = patch_roots[f_idx];
                    if !cf_has_int_w[r_front] {
                        cf_int_w[r_front].copy_from_slice(&patch_if[f_idx]);
                        cf_has_int_w[r_front] = true;
                        propagate_coface_bfs(
                            r_front,
                            &mut cf_has_int_w,
                            &mut cf_int_w,
                            &mut bfs_consistent,
                        );
                    }
                    if bfs_consistent && !cf_has_int_w[r_back] {
                        cf_int_w[r_back].copy_from_slice(&patch_ib[f_idx]);
                        cf_has_int_w[r_back] = true;
                        propagate_coface_bfs(
                            r_back,
                            &mut cf_has_int_w,
                            &mut cf_int_w,
                            &mut bfs_consistent,
                        );
                    }
                    if bfs_consistent {
                        continue;
                    }
                } else {
                    continue;
                }
            }
        }
        for &(_, _, f_idx) in list.iter().take(8) {
            eval_patch_winding(
                f_idx,
                &mut patch_evaluated,
                &mut patch_wb,
                &mut patch_wf,
                &mut patch_ib,
                &mut patch_if,
            );
            if allow_early_stop {
                let cleanly_quantized = (0..body_count).all(|b| {
                    (patch_wb[f_idx][b] - (patch_ib[f_idx][b] as f64)).abs() < 0.02
                        && (patch_wf[f_idx][b] - (patch_if[f_idx][b] as f64)).abs() < 0.02
                });
                if cleanly_quantized {
                    if bfs_consistent {
                        let (r_front, r_back) = patch_roots[f_idx];
                        if !cf_has_int_w[r_front] {
                            cf_int_w[r_front].copy_from_slice(&patch_if[f_idx]);
                            cf_has_int_w[r_front] = true;
                            propagate_coface_bfs(
                                r_front,
                                &mut cf_has_int_w,
                                &mut cf_int_w,
                                &mut bfs_consistent,
                            );
                        }
                        if bfs_consistent && !cf_has_int_w[r_back] {
                            cf_int_w[r_back].copy_from_slice(&patch_ib[f_idx]);
                            cf_has_int_w[r_back] = true;
                            propagate_coface_bfs(
                                r_back,
                                &mut cf_has_int_w,
                                &mut cf_int_w,
                                &mut bfs_consistent,
                            );
                        }
                    }
                    break;
                }
            }
        }
    }

    // If BFS encountered an inconsistent cycle, ensure every CoFace root has a directly evaluated patch
    if no_leaked_patches && !bfs_consistent {
        for &cf in &cf_order {
            let list = &cf_patches[cf];
            let has_eval = list.iter().any(|&(_, _, fi)| patch_evaluated[fi]);
            if !has_eval {
                for &(_, _, f_idx) in list.iter().take(8) {
                    eval_patch_winding(
                        f_idx,
                        &mut patch_evaluated,
                        &mut patch_wb,
                        &mut patch_wf,
                        &mut patch_ib,
                        &mut patch_if,
                    );
                    let cleanly_quantized = (0..body_count).all(|b| {
                        (patch_wb[f_idx][b] - (patch_ib[f_idx][b] as f64)).abs() < 0.02
                            && (patch_wf[f_idx][b] - (patch_if[f_idx][b] as f64)).abs() < 0.02
                    });
                    if cleanly_quantized {
                        break;
                    }
                }
            }
        }
    }

    let mut vol_w_sum: Vec<Vec<f64>> = vec![vec![0.0; body_count]; num_cf];
    let mut vol_area_sum: Vec<f64> = vec![0.0; num_cf];

    for (f_idx, patch) in patches.iter().enumerate() {
        if !patch.alive || !patch_evaluated[f_idx] {
            continue;
        }
        let r_front = uf.find(2 * f_idx);
        let r_back = uf.find(2 * f_idx + 1);
        let w_weight = patch.area_2d.max(1e-12);
        vol_area_sum[r_front] += w_weight;
        vol_area_sum[r_back] += w_weight;

        for b in 0..body_count {
            vol_w_sum[r_back][b] += w_weight * patch_wb[f_idx][b];
            vol_w_sum[r_front][b] += w_weight * patch_wf[f_idx][b];
        }
    }

    let mut vol_keep: Vec<bool> = vec![false; num_cf];
    let mut w_eval = vec![0.0f64; body_count];
    for cf in 0..num_cf {
        if uf.find(cf) != cf {
            continue;
        }
        if bfs_consistent && cf_has_int_w[cf] {
            for b in 0..body_count {
                w_eval[b] = cf_int_w[cf][b] as f64;
            }
            vol_keep[cf] = predicate(&w_eval);
            continue;
        }
        if vol_area_sum[cf] <= 0.0 {
            continue;
        }
        let inv_a = 1.0 / vol_area_sum[cf];
        for b in 0..body_count {
            w_eval[b] = (vol_w_sum[cf][b] * inv_a).round();
        }
        vol_keep[cf] = predicate(&w_eval);
    }

    // Stage 4b: Algebraic Boundary Extraction (`b₂ = ∂₃ · c₃`) & Coplanar Seam Cancellation
    // Group surviving patches by `(plane_id, orient_sign)`
    let mut plane_pos_patches: Vec<Vec<usize>> = vec![Vec::new(); planes.len()];
    let mut plane_neg_patches: Vec<Vec<usize>> = vec![Vec::new(); planes.len()];

    for (f_idx, patch) in patches.iter().enumerate() {
        if !patch.alive {
            continue;
        }
        let keep_front = vol_keep[uf.find(2 * f_idx)];
        let keep_back = vol_keep[uf.find(2 * f_idx + 1)];
        if keep_back && !keep_front {
            plane_pos_patches[patch.plane_id].push(f_idx);
        } else if keep_front && !keep_back {
            plane_neg_patches[patch.plane_id].push(f_idx);
        }
    }

    let mut unified_faces_vids: Vec<(usize, i32, Vec<u32>, Vec<Vec<u32>>)> = Vec::new();
    for p_id in 0..planes.len() {
        for &(orient_sign, ref p_list) in &[
            (1i32, &plane_pos_patches[p_id]),
            (-1i32, &plane_neg_patches[p_id]),
        ] {
            if p_list.is_empty() {
                continue;
            }
            // Fast path when a carrier plane has a single surviving patch with no internal slit edges:
            // emit its `outer_vids` and `holes_vids` directly in O(1) without re-running planar arrangement!
            if p_list.len() == 1 {
                let patch = &patches[p_list[0]];
                if patch.outer_vids.len() >= 3 {
                    let de = &patch.directed_edges;
                    let mut has_duplicate_edge = false;
                    for i in 0..de.len() {
                        for j in (i + 1)..de.len() {
                            if de[i].0 == de[j].0 {
                                has_duplicate_edge = true;
                                break;
                            }
                        }
                        if has_duplicate_edge {
                            break;
                        }
                    }
                    if !has_duplicate_edge {
                        let mut outer_vids = patch.outer_vids.clone();
                        let mut holes_vids = patch.holes_vids.clone();
                        if orient_sign < 0 {
                            outer_vids.reverse();
                            for h in &mut holes_vids {
                                h.reverse();
                            }
                        }
                        unified_faces_vids.push((p_id, orient_sign, outer_vids, holes_vids));
                        continue;
                    }
                }
            }
            unify_coplanar_patches_3d(
                p_id,
                orient_sign,
                p_list,
                &patches,
                &edges,
                &verts,
                &planes[p_id],
                &mut unified_faces_vids,
            );
        }
    }

    // Stage 4c: Global 3D Degree-2 Collinear Vertex Simplification
    if options.simplify_collinear {
        simplify_global_collinear_vertices_3d(&mut unified_faces_vids, &verts, r0);
    }

    // Stage 4d: Option B — Strict Planarity Split for Warped Faces & Final BRepResult3D Assembly
    assemble_brep_result_3d(unified_faces_vids, &verts, &planes, center, diagonal)
}

/// Unify adjacent surviving patches on plane `p_id` with orientation `orient_sign ∈ {+1, -1}`
/// by cancelling shared internal coplanar half-edges `(u -> v)` and `(v -> u)`.
fn unify_coplanar_patches_3d(
    p_id: usize,
    orient_sign: i32,
    patch_indices: &[usize],
    patches: &[FacePatch3D],
    edges: &[Edge3D],
    verts: &[Vert3D],
    pl: &CanonicalPlane,
    out_faces: &mut Vec<(usize, i32, Vec<u32>, Vec<Vec<u32>>)>,
) {
    // Net directed edge count in canonical `+N_c` orientation (`v0 -> v1` is `+1`, `v1 -> v0` is `-1`)
    let mut net_edges: Vec<(usize, i32)> = Vec::new();
    for &f_idx in patch_indices {
        for &(e_id, sign) in &patches[f_idx].directed_edges {
            if let Some(pos) = net_edges.iter().position(|&(id, _)| id == e_id) {
                net_edges[pos].1 += sign;
            } else {
                net_edges.push((e_id, sign));
            }
        }
    }

    // Build surviving directed half-edges `(orig_vid, dest_vid, e_id)` in `+N_c` CCW frame
    let mut dir_hes: Vec<(u32, u32)> = Vec::new();
    for (e_id, net) in net_edges {
        if net == 0 {
            continue;
        }
        let u = edges[e_id].v0;
        let v = edges[e_id].v1;
        if net > 0 {
            for _ in 0..net {
                dir_hes.push((u, v));
            }
        } else {
            for _ in 0..(-net) {
                dir_hes.push((v, u));
            }
        }
    }
    if dir_hes.len() < 3 {
        return;
    }

    let mut local_vids: Vec<u32> = Vec::with_capacity(dir_hes.len() * 2);
    for &(u, v) in &dir_hes {
        local_vids.push(u);
        local_vids.push(v);
    }
    local_vids.sort_unstable();
    local_vids.dedup();
    let num_lv = local_vids.len();
    if num_lv < 3 {
        return;
    }
    let to_lv = |vid: u32| -> usize { local_vids.binary_search(&vid).unwrap() };

    let v_2d: Vec<Point2D> = local_vids
        .iter()
        .map(|&vid| pl.project_2d(verts[vid as usize].rep))
        .collect();
    let num_he = dir_hes.len();
    let mut vert_in: Vec<Vec<usize>> = vec![Vec::new(); num_lv];
    let mut vert_out: Vec<Vec<usize>> = vec![Vec::new(); num_lv];
    for (h_idx, &(u, v)) in dir_hes.iter().enumerate() {
        vert_out[to_lv(u)].push(h_idx);
        vert_in[to_lv(v)].push(h_idx);
    }

    // At each vertex `w`, pair incoming half-edge `u -> w` with the outgoing half-edge `w -> v`
    // that makes the tightest interior CCW turn (prev spoke in CCW order around `w`).
    let mut he_next: Vec<usize> = vec![usize::MAX; num_he];
    for lw in 0..num_lv {
        if vert_in[lw].is_empty() || vert_out[lw].is_empty() {
            continue;
        }
        if vert_in[lw].len() == 1 && vert_out[lw].len() == 1 {
            he_next[vert_in[lw][0]] = vert_out[lw][0];
            continue;
        }
        let pw = v_2d[lw];
        let mut spokes: Vec<(f64, bool, usize)> =
            Vec::with_capacity(vert_in[lw].len() + vert_out[lw].len());
        for &h_in in &vert_in[lw] {
            let pu = v_2d[to_lv(dir_hes[h_in].0)];
            let ang = (pu.y - pw.y).atan2(pu.x - pw.x);
            spokes.push((ang, false, h_in));
        }
        for &h_out in &vert_out[lw] {
            let pv = v_2d[to_lv(dir_hes[h_out].1)];
            let ang = (pv.y - pw.y).atan2(pv.x - pw.x);
            spokes.push((ang, true, h_out));
        }
        spokes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

        let m = spokes.len();
        let mut used_out = vec![false; num_he];
        for idx in 0..m {
            let (_, is_out, h_in) = spokes[idx];
            if is_out {
                continue;
            }
            // Walk CW (decreasing angle) from incoming spoke `u <- w` to find the first unused outgoing spoke
            for step in 1..=m {
                let (_, cand_out, h_cand) = spokes[(idx + m - step) % m];
                if cand_out && !used_out[h_cand] {
                    used_out[h_cand] = true;
                    he_next[h_in] = h_cand;
                    break;
                }
            }
        }
    }

    let mut visited = vec![false; num_he];
    let mut shells: Vec<(Vec<u32>, f64)> = Vec::new();
    let mut holes: Vec<(Vec<u32>, f64)> = Vec::new();

    for h_start in 0..num_he {
        if visited[h_start] || he_next[h_start] == usize::MAX {
            continue;
        }
        let mut raw_vids: Vec<u32> = Vec::new();
        let mut cur = h_start;
        while cur != usize::MAX && !visited[cur] {
            visited[cur] = true;
            raw_vids.push(dir_hes[cur].0);
            cur = he_next[cur];
        }
        if raw_vids.len() < 3 {
            continue;
        }

        // Decompose pinched figure-8 loops at repeated `VertId`s into simple loops
        let mut v_stack: Vec<u32> = Vec::with_capacity(raw_vids.len());
        let mut simple_rings: Vec<Vec<u32>> = Vec::new();
        for vid in raw_vids {
            if let Some(pos) = v_stack.iter().position(|&sv| sv == vid) {
                if v_stack.len() - pos >= 3 {
                    simple_rings.push(v_stack[pos..].to_vec());
                }
                v_stack.truncate(pos);
            }
            v_stack.push(vid);
        }
        if v_stack.len() >= 3 {
            simple_rings.push(v_stack);
        }

        for ring in simple_rings {
            let pts_2d: Vec<Point2D> = ring.iter().map(|&vid| v_2d[to_lv(vid)]).collect();
            let area = signed_area_2d_pts(&pts_2d);
            if area > 1e-11 {
                shells.push((ring, area));
            } else if area < -1e-11 {
                holes.push((ring, area.abs()));
            }
        }
    }

    if shells.is_empty() {
        return;
    }

    let mut poly_holes: Vec<Vec<Vec<u32>>> = vec![Vec::new(); shells.len()];
    if !holes.is_empty() {
        if shells.len() == 1 {
            for (h_ring, _) in holes {
                poly_holes[0].push(h_ring);
            }
        } else {
            let shells_2d: Vec<Vec<Point2D>> = shells
                .iter()
                .map(|(sh, _)| sh.iter().map(|&vid| v_2d[to_lv(vid)]).collect())
                .collect();
            for (h_ring, h_area) in holes {
                let test_pt = h_ring
                    .iter()
                    .map(|&vid| v_2d[to_lv(vid)])
                    .find(|hp| {
                        shells_2d.iter().all(|sh| {
                            sh.iter()
                                .all(|sp| (sp.x - hp.x).hypot(sp.y - hp.y) > 1e-8)
                        })
                    })
                    .unwrap_or_else(|| v_2d[to_lv(h_ring[0])]);
                let mut best_s = 0usize;
                let mut best_area = f64::INFINITY;
                for (s_idx, (_, s_area)) in shells.iter().enumerate() {
                    if *s_area <= h_area + 1e-11 {
                        continue;
                    }
                    if loops_winding_2d(test_pt, core::slice::from_ref(&shells_2d[s_idx])) != 0
                        && *s_area < best_area
                    {
                        best_area = *s_area;
                        best_s = s_idx;
                    }
                }
                poly_holes[best_s].push(h_ring);
            }
        }
    }

    for (s_idx, (mut outer_vids, _)) in shells.into_iter().enumerate() {
        let mut h_list = core::mem::take(&mut poly_holes[s_idx]);
        if orient_sign < 0 {
            outer_vids.reverse();
            for h in &mut h_list {
                h.reverse();
            }
        }
        out_faces.push((p_id, orient_sign, outer_vids, h_list));
    }
}

/// Stage 4c: Simplify collinear vertices that have global 3D edge degree == 2 across the entire solid.
fn simplify_global_collinear_vertices_3d(
    faces: &mut [(usize, i32, Vec<u32>, Vec<Vec<u32>>)],
    verts: &[Vert3D],
    r0: f64,
) {
    let mut can_prune = vec![false; verts.len()];
    let mut any_gt3 = false;
    for (_, _, outer, holes) in faces.iter() {
        if outer.len() > 3 {
            any_gt3 = true;
            for &vid in outer {
                can_prune[vid as usize] = true;
            }
        }
        for h in holes {
            if h.len() > 3 {
                any_gt3 = true;
                for &vid in h {
                    can_prune[vid as usize] = true;
                }
            }
        }
    }
    if !any_gt3 {
        return;
    }

    let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); verts.len()];
    let mut add_neighbor = |u: u32, v: u32| {
        if u == v {
            return;
        }
        if can_prune[u as usize] && !neighbors[u as usize].contains(&v) {
            neighbors[u as usize].push(v);
        }
        if can_prune[v as usize] && !neighbors[v as usize].contains(&u) {
            neighbors[v as usize].push(u);
        }
    };

    for (_, _, outer, holes) in faces.iter() {
        let n = outer.len();
        for i in 0..n {
            add_neighbor(outer[i], outer[(i + 1) % n]);
        }
        for h in holes {
            let hn = h.len();
            for i in 0..hn {
                add_neighbor(h[i], h[(i + 1) % hn]);
            }
        }
    }

    let mut removable = vec![false; verts.len()];
    let mut any_removable = false;
    let sin_tol_sq = 1e-6;
    let dist_tol_sq = (0.25 * r0).max(1e-9).powi(2);
    for v_id in 0..verts.len() {
        if can_prune[v_id] && neighbors[v_id].len() == 2 {
            let u = neighbors[v_id][0] as usize;
            let w = neighbors[v_id][1] as usize;
            let pu = verts[u].rep;
            let pv = verts[v_id].rep;
            let pw = verts[w].rep;
            let d1 = pv.sub(pu);
            let d2 = pw.sub(pv);
            let l1_sq = d1.norm_sq();
            let l2_sq = d2.norm_sq();
            if l1_sq <= 1e-20 || l2_sq <= 1e-20 {
                removable[v_id] = true;
                any_removable = true;
                continue;
            }
            if d1.dot(d2) > 0.0 {
                let cr_sq = d1.cross(d2).norm_sq();
                let uw = pw.sub(pu);
                let uw_sq = uw.norm_sq();
                let perp_sq = if uw_sq > 1e-20 {
                    cr_sq / uw_sq
                } else {
                    0.0
                };
                if cr_sq <= sin_tol_sq * l1_sq * l2_sq && perp_sq <= dist_tol_sq {
                    removable[v_id] = true;
                    any_removable = true;
                }
            }
        }
    }
    if !any_removable {
        return;
    }

    let prune_ring = |ring: &mut Vec<u32>| {
        if ring.len() <= 3 {
            return;
        }
        let mut i = 0;
        while i < ring.len() && ring.len() > 3 {
            let vid = ring[i] as usize;
            if removable[vid] {
                ring.remove(i);
            } else {
                i += 1;
            }
        }
    };

    for (_, _, outer, holes) in faces.iter_mut() {
        prune_ring(outer);
        for h in holes.iter_mut() {
            prune_ring(h);
        }
    }
}

/// Stage 4d: Assemble `BRepResult3D`, applying Option B (strict planarity split along boundary-vertex
/// triangulation diagonals) to any face warped beyond planarity tolerance at high `r0`.
fn assemble_brep_result_3d(
    unified_faces_vids: Vec<(usize, i32, Vec<u32>, Vec<Vec<u32>>)>,
    verts: &[Vert3D],
    planes: &[CanonicalPlane],
    center: Point3D,
    diagonal: f64,
) -> BRepResult3D {
    let planar_tol = 1e-5 * diagonal;
    let mut out_faces: Vec<Face3D> = Vec::with_capacity(unified_faces_vids.len());
    let mut vert_dir_edges: Vec<Vec<(u32, i32)>> = vec![Vec::new(); verts.len()];
    let mut used_vert = vec![false; verts.len()];

    let mut record_half_edge = |u: u32, v: u32| {
        if u == v {
            return;
        }
        used_vert[u as usize] = true;
        used_vert[v as usize] = true;
        let (a, b, sign) = if u < v { (u, v, 1) } else { (v, u, -1) };
        let list = &mut vert_dir_edges[a as usize];
        for entry in list.iter_mut() {
            if entry.0 == b {
                entry.1 += sign;
                return;
            }
        }
        list.push((b, sign));
    };

    for (p_id, orient_sign, outer_vids, holes_vids) in unified_faces_vids {
        if outer_vids.len() < 3 {
            continue;
        }

        let carrier_n = if orient_sign >= 0 {
            planes[p_id].normal
        } else {
            planes[p_id].normal.scale(-1.0)
        };

        if outer_vids.len() == 3 && holes_vids.is_empty() {
            let v0 = outer_vids[0];
            let v1 = outer_vids[1];
            let v2 = outer_vids[2];
            record_half_edge(v0, v1);
            record_half_edge(v1, v2);
            record_half_edge(v2, v0);
            let a = verts[v0 as usize].rep.add(center);
            let b = verts[v1 as usize].rep.add(center);
            let c = verts[v2 as usize].rep.add(center);
            let mut face_n = b.sub(a).cross(c.sub(a)).normalized().unwrap_or(carrier_n);
            if face_n.dot(carrier_n) < 0.0 {
                face_n = face_n.scale(-1.0);
            }
            out_faces.push(Face3D {
                outer: vec![a, b, c],
                holes: Vec::new(),
                normal: face_n,
            });
            continue;
        }

        let outer_3d: Vec<Point3D> = outer_vids
            .iter()
            .map(|&vid| verts[vid as usize].rep.add(center))
            .collect();
        let holes_3d: Vec<Vec<Point3D>> = holes_vids
            .iter()
            .map(|h| {
                h.iter()
                    .map(|&vid| verts[vid as usize].rep.add(center))
                    .collect()
            })
            .collect();

        let mut face_n = compute_newell_normal(&outer_3d).unwrap_or(carrier_n);
        if face_n.dot(carrier_n) < 0.0 {
            face_n = face_n.scale(-1.0);
        }

        // Check maximum out-of-plane vertex deviation for Option B
        let p0 = outer_3d[0];
        let mut max_dev = 0.0f64;
        for &pt in &outer_3d {
            max_dev = max_dev.max(face_n.dot(pt.sub(p0)).abs());
        }
        for h in &holes_3d {
            for &pt in h {
                max_dev = max_dev.max(face_n.dot(pt.sub(p0)).abs());
            }
        }

        if (outer_vids.len() > 3 || !holes_vids.is_empty()) && max_dev > planar_tol {
            // Option B: Split non-planar face along triangulation diagonals using only boundary vertex IDs
            let tris_vids =
                triangulate_face_vids_3d(&outer_vids, &holes_vids, &outer_3d, &holes_3d, face_n);
            for [va, vb, vc] in tris_vids {
                record_half_edge(va, vb);
                record_half_edge(vb, vc);
                record_half_edge(vc, va);
                let a = verts[va as usize].rep.add(center);
                let b = verts[vb as usize].rep.add(center);
                let c = verts[vc as usize].rep.add(center);
                let tn = compute_newell_normal(&[a, b, c]).unwrap_or(face_n);
                out_faces.push(Face3D {
                    outer: vec![a, b, c],
                    holes: Vec::new(),
                    normal: tn,
                });
            }
        } else {
            let n_out = outer_vids.len();
            for i in 0..n_out {
                record_half_edge(outer_vids[i], outer_vids[(i + 1) % n_out]);
            }
            for h in &holes_vids {
                let hn = h.len();
                for i in 0..hn {
                    record_half_edge(h[i], h[(i + 1) % hn]);
                }
            }
            out_faces.push(Face3D {
                outer: outer_3d,
                holes: holes_3d,
                normal: face_n,
            });
        }
    }

    let vertex_count = used_vert.iter().filter(|&&u| u).count();
    let mut edge_count = 0usize;
    let mut unmatched_half_edge_count = 0usize;
    for list in &vert_dir_edges {
        edge_count += list.len();
        for &(_, net) in list {
            unmatched_half_edge_count += net.unsigned_abs() as usize;
        }
    }

    let solid = Solid3D { faces: out_faces };
    let total_volume = solid.volume();
    let total_surface_area = solid.surface_area();

    BRepResult3D {
        solid,
        total_volume,
        total_surface_area,
        vertex_count,
        edge_count,
        unmatched_half_edge_count,
    }
}

/// Compute the 3D Boolean union (`A ∪ B`) of two planar B-Rep solids.
pub fn union_3d(a: &Solid3D, b: &Solid3D, options: KernelOptions) -> BRepResult3D {
    let rule = options.winding_rule;
    execute_brep_3d(
        &[a, b],
        |w| rule.apply(w[0]) || rule.apply(w[1]),
        options,
    )
}

/// Compute the 3D Boolean intersection (`A ∩ B`) of two planar B-Rep solids.
pub fn intersect_3d(a: &Solid3D, b: &Solid3D, options: KernelOptions) -> BRepResult3D {
    let rule = options.winding_rule;
    execute_brep_3d(
        &[a, b],
        |w| rule.apply(w[0]) && rule.apply(w[1]),
        options,
    )
}

/// Compute the 3D Boolean difference (`A \ B`) of two planar B-Rep solids.
pub fn difference_3d(a: &Solid3D, b: &Solid3D, options: KernelOptions) -> BRepResult3D {
    let rule = options.winding_rule;
    execute_brep_3d(
        &[a, b],
        |w| rule.apply(w[0]) && !rule.apply(w[1]),
        options,
    )
}

/// Compute the 3D Boolean symmetric difference (`A ⊕ B`) of two planar B-Rep solids.
pub fn xor_3d(a: &Solid3D, b: &Solid3D, options: KernelOptions) -> BRepResult3D {
    let rule = options.winding_rule;
    execute_brep_3d(
        &[a, b],
        |w| rule.apply(w[0]) != rule.apply(w[1]),
        options,
    )
}

/// Heal a single 3D planar face soup / self-intersecting polyhedron into a watertight 2-manifold B-Rep.
pub fn heal_3d(solid: &Solid3D, options: KernelOptions) -> BRepResult3D {
    let rule = options.winding_rule;
    execute_brep_3d(&[solid], |w| rule.apply(w[0]), options)
}

/// Dispatch a 3D Boolean operation by [`BooleanOp`].
pub fn clip_3d(op: BooleanOp, a: &Solid3D, b: &Solid3D, options: KernelOptions) -> BRepResult3D {
    match op {
        BooleanOp::Union => union_3d(a, b, options),
        BooleanOp::Intersect => intersect_3d(a, b, options),
        BooleanOp::Difference => difference_3d(a, b, options),
        BooleanOp::Xor => xor_3d(a, b, options),
    }
}

/// Compute the 3D Boolean union (`A ∪ B`) of two planar B-Rep solids.
#[inline]
pub fn union(a: &Solid3D, b: &Solid3D, options: KernelOptions) -> BRepResult3D {
    union_3d(a, b, options)
}

/// Compute the 3D Boolean intersection (`A ∩ B`) of two planar B-Rep solids.
#[inline]
pub fn intersect(a: &Solid3D, b: &Solid3D, options: KernelOptions) -> BRepResult3D {
    intersect_3d(a, b, options)
}

/// Compute the 3D Boolean difference (`A \ B`) of two planar B-Rep solids.
#[inline]
pub fn difference(a: &Solid3D, b: &Solid3D, options: KernelOptions) -> BRepResult3D {
    difference_3d(a, b, options)
}

/// Compute the 3D Boolean symmetric difference (`A ⊕ B`) of two planar B-Rep solids.
#[inline]
pub fn xor(a: &Solid3D, b: &Solid3D, options: KernelOptions) -> BRepResult3D {
    xor_3d(a, b, options)
}

/// Heal a single 3D planar face soup / self-intersecting polyhedron into a watertight 2-manifold B-Rep.
#[inline]
pub fn heal(solid: &Solid3D, options: KernelOptions) -> BRepResult3D {
    heal_3d(solid, options)
}

/// Dispatch a 3D Boolean operation by [`BooleanOp`].
#[inline]
pub fn clip(op: BooleanOp, a: &Solid3D, b: &Solid3D, options: KernelOptions) -> BRepResult3D {
    clip_3d(op, a, b, options)
}

/// Triangulate a 3D planar face (`outer` + `holes`) via 2D ear-clipping on its carrier plane.
#[inline]
pub fn triangulate_face(face: &Face3D) -> Vec<[Point3D; 3]> {
    triangulate_face_3d(face)
}

/// Execute a custom N-body 3D Boolean predicate over the consolidated 3D cellular complex.
#[inline]
pub fn execute_brep<F>(
    solids: &[&Solid3D],
    keep_predicate: F,
    options: KernelOptions,
) -> BRepResult3D
where
    F: Fn(&[f64]) -> bool,
{
    execute_brep_3d(solids, keep_predicate, options)
}

// ============================================================================
// WebAssembly C-ABI Exports for 3D Planar B-Rep Operations
// ============================================================================

struct WasmBRep3DBuffer {
    /// Flat input buffer:
    /// Encodes a sequence of solids/faces/loops as `f64` values:
    /// `[num_solids, for each solid: num_faces, for each face: nx, ny, nz, num_loops, for each loop: num_pts, x0, y0, z0, ...]`
    input_data: Vec<f64>,
    parsed_solids: Vec<Solid3D>,
    solids_dirty: bool,
    /// Output flat 3D coordinates `[x0, y0, z0, x1, y1, z1, ...]`
    out_coords: Vec<f64>,
    /// Output loop metadata: 4 u32s per loop: `[face_id, is_hole (0 or 1), start_pt_idx, pt_count]`
    out_loop_meta: Vec<u32>,
    /// Output face normals: 3 f64s per face: `[nx, ny, nz]`
    out_face_normals: Vec<f64>,
    /// Output triangulated mesh vertices for fast WebGL/Canvas rendering: `[ax, ay, az, bx, by, bz, cx, cy, cz, nx, ny, nz, face_id, ...]` (13 f64s per triangle)
    out_triangles: Vec<f64>,
    out_volume: f64,
    out_area: f64,
    out_vertex_count: u32,
    out_edge_count: u32,
    out_unmatched_he_count: u32,
}

impl WasmBRep3DBuffer {
    fn new() -> Self {
        Self {
            input_data: Vec::new(),
            parsed_solids: Vec::new(),
            solids_dirty: true,
            out_coords: Vec::new(),
            out_loop_meta: Vec::new(),
            out_face_normals: Vec::new(),
            out_triangles: Vec::new(),
            out_volume: 0.0,
            out_area: 0.0,
            out_vertex_count: 0,
            out_edge_count: 0,
            out_unmatched_he_count: 0,
        }
    }

    fn ensure_parsed(&mut self) {
        if !self.solids_dirty {
            return;
        }
        self.solids_dirty = false;
        self.parsed_solids.clear();
        let data = &self.input_data;
        if data.is_empty() {
            return;
        }
        let mut cursor = 0usize;
        let num_solids = data[cursor] as usize;
        cursor += 1;

        self.parsed_solids.reserve(num_solids);
        for _ in 0..num_solids {
            if cursor >= data.len() {
                break;
            }
            let num_faces = data[cursor] as usize;
            cursor += 1;
            let mut faces = Vec::with_capacity(num_faces);
            for _ in 0..num_faces {
                if cursor + 4 > data.len() {
                    break;
                }
                let nx = data[cursor];
                let ny = data[cursor + 1];
                let nz = data[cursor + 2];
                let num_loops = data[cursor + 3] as usize;
                cursor += 4;

                let mut outer = Vec::new();
                let mut holes = Vec::new();
                for l_idx in 0..num_loops {
                    if cursor >= data.len() {
                        break;
                    }
                    let num_pts = data[cursor] as usize;
                    cursor += 1;
                    let mut ring = Vec::with_capacity(num_pts);
                    for _ in 0..num_pts {
                        if cursor + 3 > data.len() {
                            break;
                        }
                        ring.push(Point3D::new(
                            data[cursor],
                            data[cursor + 1],
                            data[cursor + 2],
                        ));
                        cursor += 3;
                    }
                    if l_idx == 0 {
                        outer = ring;
                    } else {
                        holes.push(ring);
                    }
                }
                if outer.len() >= 3 {
                    faces.push(Face3D {
                        outer,
                        holes,
                        normal: Point3D::new(nx, ny, nz),
                    });
                }
            }
            self.parsed_solids.push(Solid3D { faces });
        }
    }
}

std::thread_local! {
    static WASM_BREP3D_BUF: std::cell::RefCell<WasmBRep3DBuffer> =
        std::cell::RefCell::new(WasmBRep3DBuffer::new());
}

#[no_mangle]
pub extern "C" fn wasm_brep3d_alloc_input(f64_count: usize) -> *mut f64 {
    WASM_BREP3D_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.input_data.clear();
        buf.input_data.resize(f64_count, 0.0);
        buf.solids_dirty = true;
        buf.input_data.as_mut_ptr()
    })
}

/// Op codes:
/// 0 = Union (A ∪ B)
/// 1 = Intersect (A ∩ B)
/// 2 = Difference (A \ B)
/// 3 = Xor (A ⊕ B)
/// 4 = Heal (Solid 0)
#[no_mangle]
pub extern "C" fn wasm_brep3d_execute(
    op_code: u32,
    winding_rule: u32,
    snap_factor: f64,
    simplify_collinear: u32,
) -> u32 {
    WASM_BREP3D_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.input_data.is_empty() {
            return 0;
        }
        buf.ensure_parsed();

        let rule = match winding_rule {
            0 => WindingRule::Positive,
            2 => WindingRule::EvenOdd,
            _ => WindingRule::NonZero,
        };
        let opts = KernelOptions {
            winding_rule: rule,
            snap_tolerance: if snap_factor > 0.0 {
                Some(snap_factor)
            } else {
                None
            },
            simplify_collinear: simplify_collinear != 0,
        };

        let empty_solid = Solid3D { faces: Vec::new() };
        let s0 = buf.parsed_solids.first().unwrap_or(&empty_solid);
        let s1 = buf.parsed_solids.get(1).unwrap_or(&empty_solid);

        let res = match op_code {
            0 => union_3d(s0, s1, opts),
            1 => intersect_3d(s0, s1, opts),
            2 => difference_3d(s0, s1, opts),
            3 => xor_3d(s0, s1, opts),
            _ => heal_3d(s0, opts),
        };

        buf.out_coords.clear();
        buf.out_loop_meta.clear();
        buf.out_face_normals.clear();
        buf.out_triangles.clear();

        let mut pt_cursor = 0u32;
        for (f_idx, face) in res.solid.faces.iter().enumerate() {
            buf.out_face_normals.push(face.normal.x);
            buf.out_face_normals.push(face.normal.y);
            buf.out_face_normals.push(face.normal.z);

            let outer_cnt = face.outer.len() as u32;
            buf.out_loop_meta.push(f_idx as u32);
            buf.out_loop_meta.push(0);
            buf.out_loop_meta.push(pt_cursor);
            buf.out_loop_meta.push(outer_cnt);
            for p in &face.outer {
                buf.out_coords.push(p.x);
                buf.out_coords.push(p.y);
                buf.out_coords.push(p.z);
            }
            pt_cursor += outer_cnt;

            for hole in &face.holes {
                let hole_cnt = hole.len() as u32;
                buf.out_loop_meta.push(f_idx as u32);
                buf.out_loop_meta.push(1);
                buf.out_loop_meta.push(pt_cursor);
                buf.out_loop_meta.push(hole_cnt);
                for p in hole {
                    buf.out_coords.push(p.x);
                    buf.out_coords.push(p.y);
                    buf.out_coords.push(p.z);
                }
                pt_cursor += hole_cnt;
            }

            let f_num = f_idx as f64;
            let nx = face.normal.x;
            let ny = face.normal.y;
            let nz = face.normal.z;
            if face.holes.is_empty() && face.outer.len() == 3 {
                let a = face.outer[0];
                let b = face.outer[1];
                let c = face.outer[2];
                buf.out_triangles.extend_from_slice(&[
                    a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, nx, ny, nz, f_num,
                ]);
            } else if face.holes.is_empty() && face.outer.len() == 4 {
                let p0 = face.outer[0];
                let p1 = face.outer[1];
                let p2 = face.outer[2];
                let p3 = face.outer[3];
                if quad_choose_diagonal_02(p0, p1, p2, p3, face.normal) {
                    buf.out_triangles.extend_from_slice(&[
                        p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, nx, ny, nz, f_num,
                        p0.x, p0.y, p0.z, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z, nx, ny, nz, f_num,
                    ]);
                } else {
                    buf.out_triangles.extend_from_slice(&[
                        p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p3.x, p3.y, p3.z, nx, ny, nz, f_num,
                        p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z, nx, ny, nz, f_num,
                    ]);
                }
            } else {
                let tris = triangulate_face_3d(face);
                for [a, b, c] in tris {
                    buf.out_triangles.extend_from_slice(&[
                        a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, nx, ny, nz, f_num,
                    ]);
                }
            }
        }

        buf.out_volume = res.total_volume;
        buf.out_area = res.total_surface_area;
        buf.out_vertex_count = res.vertex_count as u32;
        buf.out_edge_count = res.edge_count as u32;
        buf.out_unmatched_he_count = res.unmatched_half_edge_count as u32;

        res.solid.faces.len() as u32
    })
}

#[no_mangle]
pub extern "C" fn wasm_brep3d_get_out_coords_ptr() -> *const f64 {
    WASM_BREP3D_BUF.with(|cell| cell.borrow().out_coords.as_ptr())
}

#[no_mangle]
pub extern "C" fn wasm_brep3d_get_out_coords_len() -> usize {
    WASM_BREP3D_BUF.with(|cell| cell.borrow().out_coords.len())
}

#[no_mangle]
pub extern "C" fn wasm_brep3d_get_out_loop_meta_ptr() -> *const u32 {
    WASM_BREP3D_BUF.with(|cell| cell.borrow().out_loop_meta.as_ptr())
}

#[no_mangle]
pub extern "C" fn wasm_brep3d_get_out_loop_count() -> usize {
    WASM_BREP3D_BUF.with(|cell| cell.borrow().out_loop_meta.len() / 4)
}

#[no_mangle]
pub extern "C" fn wasm_brep3d_get_out_face_normals_ptr() -> *const f64 {
    WASM_BREP3D_BUF.with(|cell| cell.borrow().out_face_normals.as_ptr())
}

#[no_mangle]
pub extern "C" fn wasm_brep3d_get_out_face_count() -> usize {
    WASM_BREP3D_BUF.with(|cell| cell.borrow().out_face_normals.len() / 3)
}

#[no_mangle]
pub extern "C" fn wasm_brep3d_get_out_triangles_ptr() -> *const f64 {
    WASM_BREP3D_BUF.with(|cell| cell.borrow().out_triangles.as_ptr())
}

#[no_mangle]
pub extern "C" fn wasm_brep3d_get_out_triangles_len() -> usize {
    WASM_BREP3D_BUF.with(|cell| cell.borrow().out_triangles.len())
}

#[no_mangle]
pub extern "C" fn wasm_brep3d_get_out_volume() -> f64 {
    WASM_BREP3D_BUF.with(|cell| cell.borrow().out_volume)
}

#[no_mangle]
pub extern "C" fn wasm_brep3d_get_out_area() -> f64 {
    WASM_BREP3D_BUF.with(|cell| cell.borrow().out_area)
}

#[no_mangle]
pub extern "C" fn wasm_brep3d_get_out_vertex_count() -> u32 {
    WASM_BREP3D_BUF.with(|cell| cell.borrow().out_vertex_count)
}

#[no_mangle]
pub extern "C" fn wasm_brep3d_get_out_edge_count() -> u32 {
    WASM_BREP3D_BUF.with(|cell| cell.borrow().out_edge_count)
}

#[no_mangle]
pub extern "C" fn wasm_brep3d_get_out_unmatched_he_count() -> u32 {
    WASM_BREP3D_BUF.with(|cell| cell.borrow().out_unmatched_he_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_3d_overlapping_boxes_inclusion_exclusion_and_manifold() {
        // Two 10x10x10 cubes overlapping in a 5x5x5 corner (vol = 125)
        let box_a = Solid3D::from_box([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let box_b = Solid3D::from_box([5.0, 5.0, 5.0], [15.0, 15.0, 15.0]);
        let opts = KernelOptions::default();

        let u = union_3d(&box_a, &box_b, opts);
        let i = intersect_3d(&box_a, &box_b, opts);
        let d = difference_3d(&box_a, &box_b, opts);
        let x = xor_3d(&box_a, &box_b, opts);

        assert!((u.total_volume - 1875.0).abs() < 1e-4, "u.vol = {}", u.total_volume);
        assert!((i.total_volume - 125.0).abs() < 1e-4, "i.vol = {}", i.total_volume);
        assert!((d.total_volume - 875.0).abs() < 1e-4, "d.vol = {}", d.total_volume);
        assert!((x.total_volume - 1750.0).abs() < 1e-4, "x.vol = {}", x.total_volume);

        // Every output solid must be a closed, watertight 2-manifold (0 unmatched half-edges)
        assert_eq!(u.unmatched_half_edge_count, 0);
        assert_eq!(i.unmatched_half_edge_count, 0);
        assert_eq!(d.unmatched_half_edge_count, 0);
        assert_eq!(x.unmatched_half_edge_count, 0);
    }

    #[test]
    fn test_3d_coplanar_kissing_boxes_and_through_hole() {
        // Two 10x10x10 boxes sharing the exact plane x = 10:
        // Union should merge their coplanar top/bottom/front/back faces into 6 single faces!
        let left = Solid3D::from_box([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let right = Solid3D::from_box([10.0, 0.0, 0.0], [20.0, 10.0, 10.0]);
        let opts = KernelOptions::default();

        let u = union_3d(&left, &right, opts);
        assert!((u.total_volume - 2000.0).abs() < 1e-4);
        assert_eq!(u.solid.faces.len(), 6);
        assert_eq!(u.unmatched_half_edge_count, 0);

        // Intersection of kissing boxes has 0 volume and 0 faces
        let i = intersect_3d(&left, &right, opts);
        assert_eq!(i.solid.faces.len(), 0);
        assert!(i.total_volume.abs() < 1e-6);

        // Subtracting a through-pillar [3,7] x [3,7] x [-2, 12] from `left` punches a square hole
        // in the top and bottom faces of `left`!
        let pillar = Solid3D::from_box([3.0, 3.0, -2.0], [7.0, 7.0, 12.0]);
        let holed = difference_3d(&left, &pillar, opts);
        // Expected volume: 1000 - 16 * 10 = 840
        assert!((holed.total_volume - 840.0).abs() < 1e-4, "holed.vol = {}", holed.total_volume);
        assert_eq!(holed.unmatched_half_edge_count, 0);
        // Top and bottom faces should each have 1 interior hole!
        let faces_with_holes = holed.solid.faces.iter().filter(|f| !f.holes.is_empty()).count();
        assert_eq!(faces_with_holes, 2);

        // Verify that the triangulated mesh of `holed` has the EXACT same area and volume as the B-Rep faces!
        let mut tri_area = 0.0;
        let mut tri_vol6 = 0.0;
        for f in &holed.solid.faces {
            let f_tris = triangulate_face_3d(f);
            let mut f_tri_area = 0.0;
            for [a, b, c] in f_tris {
                let cr = b.sub(a).cross(c.sub(a));
                // Every triangle normal must align positively with `f.normal`
                assert!(
                    cr.dot(f.normal) > 0.0,
                    "Triangle orientation inverted relative to face normal!"
                );
                f_tri_area += 0.5 * cr.norm();
                tri_vol6 += a.dot(b.cross(c));
            }
            assert!(
                (f_tri_area - f.area()).abs() < 1e-6,
                "Face triangulation area mismatch: {} != {}",
                f_tri_area,
                f.area()
            );
            tri_area += f_tri_area;
        }
        assert!((tri_area - holed.total_surface_area).abs() < 1e-5);
        assert!((tri_vol6 / 6.0 - holed.total_volume).abs() < 1e-5);

        // Sweep pillar across interior, kissing boundary, and corner notches + rotation
        let block = Solid3D::from_box([-6.0, -6.0, -4.0], [6.0, 6.0, 4.0]);
        for &dx in &[-4.0, -3.5, -2.0, 0.0, 1.75, 3.5, 4.5] {
            for &dy in &[-3.5, 0.0, 2.25, 3.5] {
                let cutter = Solid3D::from_box(
                    [-2.5 + dx, -2.5 + dy, -6.5],
                    [2.5 + dx, 2.5 + dy, 6.5],
                );
                let res = difference_3d(&block, &cutter, opts);
                assert_eq!(
                    res.unmatched_half_edge_count, 0,
                    "Sweep dx={}, dy={} unmatched half-edges={}",
                    dx, dy, res.unmatched_half_edge_count
                );
                for f in &res.solid.faces {
                    let origin = f.outer[0];
                    let (u_ax, v_ax) = build_plane_frame(f.normal);
                    let proj = |p: &Point3D| Point2D {
                        x: p.sub(origin).dot(u_ax),
                        y: p.sub(origin).dot(v_ax),
                    };
                    let mut f_loops_2d = vec![f.outer.iter().map(proj).collect::<Vec<_>>()];
                    for h in &f.holes {
                        f_loops_2d.push(h.iter().map(proj).collect::<Vec<_>>());
                    }
                    let tris = triangulate_face_3d(f);
                    let mut f_area = 0.0;
                    for [ta, tb, tc] in tris {
                        let cr = tb.sub(ta).cross(tc.sub(ta));
                        assert!(cr.dot(f.normal) > 0.0);
                        let cent_3d = ta.add(tb).add(tc).scale(1.0 / 3.0);
                        assert_eq!(
                            loops_winding_2d(proj(&cent_3d), &f_loops_2d),
                            1,
                            "Sweep dx={}, dy={} triangle centroid inside hole or outside face!",
                            dx,
                            dy
                        );
                        f_area += 0.5 * cr.norm();
                    }
                    assert!((f_area - f.area()).abs() < 1e-5);
                }
            }
        }

        // Two-hole test: subtract two parallel pillars from `block` sequentially and verify 2 holes per cap
        let p1 = Solid3D::from_box([-4.5, -2.0, -6.0], [-1.0, 2.0, 6.0]);
        let p2 = Solid3D::from_box([1.0, -2.0, -6.0], [4.5, 2.0, 6.0]);
        let step1 = difference_3d(&block, &p1, opts);
        let step2 = difference_3d(&step1.solid, &p2, opts);
        assert_eq!(step2.unmatched_half_edge_count, 0);
        let expected_vol = 12.0 * 12.0 * 8.0 - 2.0 * (3.5 * 4.0 * 8.0);
        assert!((step2.total_volume - expected_vol).abs() < 1e-5);
        for f in &step2.solid.faces {
            if f.normal.z.abs() > 0.9 {
                assert_eq!(f.holes.len(), 2, "Cap face should have 2 interior holes!");
            }
            let tris = triangulate_face_3d(f);
            let f_area: f64 = tris
                .iter()
                .map(|[a, b, c]| 0.5 * b.sub(*a).cross(c.sub(*a)).norm())
                .sum();
            assert!((f_area - f.area()).abs() < 1e-5);
        }
    }

    #[test]
    fn test_3d_extruded_polygons_and_adversarial_fuzzer() {
        // 1. Extruded L-bracket unioned with a tilted prism
        let l_poly = [
            [0.0, 0.0],
            [12.0, 0.0],
            [12.0, 4.0],
            [4.0, 4.0],
            [4.0, 12.0],
            [0.0, 12.0],
        ];
        let l_solid = Solid3D::from_extruded_polygon(&l_poly, &[], 0.0, 8.0);
        // Area of L = 12*4 + 4*8 = 80; height = 8 => vol = 640
        assert!((l_solid.volume() - 640.0).abs() < 1e-4);

        let cross_bar = Solid3D::from_box([2.0, 2.0, 2.0], [10.0, 10.0, 6.0]);
        let opts = KernelOptions::default();

        let u = union_3d(&l_solid, &cross_bar, opts);
        let i = intersect_3d(&l_solid, &cross_bar, opts);
        let d_ab = difference_3d(&l_solid, &cross_bar, opts);
        let d_ba = difference_3d(&cross_bar, &l_solid, opts);
        let x = xor_3d(&l_solid, &cross_bar, opts);

        let vol_a = l_solid.volume();
        let vol_b = cross_bar.volume();
        assert!(
            (u.total_volume + i.total_volume - (vol_a + vol_b)).abs() < 1e-3,
            "Inclusion-Exclusion failed: {} + {} != {} + {}",
            u.total_volume,
            i.total_volume,
            vol_a,
            vol_b
        );
        assert!(
            (d_ab.total_volume + d_ba.total_volume - x.total_volume).abs() < 1e-3,
            "Symmetric difference decomposition failed"
        );
        assert_eq!(u.unmatched_half_edge_count, 0);
        assert_eq!(i.unmatched_half_edge_count, 0);
        assert_eq!(d_ab.unmatched_half_edge_count, 0);
        assert_eq!(x.unmatched_half_edge_count, 0);

        // 2. 40-trial randomized 3D Boolean fuzzer (boxes + extruded polygons + coplanar snaps)
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next_f64 = || -> f64 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed as f64) / (u64::MAX as f64)
        };

        for trial in 0..40 {
            let x0 = (next_f64() * 8.0).floor();
            let y0 = (next_f64() * 8.0).floor();
            let z0 = (next_f64() * 8.0).floor();
            let dx = 3.0 + (next_f64() * 8.0).floor();
            let dy = 3.0 + (next_f64() * 8.0).floor();
            let dz = 3.0 + (next_f64() * 8.0).floor();
            let a = Solid3D::from_box([x0, y0, z0], [x0 + dx, y0 + dy, z0 + dz]);

            // Mix exact grid-aligned (coplanar/kissing/T-junction) and jittered boxes
            let jitter = if trial % 2 == 0 {
                0.0
            } else {
                (next_f64() - 0.5) * 0.37
            };
            let bx0 = (next_f64() * 8.0).floor() + jitter;
            let by0 = (next_f64() * 8.0).floor() + jitter;
            let bz0 = (next_f64() * 8.0).floor() + jitter;
            let bdx = 3.0 + (next_f64() * 8.0).floor();
            let bdy = 3.0 + (next_f64() * 8.0).floor();
            let bdz = 3.0 + (next_f64() * 8.0).floor();
            let b = Solid3D::from_box([bx0, by0, bz0], [bx0 + bdx, by0 + bdy, bz0 + bdz]);

            let u = union_3d(&a, &b, opts);
            let i = intersect_3d(&a, &b, opts);
            let d = difference_3d(&a, &b, opts);

            let va = a.volume();
            let vb = b.volume();

            assert!(
                (u.total_volume + i.total_volume - (va + vb)).abs() < 1e-2,
                "Trial {} failed inclusion-exclusion: u={} + i={} != va={} + vb={}",
                trial,
                u.total_volume,
                i.total_volume,
                va,
                vb
            );
            assert!(
                (d.total_volume + i.total_volume - va).abs() < 1e-2,
                "Trial {} failed difference identity: d={} + i={} != va={}",
                trial,
                d.total_volume,
                i.total_volume,
                va
            );
            assert_eq!(
                u.unmatched_half_edge_count, 0,
                "Trial {} union non-manifold: {} unmatched half-edges",
                trial, u.unmatched_half_edge_count
            );
            assert_eq!(
                i.unmatched_half_edge_count, 0,
                "Trial {} intersect non-manifold: {} unmatched half-edges",
                trial, i.unmatched_half_edge_count
            );
            assert_eq!(
                d.unmatched_half_edge_count, 0,
                "Trial {} difference non-manifold: {} unmatched half-edges",
                trial, d.unmatched_half_edge_count
            );

            for (label, res) in [("union", &u), ("intersect", &i), ("difference", &d)] {
                let mut mesh_vol6 = 0.0;
                for f in &res.solid.faces {
                    let tris = triangulate_face_3d(f);
                    let mut f_area = 0.0;
                    let origin = f.outer[0];
                    let (u_ax, v_ax) = build_plane_frame(f.normal);
                    let proj = |p: &Point3D| Point2D {
                        x: p.sub(origin).dot(u_ax),
                        y: p.sub(origin).dot(v_ax),
                    };
                    let mut f_loops_2d = vec![f.outer.iter().map(proj).collect::<Vec<_>>()];
                    for h in &f.holes {
                        f_loops_2d.push(h.iter().map(proj).collect::<Vec<_>>());
                    }

                    for [ta, tb, tc] in tris {
                        let cr = tb.sub(ta).cross(tc.sub(ta));
                        assert!(
                            cr.dot(f.normal) > 0.0,
                            "Trial {} {} triangle normal inverted!",
                            trial,
                            label
                        );
                        let cent_3d = ta.add(tb).add(tc).scale(1.0 / 3.0);
                        let cent_2d = proj(&cent_3d);
                        assert_eq!(
                            loops_winding_2d(cent_2d, &f_loops_2d),
                            1,
                            "Trial {} {} triangle centroid lies outside face or inside a hole!",
                            trial,
                            label
                        );
                        f_area += 0.5 * cr.norm();
                        mesh_vol6 += ta.dot(tb.cross(tc));
                    }
                    assert!(
                        (f_area - f.area()).abs() < 1e-4,
                        "Trial {} {} face triangulation area mismatch: {} != {}",
                        trial,
                        label,
                        f_area,
                        f.area()
                    );
                }
                assert!(
                    (mesh_vol6 / 6.0 - res.total_volume).abs() < 1e-3,
                    "Trial {} {} mesh volume mismatch: {} != {}",
                    trial,
                    label,
                    mesh_vol6 / 6.0,
                    res.total_volume
                );
            }
        }
    }

    fn make_cylinder_3d(radius: f64, z_min: f64, z_max: f64, segs: usize) -> Solid3D {
        let mut ring = Vec::with_capacity(segs);
        for i in 0..segs {
            let a = (2.0 * core::f64::consts::PI * (i as f64)) / (segs as f64);
            ring.push((radius * a.cos(), radius * a.sin()));
        }
        let mut faces = Vec::new();
        faces.push(Face3D {
            outer: ring.iter().rev().map(|&(x, y)| Point3D::new(x, y, z_min)).collect(),
            holes: vec![],
            normal: Point3D::new(0.0, 0.0, -1.0),
        });
        faces.push(Face3D {
            outer: ring.iter().map(|&(x, y)| Point3D::new(x, y, z_max)).collect(),
            holes: vec![],
            normal: Point3D::new(0.0, 0.0, 1.0),
        });
        for i in 0..segs {
            let (x0, y0) = ring[i];
            let (x1, y1) = ring[(i + 1) % segs];
            let outer = vec![
                Point3D::new(x0, y0, z_min),
                Point3D::new(x1, y1, z_min),
                Point3D::new(x1, y1, z_max),
                Point3D::new(x0, y0, z_max),
            ];
            let normal = compute_newell_normal(&outer).unwrap();
            faces.push(Face3D { outer, holes: vec![], normal });
        }
        Solid3D { faces }
    }

    fn rotate_and_shift_solid(
        solid: &Solid3D,
        rx_deg: f64,
        ry_deg: f64,
        rz_deg: f64,
        tx: f64,
        ty: f64,
        tz: f64,
    ) -> Solid3D {
        let rx = rx_deg.to_radians();
        let ry = ry_deg.to_radians();
        let rz = rz_deg.to_radians();
        let (cx, sx) = (rx.cos(), rx.sin());
        let (cy, sy) = (ry.cos(), ry.sin());
        let (cz, sz) = (rz.cos(), rz.sin());
        let xform = |p: Point3D| -> Point3D {
            let y1 = p.y * cx - p.z * sx;
            let z1 = p.y * sx + p.z * cx;
            let x2 = p.x * cy + z1 * sy;
            let z2 = -p.x * sy + z1 * cy;
            let x3 = x2 * cz - y1 * sz;
            let y3 = x2 * sz + y1 * cz;
            Point3D::new(x3 + tx, y3 + ty, z2 + tz)
        };
        let mut faces = Vec::with_capacity(solid.faces.len());
        for f in &solid.faces {
            let outer: Vec<Point3D> = f.outer.iter().copied().map(xform).collect();
            let normal = compute_newell_normal(&outer).unwrap();
            let holes = f.holes.iter().map(|h| h.iter().copied().map(xform).collect()).collect();
            faces.push(Face3D { outer, holes, normal });
        }
        Solid3D { faces }
    }

    fn make_cone_3d(r_bot: f64, r_top: f64, z_min: f64, z_max: f64, segs: usize) -> Solid3D {
        let mut bot = Vec::with_capacity(segs);
        let mut top = Vec::with_capacity(segs);
        for i in 0..segs {
            let a = (2.0 * core::f64::consts::PI * (i as f64)) / (segs as f64);
            bot.push((r_bot * a.cos(), r_bot * a.sin()));
            top.push((r_top * a.cos(), r_top * a.sin()));
        }
        let mut faces = Vec::new();
        faces.push(Face3D {
            outer: bot
                .iter()
                .rev()
                .map(|&(x, y)| Point3D::new(x, y, z_min))
                .collect(),
            holes: vec![],
            normal: Point3D::new(0.0, 0.0, -1.0),
        });
        if r_top <= 1e-9 {
            let apex = Point3D::new(0.0, 0.0, z_max);
            for i in 0..segs {
                let (x0, y0) = bot[i];
                let (x1, y1) = bot[(i + 1) % segs];
                let outer = vec![Point3D::new(x0, y0, z_min), Point3D::new(x1, y1, z_min), apex];
                let normal = compute_newell_normal(&outer).unwrap();
                faces.push(Face3D {
                    outer,
                    holes: vec![],
                    normal,
                });
            }
        } else {
            faces.push(Face3D {
                outer: top
                    .iter()
                    .map(|&(x, y)| Point3D::new(x, y, z_max))
                    .collect(),
                holes: vec![],
                normal: Point3D::new(0.0, 0.0, 1.0),
            });
            for i in 0..segs {
                let (bx0, by0) = bot[i];
                let (bx1, by1) = bot[(i + 1) % segs];
                let (tx0, ty0) = top[i];
                let (tx1, ty1) = top[(i + 1) % segs];
                let outer = vec![
                    Point3D::new(bx0, by0, z_min),
                    Point3D::new(bx1, by1, z_min),
                    Point3D::new(tx1, ty1, z_max),
                    Point3D::new(tx0, ty0, z_max),
                ];
                let normal = compute_newell_normal(&outer).unwrap();
                faces.push(Face3D {
                    outer,
                    holes: vec![],
                    normal,
                });
            }
        }
        Solid3D { faces }
    }

    #[test]
    fn test_glancing_cylinders_and_cones_at_coarse_tolerance() {
        let opts = KernelOptions {
            snap_tolerance: Some(0.1),
            winding_rule: WindingRule::NonZero,
            simplify_collinear: false,
        };

        // C2: Two cylinders nearly parallel at 1.75 deg glancing angle
        let c2_a = make_cylinder_3d(3.5, -6.0, 6.0, 16);
        let c2_b = rotate_and_shift_solid(&c2_a, 1.75, 0.8, 7.5, 0.4, 0.0, 0.0);
        let u_c2 = union_3d(&c2_a, &c2_b, opts);
        let i_c2 = intersect_3d(&c2_a, &c2_b, opts);
        let d_c2 = difference_3d(&c2_a, &c2_b, opts);
        assert_eq!(u_c2.unmatched_half_edge_count, 0);
        assert_eq!(i_c2.unmatched_half_edge_count, 0);
        assert_eq!(d_c2.unmatched_half_edge_count, 0);
        assert!(
            (u_c2.total_volume + i_c2.total_volume - (c2_a.volume() + c2_b.volume())).abs() < 1.0,
            "C2 inclusion-exclusion failed: {} + {} != {} + {}",
            u_c2.total_volume,
            i_c2.total_volume,
            c2_a.volume(),
            c2_b.volume()
        );
        assert!((d_c2.total_volume + i_c2.total_volume - c2_a.volume()).abs() < 1.0);
        for tol in [1e-7, 1e-4, 1e-2, 0.1] {
            let t_opts = KernelOptions {
                snap_tolerance: Some(tol),
                winding_rule: WindingRule::NonZero,
                simplify_collinear: false,
            };
            let u = union_3d(&c2_a, &c2_b, t_opts);
            let i = intersect_3d(&c2_a, &c2_b, t_opts);
            assert!(
                (u.total_volume + i.total_volume - (c2_a.volume() + c2_b.volume())).abs() < 1.0,
                "C2 tol={} inclusion-exclusion failed: {} + {} != {} + {}",
                tol,
                u.total_volume,
                i.total_volume,
                c2_a.volume(),
                c2_b.volume()
            );
        }

        // K3: Cone & cylinder coaxial base-to-cap kissing & glancing 2.2 deg
        let k3_a = make_cylinder_3d(4.0, -4.0, 4.0, 16);
        let k3_b = rotate_and_shift_solid(
            &make_cone_3d(5.0, 1.5, -4.0, 4.0, 16),
            2.2,
            0.0,
            11.25,
            0.15,
            0.0,
            0.0,
        );
        let u_k3 = union_3d(&k3_a, &k3_b, opts);
        let i_k3 = intersect_3d(&k3_a, &k3_b, opts);
        let d_k3 = difference_3d(&k3_a, &k3_b, opts);
        assert_eq!(u_k3.unmatched_half_edge_count, 0);
        assert_eq!(i_k3.unmatched_half_edge_count, 0);
        assert_eq!(d_k3.unmatched_half_edge_count, 0);
        assert!(
            (u_k3.total_volume + i_k3.total_volume - (k3_a.volume() + k3_b.volume())).abs() < 1.0,
            "K3 inclusion-exclusion failed: {} + {} != {} + {}",
            u_k3.total_volume,
            i_k3.total_volume,
            k3_a.volume(),
            k3_b.volume()
        );
        assert!((d_k3.total_volume + i_k3.total_volume - k3_a.volume()).abs() < 1.0);

        // K4: Two cones glancing along slant generators 1.5 deg
        let k4_a = make_cone_3d(4.0, 0.0, -4.0, 4.0, 16);
        let k4_b = rotate_and_shift_solid(&k4_a, 181.5, 1.0, 11.25, 3.75, 0.0, 0.0);
        let u_k4 = union_3d(&k4_a, &k4_b, opts);
        let i_k4 = intersect_3d(&k4_a, &k4_b, opts);
        let d_k4 = difference_3d(&k4_a, &k4_b, opts);
        assert_eq!(u_k4.unmatched_half_edge_count, 0);
        assert_eq!(i_k4.unmatched_half_edge_count, 0);
        assert_eq!(d_k4.unmatched_half_edge_count, 0);
        assert!(
            (u_k4.total_volume + i_k4.total_volume - (k4_a.volume() + k4_b.volume())).abs() < 2.0,
            "K4 inclusion-exclusion failed: {} + {} != {} + {}",
            u_k4.total_volume,
            i_k4.total_volume,
            k4_a.volume(),
            k4_b.volume()
        );
        assert!((d_k4.total_volume + i_k4.total_volume - k4_a.volume()).abs() < 2.0);

        let box_a = Solid3D::from_box([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]);
        let box_b = rotate_and_shift_solid(
            &Solid3D::from_box([-4.5, -4.5, -4.5], [4.5, 4.5, 4.5]),
            -22.0 * 0.6,
            -22.0,
            -22.0 * 0.35,
            3.1,
            0.8,
            -0.4,
        );
        for tol in [1e-4, 0.05, 10.0f64.powf(-0.55), 0.5] {
            let b_opts = KernelOptions {
                snap_tolerance: Some(tol),
                winding_rule: WindingRule::NonZero,
                simplify_collinear: false,
            };
            let u_b = union_3d(&box_a, &box_b, b_opts);
            let i_b = intersect_3d(&box_a, &box_b, b_opts);
            let d_b = difference_3d(&box_a, &box_b, b_opts);
            let x_b = xor_3d(&box_a, &box_b, b_opts);
            assert!(
                u_b.total_volume > 0.0
                    && i_b.total_volume > 0.0
                    && d_b.total_volume > 0.0
                    && x_b.total_volume > 0.0,
                "tol={} volumes: u={}, i={}, d={}, x={}",
                tol,
                u_b.total_volume,
                i_b.total_volume,
                d_b.total_volume,
                x_b.total_volume
            );
        }

        // P1: Extruded multi-hole flange plate cut by 1.3 deg glancing frustum
        let p1_a = Solid3D::from_extruded_polygon(
            &[[-6.0, -6.0], [6.0, -6.0], [6.0, 6.0], [-6.0, 6.0]],
            &[
                vec![[-4.0, -4.0], [-1.0, -4.0], [-1.0, -1.0], [-4.0, -1.0]],
                vec![[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]],
            ],
            -2.0,
            2.0,
        );
        let p1_b = rotate_and_shift_solid(
            &make_cone_3d(4.5, 2.0, -4.0, 4.0, 16),
            1.3,
            -0.9,
            7.5,
            0.2,
            0.2,
            0.0,
        );
        for tol_p1 in [1e-4, 1e-2, 0.1] {
            let p1_opts = KernelOptions {
                snap_tolerance: Some(tol_p1),
                winding_rule: WindingRule::NonZero,
                simplify_collinear: false,
            };
            let u_p1 = union_3d(&p1_a, &p1_b, p1_opts);
            let i_p1 = intersect_3d(&p1_a, &p1_b, p1_opts);
            let d_p1 = difference_3d(&p1_a, &p1_b, p1_opts);
            assert_eq!(u_p1.unmatched_half_edge_count, 0);
            assert_eq!(i_p1.unmatched_half_edge_count, 0);
            assert_eq!(d_p1.unmatched_half_edge_count, 0);
            assert!(
                (u_p1.total_volume + i_p1.total_volume - (p1_a.volume() + p1_b.volume())).abs()
                    < 1.0,
                "P1 tol={} inclusion-exclusion failed: {} + {} != {} + {}",
                tol_p1,
                u_p1.total_volume,
                i_p1.total_volume,
                p1_a.volume(),
                p1_b.volume()
            );
            assert!((d_p1.total_volume + i_p1.total_volume - p1_a.volume()).abs() < 1.0);
        }
    }
}
