# 3D Planar B-Rep Kernel — Stratified Tolerance Complex Architecture

Based on [`BuildingARobustBRepKernel.md`](./BuildingARobustBRepKernel.md) and built directly atop the 2D `polystrata` Stratified Tolerance Complex.

---

## 1. Core Thesis: Lifting the Stratified Tolerance Complex from 2D to 3D Planar B-Reps

Every concept in our 2D kernel (`polystrata`) shifts up one dimension while reusing the 2D planar arrangement engine on each canonical face plane:

| Dimension | 2D Kernel (`polystrata`) | 3D Planar B-Rep Kernel (`polystrata-3d`) |
| :--- | :--- | :--- |
| **0-Strata** | 2D Vertex Disk $B(\mathbf{p}_i, r_i) \subset \mathbb{R}^2$ | 3D Vertex Ball $B(\mathbf{p}_i, r_i) \subset \mathbb{R}^3$ |
| **1-Strata** | 2D Segment Tube $\Omega(e_j) \subset \mathbb{R}^2$ | 3D Straight Edge Tube $\Omega(e_j) \subset \mathbb{R}^3$ |
| **2-Strata** | 2D Face Region (`RegionId`) | 3D Planar Face Patch $\Omega(f_k)$ on canonical plane $\pi_c = (\mathbf{N}_c, d_c)$ |
| **3-Strata** | — | 3D Volume Cell (`VolumeId`) bounded by oriented `CoFace`s (`2*f`, `2*f+1`) |
| **Radial Sort Hub** | Sort `HalfEdge`s CCW by exit angle around each 2D Vertex Disk | Sort `CoFace`s CCW by dihedral angle in the normal plane of each 3D Edge Tube |
| **Winding Oracle** | 2D Subtended Angle $\frac{1}{2\pi}\sum \Delta\theta_e$ + 2D Dual-Graph BFS | 3D Solid Angle $\frac{1}{4\pi}\sum \Omega_T$ (Van Oosterom–Strackee) + 3D Dual-Graph BFS |
| **Boundary Extraction** | $\mathbf{b}_1 = \partial_2 \cdot \mathbf{c}_2$ (`left_keep != right_keep`) | $\mathbf{b}_2 = \partial_3 \cdot \mathbf{c}_3$ (`keep_front != keep_back`) |
| **Same-Domain Unification** | Collinear degree-2 vertex collapse | Coplanar face patch unification via 2D `left_keep != right_keep` on each plane $\pi_c$ |

---

## 2. The Four-Stage 3D Planar Boolean & Healing Pipeline

### Stage 0: Scale-Aware 3D Normalization & Canonical Carrier Plane Clustering
1. **Origin Centering**: Compute the global 3D AABB $[\mathbf{p}_{\min}, \mathbf{p}_{\max}]$, shift the midpoint $\mathbf{C} = \frac{1}{2}(\mathbf{p}_{\min} + \mathbf{p}_{\max})$ to $(0, 0, 0)$, and compute the 3D diagonal $L_{\text{bbox}}$.
2. **Tolerance Hierarchy**: Set base tolerance $r_0 = \max(\tau_{\text{user}}, 10^{-8} L_{\text{bbox}})$ with strict stratum hierarchy $r(f) = r_0 < r(e) = 2r_0 < r(v) = 4r_0$.
3. **Canonical Carrier Planes ($\pi_c$)**:
   - Compute the best-fit plane $(\mathbf{n}, d)$ of each input polygon loop using Newell's normal formula.
   - Canonicalize unit normal orientation ($\hat{\mathbf{n}}$ has positive largest-magnitude coordinate; flip $d \leftarrow -d$ if negated) and record `orient_sign = +1` or `-1`.
   - Cluster planes whose normal alignment and signed offset across the model scale agree within $2r_0$. Each cluster defines a canonical plane $\pi_c = (\mathbf{O}_c, \mathbf{U}_c, \mathbf{V}_c, \mathbf{N}_c)$ with an orthonormal 2D frame $(\mathbf{U}_c, \mathbf{V}_c)$.

### Stage 1: Unconstrained 3D Locus Injection (Face Boundaries + Plane-Plane Intersection Segments)
1. **Input Face Boundary Edges**: Inject all 3D boundary segments of every input face loop (outer shell + holes).
2. **Plane-Plane Intersection Lines ($L_{ab} = \pi_a \cap \pi_b$)**:
   - For every pair of distinct canonical planes $(\pi_a, \pi_b)$ whose 3D face AABBs overlap:
     - Compute line direction $\mathbf{d} = \mathbf{N}_a \times \mathbf{N}_b$. If $\|\mathbf{d}\| \le 10^{-7}$ (parallel planes), skip.
     - Find the 1D intervals along $L_{ab}$ where the line crosses inside or along the boundary of input faces on $\pi_a$, and the 1D intervals where $L_{ab}$ crosses inside or along the boundary of input faces on $\pi_b$.
     - Inject the overlapping 1D intervals (plus all individual plane-chord spans intersecting the partner plane's AABB) as candidate 3D segments along $L_{ab}$.

### Stage 2: Coupled 3D 0-Stratum & 1-Stratum Fixpoint Consolidation
Because all edges in a planar B-Rep are straight 3D line segments, running 0-stratum and 1-stratum consolidation directly in $\mathbb{R}^3$ guarantees global topological consistency across every plane simultaneously:
1. **0-Stratum Ball Merge**: Merge any two 3D vertex balls $B(\mathbf{p}_i, r_i)$ and $B(\mathbf{p}_j, r_j)$ with $\|\mathbf{p}_i - \mathbf{p}_j\| \le r_i + r_j$.
2. **Micro-Edge Contraction**: Collapse any 3D edge with $v_0 == v_1$ or length $\le r(v_0) + r(v_1)$.
3. **Ball-Tube T-Junction Split**: If a 3D vertex ball $B(\mathbf{p}_i, r_i)$ touches the interior of a non-incident 3D edge $e_j = (v_0, v_1)$ within $r_i + r(e_j)$, split $e_j$ at $v_i$ globally in $E_{3D}$.
4. **Tube-Tube 3D Segment Intersection**: If two coplanar or intersecting 3D edges $e_1, e_2$ cross in $\mathbb{R}^3$ within $2r_0$, spawn/find a 3D vertex ball $v_{\text{int}}$ at the closest midpoint and split both edges in $E_{3D}$.
5. **Coincident 3D Edge Deduplication**: Merge duplicate 3D edges connecting the same canonical vertex pair $(v_a, v_b)$.

### Stage 3: 2D Face Patch Extraction ($\partial_2$) & 3D Radial Volume Extraction ($\partial_3$)
1. **Stage 3a — Planar 2D Arrangement on Each Plane $\pi_c$ ($\partial_2$)**:
   - Collect all surviving active 3D edges $e = (v_0, v_1) \in E_{3D}$ whose endpoints lie on $\pi_c$ (within $\max(r(v), 4r_0)$).
   - Project vertices to $\pi_c$'s 2D $(u, v)$ coordinates and run the 2D planar half-edge arrangement (`build_planar_topology`): sort spokes CCW around each vertex, wire `he_next`, extract minimal bounded 2D cycles, and pair CW hole loops with their enclosing CCW outer loop.
   - For each bounded 2D face cell $R_m$ on $\pi_c$:
     - Compute a guaranteed interior 2D sample point $(u_m, v_m)$ via Left-Normal Ray-Cast.
     - Evaluate the net signed surface multiplicity $\Delta w_b(R_m) \in \mathbb{Z}$ for each body $b$ by summing `orient_sign` over all input faces of body $b$ on $\pi_c$ that contain $(u_m, v_m)$.
     - Retain $R_m$ as an active 3D Face Patch $f_m$ whenever $\exists b: \Delta w_b(R_m) \ne 0$.
2. **Stage 3b — Radial CoFace Sorting Around 3D Edges & Volume Shell Extraction ($\partial_3$)**:
   - Each active 3D Face Patch $f_m$ has two `CoFace`s: `cf_front = 2*m` (normal $+\mathbf{N}_c$) and `cf_back = 2*m + 1` (normal $-\mathbf{N}_c$).
   - Around each active 3D edge $e = (v_0, v_1)$ with unit tangent $\mathbf{t}_e$, collect all incident face patches $f_m$. If $f_m$ traverses $e$ with direction sign $\sigma \in \{-1, +1\}$, its inward face-sheet vector perpendicular to $\mathbf{t}_e$ is $\mathbf{b}_{e, m} = \sigma (\mathbf{N}_c \times \mathbf{t}_e)$.
   - Sort incident face spokes CCW around $+\mathbf{t}_e$ in the normal plane of $\mathbf{t}_e$.
   - Across the dihedral wedge between spoke $i$ and spoke $(i+1) \bmod k$, the CoFace of spoke $i$ facing CCW (`cf_front` if $\sigma_i = +1$, `cf_back` if $\sigma_i = -1$) and the CoFace of spoke $i+1$ facing CW (`cf_back` if $\sigma_{i+1} = +1$, `cf_front` if $\sigma_{i+1} = -1$) bound the **exact same 3D Volume Cell**!
   - Union-find CoFaces across all dihedral wedges to form 3D Volume Cells $S_v$.

### Stage 4: 3D Solid-Angle Winding Classification & Algebraic Boundary Extraction
1. **3D Interior Sample Point & Generalized Winding Number**:
   - For each 3D Volume Cell $S_v$, step from the 3D interior pole $\mathbf{x}_m$ of a bounding CoFace along its inward volume normal to the midpoint of the first ray-face hit $\frac{1}{2} t_{\text{hit}}$.
   - Evaluate $w_b(\mathbf{x}_v) = \frac{1}{4\pi} \sum_{T \in \partial B_b} \Omega_T(\mathbf{x}_v)$ via the exact Van Oosterom–Strackee solid angle formula (and cross-check via 3D Dual-Graph BFS across shared face patches).
   - Apply the Boolean predicate (`Union`, `Intersect`, `Difference`, `Xor`, `Heal`, etc.) to set `volumes[v].keep`.
2. **Algebraic Boundary Extraction ($\mathbf{b}_2 = \partial_3 \cdot \mathbf{c}_3$) & Coplanar Unification**:
   - Keep `cf_front` (normal $+\mathbf{N}_c$) iff `!keep_front && keep_back`; keep `cf_back` (normal $-\mathbf{N}_c$) iff `keep_front && !keep_back`.
   - On each canonical plane $\pi_c$, merge adjacent surviving patches sharing the same outward normal $\pm \mathbf{N}_c$ by cancelling shared internal 2D edges (`left_keep != right_keep` on $\pi_c$).

---

## 3. Automated Invariant Verification Oracles (3D Fuzz Suite)

1. **Watertight 2-Manifold Edge Pairing ($\partial_1 \cdot \partial_2 = 0$ and $\partial_2 \cdot \mathbf{b}_2 = 0$)**:
   - Every directed half-edge $(u \to v)$ in the extracted 3D B-Rep faces is paired with an equal number of opposite directed half-edges $(v \to u)$ (0 open boundary cracks).
2. **Divergence Theorem Exact Volume Identity**:
   - Signed volume $V(S) = \frac{1}{6} \sum_{T = (\mathbf{a}, \mathbf{b}, \mathbf{c}) \in \partial S} \mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})$ satisfies exact inclusion-exclusion:
     $$\text{Vol}(A \cup B) + \text{Vol}(A \cap B) = \text{Vol}(A) + \text{Vol}(B)$$
     $$\text{Vol}(A \setminus B) + \text{Vol}(A \cap B) = \text{Vol}(A)$$
     $$\text{Vol}(A \oplus B) = \text{Vol}(A \cup B) - \text{Vol}(A \cap B)$$
3. **3D Point-in-Solid Solid-Angle Oracle**:
   - Random 3D probe points $\mathbf{x} \in \mathbb{R}^3$ outside tolerance zones satisfy $\text{Inside}(\mathbf{x}, \text{Op}(A, B)) \iff \text{Op}(\text{Inside}(\mathbf{x}, A), \text{Inside}(\mathbf{x}, B))$.
