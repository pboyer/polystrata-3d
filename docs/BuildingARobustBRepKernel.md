# Building a Robust B-rep Kernel

Source: <https://gemini.google.com/app/3205dd5a6c962139>

Extracted from `BuildingARobustBRepKernel.pdf`. Display math is reconstructed from positioned glyphs, so some indices are still separated from their symbols.

## User

im looking to build a brep geometry kernel. i want it to exceed the robustness and speed of booleans in parasolid or occt, but i want to automatically manage tolerances for edges vertices, etc. I don't want you to drop a bunch of cool different techniques (eg simd, bvh, etc) that could potentially be combined to solve this problem. i want you to define a general purpose algorithm that will guarantee that topology is always correct. it may be that the way to do this is to intersect and then repair. regardless, i don't want to end up with many sub-problems/algorithms to verify, we want a small number of invariants and corresponding algorithms that ensure these invariants cannot be violated.

## Response

Traditional B-rep kernels like Parasolid and Open CASCADE (OCCT) struggle with Booleans because they interleave inexact metric intersection with combinatorial topology construction.

In OCCT’s BOPAlgo_PaveFiller, intersections are computed in a pairwise dimensional cascade (V × V , V × E , E × E , V × F , E × F , F × F ), tolerances are mutated on the fly, face loops are reconstructed in 2D (u, v) surface parameter space using approximated p-curves, and inside/outside classification relies on local normal/tangent cross-products. This architecture fails to scale cleanly for three reasons:

1. Non-Transitivity of Pairwise Proximity: Dynamic tolerance expansion during pairwise intersections causes cascading side effects—an intersection curve oscillating near a tangency can enter and exit a vertex sphere multiple times, or an expanded vertex can swallow an unrelated edge mid-pipeline.

2. Dual-Space (3D vs. 2D UV) Drift: Two 3D edges that meet cleanly inside a 3D vertex tolerance sphere can still cross each other or fail to close when projected as approximated 2D p-curves (u(t), v(t)) near a surface singularity or grazing contact, corrupting face loop orientation.

3. Ill-Conditioned Local Predicates: Using n1 × n2 or tangent dot products at intersection loci to determine manifold connectivity breaks down whenever surfaces meet tangentially.

To guarantee topological correctness while automatically managing tolerances—and to reduce the entire kernel to a minimal set of verifiable algorithms—you must decouple metric uncertainty resolution from combinatorial topology extraction.

### 1. The Mathematical Foundation: The Stratified Tolerance Complex

Instead of treating a B-rep as ideal zero-thickness geometry with tolerance numbers attached, define the B-rep as a Stratified Tolerance Complex K = (V, E, F, S) embedded in R3 . Every k-cell c(k) has a nominal carrier (point pi , curve γj (t), surface σk (u, v)) and a strictly positive tolerance radius r(c(k) ) > 0.

Define the Full Tolerance Zone Ω(c(k) ) as the closed r-offset of the carrier: 0-Zone (Vertex Ball): Ω(vi ) = {x ∈ R3 ∣ ∥x − pi ∥ ≤ r(vi )} 1-Zone (Edge Tube): Ω(ej ) = {x ∈ R3 ∣ dist(x, γj ) ≤ r(ej )} 2-Zone (Face Sheet): Ω(fk ) = {x ∈ R3 ∣ dist(x, σk ) ≤ r(fk )} Now define the Active Metric Stratum Ω^ (c(k) ) of each cell by subtracting the interior of all lower-dimensional tolerance zones: Ω^ (vi ) = Ω(vi ) Ω^ (ej ) = Ω(ej ) ∖ ⋃ int(Ω^ (vi )) vi ∈V Ω^ (fk ) = Ω(fk ) ∖ ⋃ int(Ω^ (vi )) ∪ ⋃ int(Ω^ (ej )) vi ∈V ej ∈E

### 2. The Three Core Invariants

Every input solid and every intermediate/output state of the Boolean algorithm is governed by three invariants. If these three invariants hold, invalid topology is mathematically impossible.

### Invariant 1: Strict Stratum Disjointness (Metric Separation)

For any two distinct cells a = b of any dimension that are not topologically incident (a ∈/ ∂b and b ∈/ ∂a), their active strata must be separated by a strictly positive distance: dist(Ω^ (a), Ω^ (b)) > 0

Consequence: Outside of the vertex balls Ω^ (V), no two edge tubes ever touch. Outside of the vertex balls and edge tubes Ω^ (V) ∪ Ω^ (E), no two face sheets ever touch. Every geometric interaction—whether V -V , V -E , E-E , E-F , F -F , grazing tangency, or partial coincidence—is simply a violation of Invariant 1.

### Invariant 2: Collar Monotonicity (Strict Engulfment)

Whenever a cell b(k) is incident to a boundary cell a(k−1) ∈ ∂b(k) , their tolerances must obey strict hierarchy (r(v) > r(e) > r(f )), and the intersection of the higher-dimensional active stratum Ω^ (b(k) ) with the boundary of the lower-dimensional tolerance zone ∂ Ω^ (a(k−1) ) must be a single topological (k − 1)-ball (collar): Vertex-Edge Collar: An incident edge curve γj (t) must cross the sphere ∂ Ω^ (vi ) strictly once with outward radial monotonicity: (γj (t) − pi ) ⋅ γj′ (t) > 0 on ∂ Ω^ (vi ) Furthermore, the tube cross-section Ω^ (ej ) ∩ ∂ Ω^ (vi ) must be a single topological disk strictly contained on the sphere ∂ Ω^ (vi ).

Edge-Face Collar: Along the cylindrical boundary of any active edge tube ∂ Ω^ (ej ), each incident face sheet Ω^ (fk ) intersects ∂ Ω^ (ej ) in a single longitudinal strip that monotonically connects the endpoint vertex spheres without folding back.

Consequence: An intersection curve that "wiggles" in and out of a vertex ball or braids with another edge near a tangency violates Invariant 2. Enforcing Invariant 2 automatically swallows micro-loops, cusps, and degenerate slivers into the lower-dimensional tolerance zone.

### Invariant 3: The Algebraic Chain Complex Identity (∂k ∘ ∂k+1 = 0)

Combinatorial topology is represented strictly as a sequence of sparse Signed Incidence Matrices over Z: ∂3 ∂2 ∂1 C3 C2 C1 C0 where ∂1 ∈ {−1, 0, +1}∣V∣×∣E∣ maps edges to boundary vertices, ∂2 ∈ {−1, 0, +1}∣E∣×∣F∣ maps faces to boundary edges, and ∂3 ∈ {−1, 0, +1}∣F∣×∣S∣ maps 3D volume cells to bounding faces.

The Algebraic Law: ∂1 ⋅ ∂2 = 0 and ∂2 ⋅ ∂3 = 0

Consequence: You never construct shells by walking half-edges with floating-point tolerances. Any subset of 3D volume cells selected by a Boolean operation has a boundary defined algebraically by matrix multiplication, which is guaranteed to be closed, manifold, and consistently oriented.

### 3. The Unified Four-Stage Boolean Algorithm

Instead of branching into dozens of special-case intersection and healing routines, the Boolean pipeline executes four deterministic stages. Stages 1 and 2 enforce Invariants 1 and 2; Stages 3 and 4 enforce Invariant 3.

1 Unconstrained Locus Injection Generate raw candidate strata without topological trimming Pool all 0, 1, 2-cells of input solids A and B into a unified working complex. For every pair of face sheets Ω(fa ) and Ω(fb ) that intersect, run a single surface-surface intersection solver to emit raw candidate 1-zones Ω(enew ) (with radius r(enew ) = r(fa ) + r(fb ) + ϵfit ) and raw endpoint 0-zones Ω(vnew ) where intersection curves terminate on patch boundaries or singularities.

Crucial rule: Do not clip curves against edges, project vertices, or resolve tangencies during surface intersection. Emit raw curves and let Stage 2 unify all interactions.

2 Fixpoint Stratum Consolidation Enforce Invariants 1 and 2 via a single contraction loop Maintain a priority queue of Invariance Violations ordered by dimension (0 → 1 → 2) and execute three atomic operators in a loop until the queue is empty (the fixpoint):

1. 0-Stratum Consolidation (Ball Disjointness & Monotonicity): Ball-Ball Overlap: If Ω^ (vi ) ∩ Ω^ (vj ) = ∅, replace vi , vj with their minimal enclosing ball B(p∗ , r∗ ) such that Ω^ (vi ) ∪ Ω^ (vj ) ⊂ B(p∗ , r∗ ).

Ball-Tube Piercing: If Ω^ (vi ) ∩ Ω^ (ej ) = ∅ for a non-incident edge ej , split ej topologically at vi and inflate r(vi ) minimally so the tube cross-section is strictly engulfed: r(vi ) > dist(pi , γj ) + r(ej ).

Collar Monotonicity Check: For every edge ej incident to vi , find all intersections of γj (t) with the sphere ∂ Ω^ (vi ).

If γj (t) lies entirely inside Ω^ (vi ), contract (delete) ej —the micro-edge is absorbed into vi .

If γj (t) crosses ∂ Ω^ (vi ) more than once (an oscillation or grazing hook), inflate r(vi ) to enclose the farthest crossing point plus r(ej ), ensuring γj (t) exits ∂ Ω^ (vi ) strictly once.

2. 1-Stratum Consolidation (Tube Disjointness): Because step 1 holds, all active edge tubes Ω^ (ej ) are strictly truncated outside the vertex balls. Now check Ω^ (e1 ) ∩ Ω^ (e2 ) for distinct edges: Localized Crossing / Grazing: If Ω^ (e1 ) ∩ Ω^ (e2 ) is a bounded region, spawn a new 0-stratum ball Ω^ (vnew ) enclosing the connected intersection component, split both edges, and push to Step 1.

Coincident Interval: If Ω^ (e1 ) and Ω^ (e2 ) overlap along an entire segment between two already-shared endpoint balls va , vb , merge them into a single edge with a unified tube encompassing both (Ω^ (e12 ) ⊃ Ω^ (e1 ) ∪ Ω^ (e2 )), and verify r(va ), r(vb ) > r(e12 ).

3. 2-Stratum Coincidence: Any region of a face sheet Ω^ (fa ) that remains entirely inside another face sheet Ω^ (fb ) across an entire connected patch bounded by active 1-strata is merged into a single shared 2-stratum with combined thickness r(fab ).

Termination Guarantee: Every operator either merges entities (reducing cell count) or splits a 1-cell at a finite locus and monotonically increases r. Because cell counts are finite, the loop is a finite lattice contraction that must reach a fixpoint where Invariants 1 and 2 hold everywhere.

3 Parameter-Free Cellular Extraction Build ∂₁, ∂₂, and ∂₃ without 2D UV p-curves Once Invariants 1 and 2 hold, extract the combinatorial matrices ∂1 , ∂2 , ∂3 directly in 3D space without relying on 2D (u, v) parameter loops:

1. ∂1 (Edge-to-Vertex Matrix): Directly read from the surviving active edges Ω^ (ej ): each edge exits its start ball vstart (−1) and enters its end ball vend (+1).

2. ∂2 (Face-to-Edge Matrix via Barrier Subdivision): For each original surface σk , the union of all incident vertex balls and edge tubes Bk = (⋃ Ω^ (vi ) ∪ ⋃ Ω^ (ej )) ∩ σk forms a fat, watertight barrier on the surface of width at least 2r(ej ) > 2r(fk ).

Partition σk ∖ Bk into connected components (using curved surface marching / adaptive triangulation restricted strictly outside the tubes Ω^ (ej ) and balls Ω^ (vi )).

Because the barrier has finite 3D thickness r(ej ) > r(fk ), the flood-fill cannot leak across edges or fail at UV singularities. Each connected component is a valid 2-cell fm , and the tube collars it touches define its column in ∂2 such that ∂1 ⋅ ∂2 = 0 holds automatically.

3. ∂3 (Volume-to-Face Matrix via Radial Tube Links): At the midpoint of every active edge tube Ω^ (ej ), intersect all incident active face sheets Ω^ (fm ) with the normal plane of the curve γj .

Because Invariant 1 guarantees that all face sheets are strictly disjoint outside Ω^ (ej ), their intersections with the normal circle of radius r(ej ) are strictly disjoint arcs separated by finite angles.

Sorting these disjoint arcs radially around ∂ Ω^ (ej ) requires zero ill-conditioned tangent math. Pairing adjacent faces in the radial cycle around every edge partitions R3 ∖ Ω^ (K) into closed 3-cells (shells/volumes), populating ∂3 such that ∂2 ⋅ ∂3 = 0.

4 Global Winding Classification & Algebraic Boundary Extraction Extract the Boolean solid via sparse matrix multiplication To classify which 3-cells sn ∈ C3 belong to the Boolean output:

1. Pick one interior sample point xn ∈ sn per 3-cell that is strictly outside all face sheets Ω^ (F).

2. Evaluate the Generalized Winding Number wA (xn ) and wB (xn ) with respect to the original input shells: 1 (y − xn ) ⋅ dSy wA (xn ) = ∮ 3 4π ∂A ∥y − xn ∥ Because xn is bounded away from all surface strata by at least r(f ), the integral is strictly well-conditioned and evaluates cleanly to 0 or 1.

3. Apply the Boolean predicate (e.g., wA ≥ 0.5 ∧ wB ≥ 0.5 for intersection) to form the binary selection vector c3 ∈ {0, 1}∣S∣ .

### 4. Compute the final B-rep boundary faces via sparse matrix-vector multiplication:

b2 = ∂3 ⋅ c3 Because ∂2 ⋅ b2 = (∂2 ∂3 )c3 = 0, the resulting boundary b2 is algebraically guaranteed to be a closed, watertight, manifold B-rep.

### 4. Two Practical Architectural Safeguards

To make this algorithm exceed Parasolid and OCCT in both precision and speed, pair the pipeline with two design rules: Design Rule How It Works Why It Matters In Stage 2, an edge radius r(ej ) = rgeom + ϵfit consists of Refinement Before intrinsic surface uncertainty rgeom plus spline approximation error Ensures vertex tolerances r(v ) only Inflation ϵfit . Before inflating a vertex ball r(vi ) to resolve a Collar grow when surfaces are truly tangenti or (Preventing Runaway Monotonicity or Tube Disjointness violation, check if ϵfit dominates grazing, preventing unnecessary Tolerances) r(ej ). If so, locally refine the intersection curve's knot vector to swallowing of tiny neighboring features. drive ϵfit → 0 first.

Completely eliminates the largest Never use 2D (u, v) p-curves during Stages 1–4. Generate source of topological failures in OCCT Passive Derived P- (u(t), v(t)) curves lazily after Stage 4 only for the surviving faces (2D loop self-intersection and seam/pole Curves in b2 , clamped inside the vertex balls. degeneracy), and avoids computing p- curves for discarded faces.

Interactive Exploration: Stratum Consolidation & Collar Monotonicity The simulation below demonstrates Stage 2 (Fixpoint Stratum Consolidation) on a grazing curve-curve intersection where approximation oscillation would break a traditional kernel. Adjust the grazing angle or trigger the consolidation steps to see how Invariant 1 (Stratum Disjointness) and Invariant 2 (Collar Monotonicity) automatically absorb micro-oscillations into the 0-stratum ball and guarantee well-conditioned radial ordering on the collar sphere: Key takeaway: Notice in Stage 2 and Stage 3 above that once the vertex ball Ω^ (v) inflates to enforce Collar Monotonicity, every micro-loop and tangency wiggle is completely contained inside the 0-stratum interior. On the boundary sphere ∂ Ω^ (v), the four exiting edge tubes are separated by a clean, well-conditioned radial angle—turning topological wiring into a trivial discrete sort around the sphere.

## User

im skeptical. you are treating things as a mathematical formalism, which is a good starting point. but we need to test this. what is we implemented this as set operations in 2d. the first issue you will hit is that some input will be generate (self-intersecting input,

for example). so, maybe the first algorithm to implement is shape-healing in 2d regardless, what i'm looking for is a plan where we build incremental utility.

## Response

You just hit on the most important practical test of this architecture: in this model, 2D Shape Healing and 2D Booleans are not two separate algorithms—they are the exact same code path.

In OCCT or Parasolid, ShapeUpgrade / ShapeFix is a massive, heuristic pre-processor bolted onto the front of the Boolean engine because the Boolean engine assumes clean, manifold inputs. If you build a separate 2D shape healer first and a 2D Boolean engine second, you already have two complex systems to verify.

Instead, recognize what a broken, self-intersecting, micro-gapped 2D input actually is: it is simply a bag of 0-strata (vertices) and 1- strata (edges) that currently violates Invariants 1 and 2.

Feed one dirty, self-intersecting 2D contour soup into the Consolidation + Winding pipeline → you have a 2D Shape Healer.

Feed two dirty 2D contour soups (tagged A and B ) into the exact same pipeline → you have a 2D Boolean Engine that heals the inputs and computes the set operation simultaneously.

Why 2D Degeneracies Vanish Without Special-Case Healers Before laying out the build plan, trace what happens to the three classic broken 2D inputs when you run the 4-stage pipeline in R2 :

### 1. Self-Intersecting Polygon (Bowtie / Figure-8):

Input: Two non-incident edges e1 , e2 cross in their interiors.

What happens: That crossing is a violation of Invariant 1 (dist(Ω^ (e1 ), Ω^ (e2 )) = 0). Stage 2 spawns a 0-stratum disk B(pnew , rv ) at the crossing and splits both edges into four half-edges exiting ∂B(pnew , rv ) at four distinct radial angles.

Stage 3 extracts two separate loop cells. Stage 4 evaluates the 2D winding number w(x) inside each loop (e.g., +1 and −1, or +1 and +2 for an overlapping loop). Applying the predicate ∣w(x)∣ ≥ 0.5 outputs two clean, manifold polygons meeting at a valid vertex.

### 2. Micro-Gaps and Sliver Edges:

Input: An almost-closed loop where endpoints miss by 10−4 , or a tiny sliver edge of length 10−5 from a bad DXF export.

What happens: If the gap is smaller than r(v1 ) + r(v2 ), the vertex disks overlap (violating Invariant 1) and merge into a single enclosing disk B(p∗ , r∗ ), sealing the gap. If an edge lies entirely inside B(p∗ , r∗ ), it violates Invariant 2 (zero crossings of ∂B ) and is contracted out of existence.

### 3. Dangling Whiskers / Unclosed Contours:

Input: A stray edge sticks out into free space or into the interior of a polygon, ending at a degree-1 vertex.

What happens: You don't even need a "whisker removal" pass. In Stage 3, when walking the radial ports around the graph, the same 2D face cell fk lies on both sides of the dangling edge. Therefore, in the sparse incidence matrix ∂2 , the column for face fk has +1 and −1 on that edge, summing to 0. The whisker cancels out algebraically during b1 = ∂2 ⋅ c2 .

The 5-Milestone Incremental Build Plan This roadmap is structured so that every milestone ships a standalone, useful tool, zero code is thrown away between milestones, and every stage has a deterministic fuzz-testing oracle you can run on millions of random degenerate inputs.

1 Milestone 1: 0-Stratum Disk Consolidation (Line Segments) Deliverable: CAD Wireframe / DXF / SVG Gap & Sliver Cleaner Build the core data structures (VertexDisk, EdgeTube for 2D line segments) and implement only the 0-Stratum Fixpoint Loop:

1. Disk-Disk Merge: While any two vertex disks B(pi , ri ) and B(pj , rj ) overlap (∥pi − pj ∥ ≤ ri + rj ), merge them into their minimal enclosing disk B(p∗ , r∗ ).

2. Micro-Edge Contraction: Any edge whose two endpoints now map to the same vertex disk—or whose geometry lies entirely inside that disk—is deleted.

3. Disk-Segment T-Junction Split: If a vertex disk B(pi , ri ) intersects the active tube Ω^ (ej ) of a non-incident segment ( dist(pi , ej ) ≤ ri + re ), split ej at vi and inflate ri ← dist(pi , ej ) + re + ϵ.

Immediate Utility: A soup-to-graph cleaner for messy 2D CAD imports (DXF/SVG). It snaps unclosed corners, absorbs zero- length micro-segments, and resolves T-junctions where one wall butts into another.

Automated Fuzz Oracle: Generate 100,000 random segment soups with jittered endpoints. Assert after convergence that: Every pair of vertex disks satisfies ∥pi − pj ∥ > ri + rj .

Every vertex disk and non-incident active segment satisfies dist(pi , Ω^ (ej )) > ri + re .

2 Milestone 2: 1-Stratum Segment Consolidation Deliverable: Planar Arrangement & Self-Intersection Resolver Add the 1-Stratum Operator to the Milestone 1 fixpoint loop to handle segment-segment interactions outside the vertex disks:

1. Transverse & Grazing Crossings: If two active segment tubes Ω^ (e1 ) and Ω^ (e2 ) intersect outside their endpoint disks, compute the minimum enclosing disk B(pnew , rnew ) of the capsule-capsule intersection diamond, split both edges, and push the new vertex disk back into the Milestone 1 queue.

2. Coincident / Overlapping Segments: Because Milestone 1 splits any T-junctions where an endpoint touches another segment's interior, two collinear overlapping segments are guaranteed to already share both endpoint disks va and vb . If two edges share (va , vb ) and their active tubes overlap, merge them into a single edge carrier and widen its tube re to bound both.

Immediate Utility: A rock-solid 2D Planar Arrangement Builder. Feed it overlapping architectural floorplan walls, self- crossing polylines, or stacked duplicate lines, and it outputs a strictly planar graph with zero crossings.

Automated Fuzz Oracle: Throw "starburst" torture tests (100 segments nearly intersecting at 0.001∘ angles around the origin) at the loop. Assert that all surviving truncated segments Ω^ (ej ) have pairwise distance strictly greater than zero.

3 Milestone 3: Radial Port Walk & Winding-Number Healer Deliverable: Complete 2D Shape Healer & Polygon Offset Cleaner Now that Invariants 1 and 2 hold on the 1D graph, implement Stages 3 and 4 for a single shape:

1. Radial Port Sort: For each vertex disk vi , each incident active edge ej crosses the circle ∂B(pi , ri ) at a single exact point.

Sort incident half-edges counter-clockwise by their exit angle θij = atan2(y − yi , x − xi ).

2. Face Loop Extraction (∂2 ): Permute half-edges (next = radial_cw_neighbor(twin(he))) to trace closed face cycles fk and populate the sparse matrix ∂2 ∈ {−1, 0, +1}∣E∣×∣F∣ . (Dangling edges automatically get 0 in ∂2 because they traverse +1 and −1 on the same face).

3. Winding Classification & Boundary Extraction: Pick an interior point xk inside each face fk , compute the exact 2D winding number w(xk ) = 2π1 ∑e Δθj (xk ) against the original input segments, select faces with ∣w(xk )∣ ≥ 0.5 (or j positive rule w(xk ) ≥ 0.5), and multiply b1 = ∂2 ⋅ c2 .

Immediate Utility: Production 2D Shape Healer & Raw Offset Loop Cleaner. To compute a 2D polygon offset (buffer), you just offset the raw segments/arcs outward, connect corners, and feed the self-intersecting raw offset loop directly into Milestone 3—it swallows all self-intersecting offset cusps and outputs the exact offset polygon.

Automated Fuzz Oracle: Algebraic check: Assert ∂1 ⋅ b1 == 0 (guarantees every output loop is closed and manifold).

Point-in-polygon check: Sample 1,000 random points outside the tolerance zones and assert the healed output winding number matches the raw input rule.

4 Milestone 4: Multi-Tag Winding Evaluation (~30 Lines of Code) Deliverable: Multi-Body 2D Boolean & CSG Engine Extend Milestone 3 to support arbitrary 2D Boolean operations without touching a single line of the geometric intersection or consolidation code:

1. Attach a bitmask or source list of (solid_id, sign) to each input edge. When two coincident edges merge in Milestone 2, concatenate their source lists.

2. In Stage 4, for each face sample point xk , evaluate wA (xk ) over A's edges and wB (xk ) over B 's edges.

3. Evaluate any Boolean expression (A & B, A | B, A - B, A ^ B) to populate c2 , and extract b1 = ∂2 ⋅ c2 .

Immediate Utility: A complete 2D Boolean Kernel that natively accepts self-intersecting, unhealed inputs A and B and produces guaranteed-manifold outputs in one pass.

Automated Fuzz Oracle: Verify Boolean identities across 50,000 random degenerate polygon pairs: ∂1 ⋅ b1 == 0 on all operations.

Area(A ∪ B) + Area(A ∩ B) ≈ Area(Heal(A)) + Area(Heal(B)) within the total area of the tolerance zones ∑ Area(Ω(ej )).

5 Milestone 5: Curved 1-Strata (Arcs & NURBS) + Collar Monotonicity Deliverable: Curved 2D CAD Sketch Kernel & 3D Face Trimmer Generalize EdgeTube from line segments to 2D circular arcs and rational Bézier/NURBS curves:

1. Collar Monotonicity Operator: A curved edge γj (t) can enter a vertex disk B(pi , ri ), bend sharply, and exit/re-enter ∂B(pi , ri ). Intersect γj (t) with the circle ∥x − pi ∥2 = ri2 . If there is more than one crossing or the radial derivative (γj (t) − pi ) ⋅ γj′ (t) ≤ 0, inflate ri to enclose the non-monotonic hook (or subdivide the curve if spline approximation error dominates).

2. Curve-Curve Tube Consolidation: Bound rational curves with fat bounding-polygon hierarchies (convex hull + re ).

Subdivide until either disjoint or enclosed inside a newly spawned 0-stratum disk.

Immediate Utility: Curved 2D CAD Kernel (handling tangent fillets, arc-line transitions, and exact NURBS silhouettes) AND the exact 2D Face-Splitting Engine you need when moving to 3D B-rep Booleans.

Minimal Data Model to Start Coding Milestone 1 Today You can fit the entire topological core of Milestones 1–4 into four flat arrays and two sparse matrices: Structure Representation Role in the Pipeline VertexDisk struct { Vec2 p; double r; bool alive; } 0-Stratum Ω^ (vi ) = B(pi , ri ). Grows monotonically during Stage 2 merges/engulfments.

Structure Representation Role in the Pipeline

struct { uint32 v0, v1; Segment2D seg; 1-Stratum Ω^ (ej ). Active interval [t0 , t1 ] is implicitly
EdgeTube double r; SmallVec<Tag> tags; bool the portion of seg outside B(pv0 , rv0 ) and

alive; } B(pv1 , rv1 ).

struct { uint32 edge_id; int8 dir; Stored per VertexDisk in Stage 3; sorted strictly by

CollarPort double angle_rad; } angle_rad where seg crosses ∂B(pi , ri ).

∂1 is ∥V ∥ × ∥E∥, ∂2 is ∥E∥ × ∥F ∥. Extracting any

ChainComplex2D CSRMatrix<int8> d1, d2; healed shape or Boolean is just b1 = d2 * c2.

Interactive Sandbox: Why Shape Healing & Booleans Are One Algorithm Use the interactive 2D workbench below to test the exact pipeline across three adversarial cases: (1) a self-intersecting bowtie with an unclosed micro-gap and a dangling whisker, (2) a raw polygon offset loop with self-swallowing cusps, and (3) a grazing sliver Boolean A ∪ B . Step through Stages 1–4 and adjust the initial vertex tolerance rv to see how the fixpoint loop and sparse matrix ∂2 heal and extract manifold boundaries without special-case branching: Try this in the sandbox above: On Preset 1 (Bowtie + Gap + Whisker), drag the Initial Vertex Tolerance rv slider below 6 px while viewing Stage 3 or Stage 4. When rv is smaller than half the gap, the left loop doesn't close—so instead of crashing or producing a broken non-manifold shell, its edges simply become a dangling chain whose +1 and −1 sides cancel in ∂2 , leaving only the valid closed right loop! The moment you slide rv ≥ 6 px, the gap disks touch, merge in Stage 2, and both halves of the bowtie lock in as closed manifold faces.

## User

I don't understand how matrix multiplication somehow solve the construction problem? What actual data types would this system use? It sounds like a lot of nice words, but you have failed to convey a practical programming intuition

## Response

Fair critique. "Chain complexes" and "sparse matrix multiplication" sound like you are supposed to pull in a linear algebra library like Eigen or BLAS. You don't. In actual systems code, there is no matrix object at all.

What I called "sparse matrix multiplication" is literally a 5-line for loop over a flat array of edges using two integer indices: left_region and right_region.

Let’s strip away the math terminology and look at the exact structs, arrays, and functions that construct a B-rep in 2D (and how the exact same pattern maps to 3D).

### 1. The Actual Data Types

Here is the entire data model in Rust-style pseudocode using flat arena arrays and integer handles:

type VertId = u32;
type EdgeId = u32;
type HalfEdgeId = u32;
type RegionId = u32;
struct VertDisk {

pos: Vec2, radius: f64,

// Sorted CCW list of half-edges leaving this vertex disk
outgoing_spokes: SmallVec<[HalfEdgeId; 8]>,

alive: bool,

}
struct Edge {

v0: VertId, v1: VertId, geom: Segment2D, // Later: enum Curve2D { Line, Arc, Nurbs } radius: f64, source_mask: u32, // Bit 0 = Solid A, Bit 1 = Solid B alive: bool,

}
// Every Edge `e` implicitly has two HalfEdges:
// he_forward = 2 * e (v0 -> v1)
// he_reverse = 2 * e + 1 (v1 -> v0)
struct HalfEdge {

origin: VertId, target: VertId,

next_in_loop: HalfEdgeId, // Next half-edge around the same 2D region region: RegionId, // Which 2D region lies to the LEFT of this half-edge in_final_output: bool, // Set during final boundary extraction

}
struct Region {

id: RegionId, sample_pt: Vec2, // Any point strictly inside this region keep: bool, // True if this region survives the Boolean / Heal

} Now look at how the math jargon from earlier maps 1-to-1 to those struct fields: The matrix ∂1 is just edge.v0 and edge.v1.

The matrix ∂2 is just the pair (half_edges[2*e].region, half_edges[2*e + 1].region)—i.e., the RegionId to the left of the edge and the RegionId to the right of the edge.

The vector c2 is just regions[r].keep (a single bool per region).

The "matrix multiplication" b1 = ∂2 ⋅ c2 is just comparing the two bools on either side of each edge:

for e in 0..edges.len() {
if !edges[e].alive { continue; }
let he_fwd = 2 * e;
let he_rev = 2 * e + 1;
let left_keep = regions[half_edges[he_fwd].region as usize].keep;
let right_keep = regions[half_edges[he_rev].region as usize].keep;
// THIS IF-ELSE IS THE "MATRIX MULTIPLICATION":
if left_keep && !right_keep {
// Region on left is INSIDE, region on right is OUTSIDE -> keep forward half-edge
half_edges[he_fwd].in_final_output = true;

} else if !left_keep && right_keep {

// Region on right is INSIDE, region on left is OUTSIDE -> keep reverse half-edge
half_edges[he_rev].in_final_output = true;
}
// If both are true (interior seam) or both are false (exterior/whisker),
// NEITHER half-edge is marked true. The edge disappears!
}

2. Why Does That Solve the Construction Problem?

To see why that 10-line loop is such a big deal, contrast it with how traditional kernels fail.

Why Traditional Kernels Break (Direct Edge Classification) In a traditional kernel, after intersecting edges, you try to classify edges directly ("Is edge e on the boundary of A ∪ B ?"). Suppose three edges e1 , e2 , e3 meet near a tangency or messy vertex V . You run floating-point point-on-edge / normal checks on each edge independently: e1 decides it is on the boundary (entering V ). e2 hits a floating-point ambiguity near the tolerance threshold and decides it is inside solid A, so it gets discarded. e3 also decides it is inside solid A and gets discarded.

Now you try to stitch the surviving edges into closed polygon loops. You arrive at vertex V along e1 , and there is no outgoing edge.

Your loop builder crashes with an unclosed wire / non-manifold shell error, and you have to write heuristic "shape healing" code to figure out whether e2 or e3 should be resurrected.

Why Region-Diffing Cannot Produce an Open Loop In the region-diffing approach, you never classify edges. You only set a boolean .keep = true/false on 2D regions, and an edge survives if and only if left_keep != right_keep.

Look at what happens around any vertex V with any number of incident spokes sorted counter-clockwise:

Spoke 1 (Edge B)
|
Region 1 | Region 0

(keep=true) | (keep=false)

|

---------------+--------------- Spoke 0 (Edge A)

|
Region 2 | Region 3

(keep=true) | (keep=false)

|
Spoke 2 (Edge C)

Walk counter-clockwise in a full circle around vertex V , stepping from region to region across the spokes: Region 0 (false) -> Region 1 (true) -> Region 2 (true) -> Region 3 (false) -> Region 0 (false) Crossing Spoke 0 (false -> true): spawns an outgoing boundary half-edge leaving V .

Crossing Spoke 1 (true -> true): both sides are true, so Spoke 1 is an internal seam and is deleted.

Crossing Spoke 2 (true -> false): spawns an incoming boundary half-edge entering V .

Crossing Spoke 3 (false -> false): both sides are false, so Spoke 3 is deleted.

Because the wedges around vertex V form a closed circle, as you walk around V from Region 0 back to Region 0, the boolean value .keep must flip from false to true the exact same number of times it flips from true to false.

That means: Even if your winding-number classifier returned random booleans for region.keep, every vertex in the output is still 100% guaranteed to have an equal number of incoming and outgoing half-edges. It is impossible to ever generate an open loop or a dangling edge.

Look at what happens to a dangling whisker sticking out into space: The region to the left of he_fwd is Region 0.

The region to the left of he_rev (the other side of the exact same whisker) is also Region 0.

In the loop above, left_keep is regions[0].keep and right_keep is regions[0].keep. Since left_keep == right_keep, neither half-edge is selected. The whisker vanishes without a single line of special-case whisker-detection code.

### 3. The Entire 4-Step Pipeline in Concrete Code

Now let's connect the geometry (the tolerance disks) to the topology (the half-edges and regions). The whole system is just four sequential functions:

### Function 1: consolidate_disks_and_segments(verts, edges)

This is the while (changed) loop that enforces the tolerance invariants so that Function 2 doesn't have to deal with degeneracies:

fn consolidate_disks_and_segments(verts: &mut Vec<VertDisk>, edges: &mut Vec<Edge>) {
loop {
let mut changed = false;
// 1. Merge any two vertex disks that touch or overlap
for i in 0..verts.len() {
for j in (i + 1)..verts.len() {
if !verts[i].alive || !verts[j].alive { continue; }
let dist = (verts[i].pos - verts[j].pos).length();
if dist <= verts[i].radius + verts[j].radius {
// Merge j into i: compute minimum enclosing disk of both
verts[i] = min_enclosing_disk(verts[i], verts[j]);
verts[j].alive = false;
remap_edge_endpoints(edges, j as u32, i as u32);
changed = true;
}
}
}
// 2. Kill any micro-edge that collapsed into a single vertex disk
for e in edges.iter_mut() {
if e.alive && e.v0 == e.v1 {
e.alive = false;
changed = true;
}
}
// 3. Split any edge whose active segment (outside its endpoint disks)
// gets touched by a non-incident vertex disk
for v_id in 0..verts.len() {
if !verts[v_id].alive { continue; }
for e_id in 0..edges.len() {
if !edges[e_id].alive { continue; }
if edges[e_id].v0 == v_id as u32 || edges[e_id].v1 == v_id as u32 { continue; }
let dist = dist_point_to_active_segment(verts[v_id].pos, &edges[e_id], verts);
if dist <= verts[v_id].radius + edges[e_id].radius {
// Inflate disk to swallow the edge tube cross-section and split edge in two
verts[v_id].radius = dist + edges[e_id].radius + 1e-9;
split_edge_at_vertex(edges, e_id, v_id as u32);
changed = true;
}
}
}
// 4. Intersect or merge active segments outside vertex disks
for e1 in 0..edges.len() {
for e2 in (e1 + 1)..edges.len() {
if !edges[e1].alive || !edges[e2].alive { continue; }
if let Some(hit) = intersect_active_tubes(&edges[e1], &edges[e2], verts) {
match hit {
TubeHit::SharedEndpointsCoincident { combined_radius } => {
// Both edges already connect the exact same two vertex disks!
edges[e1].radius = combined_radius;
edges[e1].source_mask |= edges[e2].source_mask;
edges[e2].alive = false;
}
TubeHit::LocalizedCrossing { center, radius } => {
// Spawn a new vertex disk at the crossing and split both edges
let new_v = spawn_vertex_disk(verts, center, radius);
split_edge_at_vertex(edges, e1, new_v);
split_edge_at_vertex(edges, e2, new_v);
}
}
changed = true;
}
}
}
if !changed { break; }
}
}

### Function 2: build_planar_half_edges(verts, edges)

Because Function 1 finished, no two edges touch outside the vertex disks, and every incident edge exits its vertex disk circle at a clean point. Now we wire the next_in_loop pointers:

fn build_planar_half_edges(verts: &mut [VertDisk], edges: &[Edge]) -> Vec<HalfEdge> {
let mut half_edges = vec![HalfEdge::default(); edges.len() * 2];
// 1. Create forward and reverse half-edges for each surviving edge
for (e_id, e) in edges.iter().enumerate() {
if !e.alive { continue; }
let he_fwd = (2 * e_id) as u32;
let he_rev = (2 * e_id + 1) as u32;
half_edges[he_fwd as usize] = HalfEdge { origin: e.v0, target: e.v1, ..Default::default() };
half_edges[he_rev as usize] = HalfEdge { origin: e.v1, target: e.v0, ..Default::default() };
verts[e.v0 as usize].outgoing_spokes.push(he_fwd);
verts[e.v1 as usize].outgoing_spokes.push(he_rev);
}
// 2. Sort outgoing_spokes CCW around each vertex disk by the angle where
// the segment exits the circle |x - v.pos| == v.radius
for v in verts.iter_mut() {
if !v.alive { continue; }
v.outgoing_spokes.sort_by(|&he_a, &he_b| {
let angle_a = exit_angle_on_disk(v, &edges[(he_a / 2) as usize]);
let angle_b = exit_angle_on_disk(v, &edges[(he_b / 2) as usize]);

angle_a.partial_cmp(&angle_b).unwrap() });

// 3. Wire next_in_loop: a half-edge entering `v` (which is `twin(spoke[i])`)
// turns left into the immediately preceding CCW spoke around `v`
let k = v.outgoing_spokes.len();
for i in 0..k {
let outgoing = v.outgoing_spokes[i];
let incoming = outgoing ^ 1; // Bitwise XOR flips 2*e <-> 2*e+1 (the twin!)
let next_outgoing_cw = v.outgoing_spokes[(i + k - 1) % k];
half_edges[incoming as usize].next_in_loop = next_outgoing_cw;
}
}

half_edges

} Notice why angle_a.partial_cmp(&angle_b) cannot fail here: if two spokes had nearly identical angles on the boundary of the disk v.radius, their edge tubes of radius e.radius would have overlapped outside the vertex disk—which Function 1 already merged or engulfed! Every spoke is guaranteed to be separated by at least Δθ ≥ 2 arcsin(re /rv ).

### Function 3: build_regions_and_classify(half_edges, raw_input_edges)

Now we walk the next_in_loop cycles to assign he.region and evaluate the winding number once per region:

fn build_regions_and_classify(
half_edges: &mut [HalfEdge],

edges: &[Edge], raw_input: &[RawEdge], op: BoolOp,

) -> Vec<Region> {
let mut visited = vec![false; half_edges.len()];
let mut regions = Vec::new();
for he_start in 0..half_edges.len() {
if !edges[he_start / 2].alive || visited[he_start] { continue; }
let region_id = regions.len() as u32;
let mut curr = he_start as u32;
while !visited[curr as usize] {
visited[curr as usize] = true;
half_edges[curr as usize].region = region_id;
curr = half_edges[curr as usize].next_in_loop;
}
// Pick one interior sample point for this region and compute winding numbers
let sample_pt = pick_interior_point_for_cycle(he_start as u32, half_edges, edges);
let w_a = winding_number(sample_pt, raw_input, SOLID_A);
let w_b = winding_number(sample_pt, raw_input, SOLID_B);
let keep = match op {
BoolOp::HealA => w_a.abs() >= 0.5,
BoolOp::Union => (w_a >= 0.5) || (w_b >= 0.5),
BoolOp::Intersect => (w_a >= 0.5) && (w_b >= 0.5),
BoolOp::Difference => (w_a >= 0.5) && (w_b < 0.5),
};
regions.push(Region { id: region_id, sample_pt, keep });
}

regions

}

### Function 4: extract_final_loops(verts, edges, half_edges, regions)

Finally, we run the left_keep != right_keep filter from Section 1, and link the surviving boundary half-edges at each vertex:

fn extract_final_loops(

verts: &[VertDisk], edges: &[Edge],

half_edges: &mut [HalfEdge], regions: &[Region], ) {

// 1. Mark which half-edges are on the boundary between keep=true and keep=false
for e_id in 0..edges.len() {
if !edges[e_id].alive { continue; }
let he_fwd = 2 * e_id;
let he_rev = 2 * e_id + 1;
let left_keep = regions[half_edges[he_fwd].region as usize].keep;
let right_keep = regions[half_edges[he_rev].region as usize].keep;
half_edges[he_fwd].in_final_output = left_keep && !right_keep;
half_edges[he_rev].in_final_output = !left_keep && right_keep;
}
// 2. Re-link `next_in_loop` ONLY among the surviving `in_final_output` half-edges!
for v in verts {
if !v.alive { continue; }
// Collect only the spokes around `v` that survived (either incoming or outgoing)
// Because of the circular bit-flip invariant, they strictly alternate around `v`:
// Incoming -> Outgoing -> Incoming -> Outgoing!
let mut active_spokes = SmallVec::<[HalfEdgeId; 4]>::new();
for &out_he in &v.outgoing_spokes {
let in_he = out_he ^ 1;
if half_edges[out_he as usize].in_final_output {

active_spokes.push(out_he); // Leaves v

} else if half_edges[in_he as usize].in_final_output { active_spokes.push(in_he); // Enters v

}
}
// Pair each incoming half-edge with the immediately preceding CCW outgoing half-edge
let m = active_spokes.len(); // Guaranteed to be an even number (0, 2, 4...)!
for i in 0..m {
let he = active_spokes[i];
if half_edges[he as usize].target == (v as *const _ as u32) { // Incoming to v
let next_out = active_spokes[(i + m - 1) % m];
half_edges[he as usize].next_in_loop = next_out;
}
}
}
}

How This Exact Same Pattern Maps to 3D B-Reps When you move from 2D to 3D, the data structures and loop remain identical—everything just shifts up one dimension: Concept In 2D (Polygon Healer / Boolean) In 3D (Solid B-Rep Healer / Boolean) Highest Cell (Region) 2D Face Region (RegionId) 3D Volume Cell (VolumeId)

Boundary Element (Edge / Edge with 2 HalfEdges (he_fwd, he_rev) Face with 2 CoFaces (cf_front, cf_back) Face) Adjacency Fields he_fwd.region (left), he_rev.region (right) cf_front.volume (front), cf_back.volume ("Matrix") (back) Sort outgoing_spokes by angle around each Sort incident CoFaces by angle around each Radial Sort Hub Vertex Disk Edge Tube Keep he_fwd iff keep[left] && Keep cf_front iff keep[front] && Boundary Extraction Loop !keep[right] !keep[back]

Does seeing it as left_keep && !right_keep on HalfEdge / CoFace structs and the alternating Incoming -> Outgoing ring around each vertex clarify what the "algebraic" step actually does in code?

## User

I feel like I'm missing something. It seems like something about this is just a standard set operation algorhm on polygons.

It looks just like off the shelf algorithms. Why is it better? Be specific

## Response

You are completely right to spot the resemblance. At a 2D level, the region-diffing and winding-number logic shares DNA with robust polygon clippers (like Angus Johnson’s Clipper library).

So why call this out as a superior architecture for a B-rep kernel instead of just saying "use an off-the-shelf polygon clipper"?

The reason is that off-the-shelf polygon clippers work by cheating in a way that breaks the moment you move to 3D CAD. When you scale this architecture up to a true 3D B-rep kernel (surfaces, NURBS, trimmed patches, and complex assemblies), off-the-shelf polygon algorithms fail completely.

Here are the four specific reasons why this framework outperforms both off-the-shelf polygon clippers and traditional CAD kernels (Parasolid/OCCT).

### 1. The Integer-Scaling Trap (Why Clipper Doesn't Scale to CAD)

Off-the-shelf robust 2D polygon clippers achieve robustness through fixed-point integer coordinate scaling: they take floating-point inputs, multiply them by a massive factor (e.g., 109 ), cast them to 64-bit integers, and perform all intersection math using exact integer arithmetic (like Shewchuk's predicates).

Why it works in 2D: You can snap line segments to a grid of integers.

Why it fails in a real CAD kernel: You cannot integer-scale a NURBS surface, an implicit algebraic curve, or a trimmed B-rep face. NURBS surface intersections yield algebraic root-finding equations where coefficients are irrational floating-point numbers. If you truncate them to an integer grid, your surfaces tear, and your face trims no longer match your edge curves.

The fix here: This architecture does not rely on integer coordinate scaling. It works directly with floating-point geometry by wrapping every entity in an explicit metric tolerance zone (Ω) whose radius r accounts for floating-point rounding error dynamically.

### 2. Reactive Patching vs. Proactive Metric Engulfment (Beating OCCT & Parasolid)

Traditional CAD kernels (Parasolid, OCCT) use reactive healing. They run floating-point intersection math, discover that two edges missed each other by 10−6 due to numerical drift, and then spawn complex heuristics (e.g., "Edge Stitcher", "Face Welder", "Sewing Tolerance") to patch the resulting cracks.

Because every patch introduces new topological exceptions, the codebase explodes into millions of lines of special-case code (OCCT's BOPAlgo module is notorious for this).

The difference here: This architecture uses proactive metric engulfment. Stage 2's fixpoint loop doesn't try to "patch" a crack after it breaks the topology. It runs a strict lattice contraction where vertex disks B(p, r) and edge tubes Ω(e) actively expand to swallow any ambiguity before a single half-edge or region is created.

By the time topology is built, the input has been mathematically proven to be clean. There is no "shape healing" pass because the intersection algorithm is the shape healing algorithm.

### 3. Geometric Edge Tracing vs. Algebraic Region Classification

Traditional clippers and CAD kernels build output boundaries by geometric edge tracing. When two edges intersect, the algorithm computes the intersection point, inserts it into both edge data structures, and then "traces" a path from intersection to intersection by evaluating tangent vectors and turning left/right at vertices.

Where edge tracing fails: Near a tangency, a grazing contact, or a micro-sliver, a tiny floating-point error in the tangent vector causes the tracer to jump to the wrong branch of an intersection graph. Result: an open loop, a self-intersecting face, or a dropped sliver.

The difference here: You never trace edges. Edges are passive geometry. The algorithm classifies 2D regions (or 3D volume cells) first using winding numbers, and then extracts boundary half-edges via the discrete filter: Keep Half-Edge ⟺ left_region.keep ∧ ¬right_region.keep Because the region classification is global and robust, and the half-edge linking is strictly constrained by the alternating incoming/outgoing ring around each vertex, it is geometrically impossible for the tracer to get lost or generate an open loop.

### 4. Direct Generalization to 3D NURBS and Trimmed Surfaces

If you take an off-the-shelf 2D polygon clipper, you cannot lift it into 3D. You cannot sort 3D faces around a 3D vertex using 2D winding numbers or simple angle sorting.

The Stratified Tolerance Complex framework scales directly from 2D to 3D because its primitives are dimension-agnostic:

1. 0-Strata are vertex balls B(p, r) in R2 or R3 .

2. 1-Strata are line/curve tubes Ω(e) in R2 or R3 .

3. 2-Strata are polygon segments in R2 , or curved face sheets σ(u, v) with thickness r(f ) in R3 .

4. 3-Strata are volume cells in R3 classified by 3D generalized winding numbers.

By implementing this first in 2D, you aren't writing a "polygon library"—you are building the exact topological engine (flat arena arrays, fixpoint tolerance convergence, half-edge ring wiring, and sparse matrix boundary extraction) that will handle trimmed NURBS patches in 3D without changing its core control flow.

## User

Two issues that seem non-trivial 1. How do we compute the winding number in a region robustly? Couldnt this be a source of incorrectness? 2. How do we determine initial tolerances for the input?

## Response

You have pinpointed the exact two numerical minefields of any geometric kernel. Let's tackle them one by one with concrete, production-grade solutions.

### Issue 1: Robust Winding Number Computation

Why It Can Fail If you pick a random interior sample point x in a region and compute the winding number by casting a ray or summing angles against all edges in the model, near-misses will destroy you. If x happens to lie 10−8 away from a line segment, floating-point cancellation in the angle formula or ray-segment intersection will yield the wrong sign, flipping the winding number from 1 to 0 (or vice versa). Your region classification flips, and the Boolean operation punches a hole through a solid.

The Robust Solution: The ϵ-Perturbed Pole / Medial Axis Pole Strategy You do not pick an arbitrary random point. You compute the Pole of the region (the point inside the face that is furthest away from all bounding edges).

1. Extract the Face Loop: You already have the closed half-edge loop for the region from your half-edge wiring step.

2. Constrained Delaunay Triangulation (CDT): Triangulate the interior of the polygon loop.

3. Find the Pole: The circumcenter of the triangle in the CDT that is furthest from its bounding edges is your sample point x. By definition, x is locked in the deepest interior pocket of the region, maximizing dist(x, ∂Region).

4. Exact Predicates for Winding Number: When summing angles or casting a ray from this maximally interior pole x to evaluate w(x) = 2π1 ∑ Δθi , evaluate the cross products using Shewchuk’s adaptive-precision floating-point arithmetic (or double- double / quad-precision float arithmetic). Because x is far from the boundary, even standard double precision is well above machine epsilon; when near a boundary, exact arithmetic resolves any sign ambiguity.

### Issue 2: Determining Initial Tolerances for the Input

Why It’s Tricky If you set the initial vertex tolerance r(v) and edge tolerance r(e) too small (e.g., 10−12 ), floating-point rounding noise will cause valid intersections to miss each other, creating accidental microscopic gaps. If you set them too large (e.g., 10−2 ), unrelated features will swallow each other, deforming your input geometry.

The Robust Solution: Multi-Scale Tolerancing with Geometric Bounding You never use a single global epsilon. Every input entity gets an initial tolerance derived deterministically from three factors: Machine Precision, Coordinate Bounding Scale, and Input Source Resolution.

Initial Tolerance r0 = max (ϵmachine ⋅ Lbbox , α ⋅ ˉledge , τuser )

Where:

1. Lbbox (Bounding Box Scale): The diagonal length of the entire input geometry's bounding box. If coordinates range in [0, 1000], Lbbox ≈ 1414. Multiplying by machine epsilon (≈ 2.22 × 10−16 ) gives the absolute baseline floating-point noise floor (≈ 3 × 10−13 ).

2. ˉledge (Average Edge Length): The mean length of all input segments/edges. Set α ≈ 10−4 to 10−5 . This ensures that micro- gaps smaller than 0.01% of an edge's length are automatically treated as touching and snapped together by Stage 2.

3. τuser (Source File Tolerance): If importing from a STEP, DXF, or SVG file, read the header's declared linear tolerance (e.g., 10−3 for CAD, 10−1 for messy architectural floorplans). If no header exists, default to 10−6 .

How Tolerances Propagate Through the Pipeline Initial Vertices: r(vi ) = r0 Initial Edges: r(ej ) = r0 + ϵfit (where ϵfit is 0 for straight line segments, and the maximum curve-to-chord deviation for arcs or NURBS).

During Stage 2 Fixpoint Merging: When two vertex disks B(pi , ri ) and B(pj , rj ) merge, the new enclosing disk's radius r∗ is computed strictly to contain both original disks plus the distance between their centers: ∗ ∥pi − pj ∥ + ri + rj r = 2 This guarantees that no geometric information from the original inputs is ever lost or pushed outside the tolerance zone.

The tolerance zone monotonically grows to swallow uncertainty, preserving mathematical inclusion.

## User

Sorry isn't the tolerance depending on the distance from the origin?

## Response

That is a sharp observation, and it touches on one of the most insidious traps in floating-point geometry: coordinate scale sensitivity.

If you use a single global absolute tolerance (like ϵ = 10−7 ) across your entire kernel, you are walking into a numerical disaster the moment your geometry moves away from the origin.

Here is why that happens, why absolute tolerances fail at large coordinates, and how a robust kernel handles scale.

The Problem: Floating-Point Density and Coordinate Scale In IEEE 754 double-precision floating-point arithmetic, numbers are not spaced evenly on a ruler; they are distributed logarithmically (like an optical encoder or a slide rule).

Near 1.0, the distance between adjacent representable numbers (machine epsilon ϵmach ) is ≈ 2.22 × 10−16 .

Near 1, 000, 000.0 (106 ), the spacing between adjacent floats degrades by a factor of 220 , jumping to ≈ 1.19 × 10−10 .

If your CAD model is sitting at coordinates (x = 100, 000, y = 50, 000)—common in civil engineering, GIS, or large assembly modeling—and you use an absolute tolerance of 10−7 to test if two points are coincident (∥p1 − p2 ∥ ≤ 10−7 ), the computer literally cannot represent numbers that close together at that scale. The distance underflows to zero or rounds unpredictably, causing your vertex disk merges to tear open or lock up.

The Solution: Relative Tolerances and Bounding-Box Scaling To make your B-rep kernel invariant to where objects are placed in space, tolerances must never be absolute fixed constants. They must scale dynamically with the local coordinate magnitude or the bounding box size of the active geometry.

A production-grade kernel splits tolerances into three layers:

1. Model-Level Bounding Scale (Lbbox ): When you load or create a shape, compute its axis-aligned bounding box diagonal Lbbox .

If the model is centered at the origin spanning 10 units, Lbbox ≈ 14. If it is a bridge centered at (100, 000, 50, 000) spanning 2, 000 units, Lbbox = 2, 000. The baseline tolerance r0 is always scaled relative to this box: r0 = ϵmach ⋅ Lbbox ⋅ K (where K is a safety multiplier, typically 103 to account for accumulation in intersection loops). At 106 coordinates, your tolerance automatically widens from 10−7 to a safe 10−4 , matching the loss of floating-point precision.

2. Distance-Dependent Local Tolerance (r(v)): When two vertex disks B(pi , ri ) and B(pj , rj ) merge or propagate, their radii are governed by the distance of pi and pj from the origin (or local centroid). Using a relative error bound: r(vi ) = max (r0 , ULP(∣pi ∣) ⋅ scale_factor) where ULP(x) (Unit in the Last Place) measures the exact spacing of floating-point numbers at coordinate x.

3. The Ultimate Fix: Localized Coordinate Centering (Model Normalization) The cleanest architectural defense used in robust geometric engines is to never perform intersection math in global world coordinates if the coordinates are large.

When you run Stage 1 through Stage 4 on a cluster of intersecting edges or a local face patch, translate the local working bounding box center to (0, 0) (or use exact multi-precision arithmetic like GMP/MPFR for coordinate transforms).

By keeping local working coordinates near the origin, you lock ϵmach at its highest possible precision (10−16 ), and only apply the scaled relative tolerance when projecting back to world space.

## User

Please check these answers (argon)

## Response

It looks like the questions or answers didn't come through—did you mean to paste a block of text or attach an image/screenshot regarding argon (whether that's chemistry/physics properties or something else)?

Drop the text or image in your next message and I'll verify them step-by-step right away.

## User

I mean the last answers form the previous agent

## Response

Aha — "jargon" got autocorrected to "argon"! You're asking me to audit the previous responses in this thread and fact-check the claims made about:

1. How Constrained Delaunay Triangulation (CDT) + Shewchuk predicates solve region winding numbers.

2. How initial tolerances (r0 ) and floating-point distance from the origin work.

3. Whether the 2D Stratified Tolerance Complex / Region-Diffing architecture actually holds up under scrutiny or if the previous responses hand-waved hard geometric problems away.

Let's do a blunt, rigorous audit of the previous answers—separating what is genuinely sound systems architecture from what was hand- waved, overstated, or subtly flawed.

### 1. Audit of the "Winding Number via CDT Pole" Answer

What the previous answer claimed: "Triangulate the interior of the polygon loop using Constrained Delaunay Triangulation (CDT). The circumcenter of the triangle furthest from its bounding edges is your sample point x... evaluate cross products using Shewchuk's adaptive- precision arithmetic."

The Verdict: Half-true, and hides a circular dependency + a curved-edge trap.

Flaw A: The Circumcenter of an Obtuse Triangle Can Lie OUTSIDE the Region. The circumcenter of a Delaunay triangle corresponds to a Voronoi vertex (medial axis node), which is great for acute triangles, but for an obtuse triangle, the circumcenter lies outside the triangle—and in a thin, curved, or concave sliver region, it can easily lie completely outside the region you are trying to classify!

The Actual Fix: You never use the circumcenter. For a triangle (A, B, C) produced by triangulation, you use the incenter (the intersection of angle bisectors, weighted by edge lengths: aA+bB+cCa+b+c ) or the centroid A+B+C3 , which are mathematically guaranteed to lie strictly inside the triangle. Even simpler: you don't need a full CDT at all! A simple ear- clipping step that finds one valid ear triangle (vprev , vcurr , vnext ) whose interior contains no other vertices of the loop gives you an interior point in O(n) time without building a Delaunay mesh.

Flaw B: What happens when edges are curves (Arcs / NURBS)? A straight-line ear or CDT between vertices of a curved loop might cross a curved edge that bows inward.

The Actual Fix: Because Stage 2 already guarantees that all active edge tubes Ω^ (ej ) are strictly disjoint outside the vertex disks, you don't even need to triangulate the whole region to find a sample point! Walk to the midpoint of any active edge ej , and step sideways into the left/right region along the normal vector by a distance of r(ej ) + 12 dclear , where dclear is the minimum distance from that midpoint to the nearest other active edge tube. Because Invariant 1 holds, dclear > 0 is guaranteed. That point is strictly inside the region, bounded away from all boundaries, and works identically for lines, NURBS curves, and 3D curved surfaces.

Flaw C: Winding Number vs. Graph Propagation Cost. Evaluating the winding number against all original input segments for every region is O(R ⋅ E), which is fine for 100 edges but slow for 100,000 edges.

The Actual Fix: You only evaluate the global winding number once per connected component (or at the unbounded exterior face where w = 0), and then propagate winding numbers across the dual graph in O(1) per edge: crossing an edge from Solid A simply increments/decrements wA by ±1. If you want redundancy against bad inputs (where w might not be an integer), you evaluate the integral at a few poles and average/vote across the dual graph.

### 2. Audit of the "Initial Tolerance & Distance from Origin" Answers

What the previous answers claimed: "Initial Tolerance r0 = max(ϵmachine ⋅ Lbbox , α ⋅ ˉledge , τuser )... Multiply Lbbox by ϵmach ... Or use local coordinate centering."

The Verdict: The previous answer actually got caught by your follow-up question, and its formula for r0 had two bugs.

Bug A: Using α ⋅ ˉledge (Average Edge Length) is Dangerous. Suppose you have a large structural plate (10 m long edges) with a tiny precision alignment pin (0.5 mm chamfer edges). The average edge length ˉledge is several meters. Setting r0 = 10−4 ⋅ ˉledge sets the initial vertex tolerance to ∼ 0.5 mm, which instantly swallows and destroys the entire chamfer feature before a single intersection is even run!

The Actual Rule: Initial tolerances on clean input geometry must never scale with average edge length. They should only reflect numerical representation uncertainty (ULP of the coordinates) plus any explicitly declared import gap tolerance τimport .

Bug B: Lbbox vs. Distance from Origin (∥p∥∞ ). As you rightly pointed out, a 1 m × 1 m box located at (106 , 106 ) has Lbbox = 2 ≈ 1.414, so ϵmach ⋅ Lbbox ≈ 3 × 10−16 . Yet the actual floating-point grid spacing (ULP) at 106 is 220 ⋅ ϵmach ≈ 1.19 × 10−10 ! So scaling by Lbbox alone completely fails when a small object is far from the origin.

The Actual Formula: For any coordinate p = (x, y, z), the local floating-point noise floor is proportional to the magnitude of the coordinate itself: ϵcoord (p) = K ⋅ ϵmach ⋅ max(∣x∣, ∣y∣, ∣z∣, 1.0) Better yet, shifting the working bounding box center C = 12 (pmin + pmax ) to (0, 0, 0) at the very start of the operation (a single subtraction p′ = p − C on the input vertices) reduces max(∣x′ ∣, ∣y′ ∣, ∣z′ ∣) to 1 Lbbox .

2

### 3. Audit of the Overall "Stratified Tolerance Complex" Proposal

Now let's look at the big picture: Is the 4-stage architecture (Disk/Tube Consolidation → Radial Port Sort → Region Classification → left_keep != right_keep) actually better than OCCT/Parasolid, or is there a hidden catch?

Here is the honest breakdown of what is rock-solid and what the real engineering bottleneck is: What is 100% Solid (and genuinely beats OCCT):

1. Boundary extraction via left_keep != right_keep (Algebraic Chain Complex): This part is not hype. In OCCT (BOPAlgo_WireEdgeSet / BOPAlgo_BuilderFace), face loops are assembled by picking edges that were classified as "ON" or "IN" and trying to chain them head-to-tail. When an intersection is grazing, one edge gets classified "IN" and an adjacent micro- edge gets classified "OUT", leaving a dangling open wire. By classifying regions/volumes first and defining surviving boundary elements strictly as the interface between keep = true and keep = false, it is mathematically impossible to output a non-closed loop or non-manifold shell.

2. Sorting spokes on the Vertex Disk boundary ∂B(pi , ri ) instead of using tangents at pi : This is also a genuine algorithmic win. At a near-tangency between two curves, the tangent vectors at the intersection point pi are nearly identical, making tangent cross-products ill-conditioned. Evaluating the cyclic order where the curves exit the finite circle ∂B(pi , ri ) at distance ri away from the tangency turns an ill-conditioned derivative test into a well-separated positional angle test.

The Hidden Catch (What the previous answers made look too easy): The hardest problem in this entire architecture is Runaway Tolerance Dominoes (Cascading Ball Inflation) in Stage 2.

Look at what can happen in the naive consolidate_disks_and_segments loop if you aren't careful: Suppose you have a dense mesh or a finely tessellated curve where vertices are spaced 0.01 mm apart.

Two grazing curves intersect at a 0.5∘ angle. To swallow the grazing intersection diamond of their tubes (radius re = 10−4 ), the new vertex disk B(pnew , rnew ) must inflate to radius rnew ≈ re / sin(0.25∘ ) ≈ 0.023 mm.

Because 0.023 mm > 0.01 mm, that newly inflated disk touches two neighboring vertices, merges with them, grows to 0.04 mm, touches another edge, inflates again, and snowballs across your model.

Parasolid and OCCT suffer from this exact disease too—in OCCT, vertices inflate during PaveFiller until they swallow whole faces, causing the Boolean to fail.

How You Actually Prevent the Snowball in Code: To prevent cascading inflation in Stage 2, you cannot just use a single round ball B(pi , ri ) to enclose a long, skinny grazing intersection diamond. You have two specific mechanisms:

1. Split Grazing Overlaps into a Coincident Interval + Two Small Endpoint Balls: When two tubes Ω(e1 ) and Ω(e2 ) cross at a shallow angle θ ≪ 1, their intersection region is a long, narrow diamond of length L ≈ 2re / sin(θ/2) and width 2re . Instead of wrapping that entire long diamond in a giant circular ball of radius L/2, you place two small vertex balls of radius ∼ 2re at the entry and exit tips of the diamond, and merge the two curves into a single shared coincident edge along the middle interval!

That keeps every vertex ball tiny (O(re ) instead of O(re /θ)) and completely stops the snowball effect.

2. Hard Cap + Subdivide: For curved edges (NURBS), if the approximation tube re = rgeom + ϵfit causes an overlap, subdivide the curve to shrink ϵfit before allowing any vertex disk to grow.

Summary Verdict on the Previous Answers Topic from Previous Verdict What to Change in Your Implementation Answers 4-Stage Architecture & left_keep != Solid Keep as-is. This genuinely eliminates non-manifold / open-loop bugs by construction. right_keep Sorting spokes on ∂B(p, r) instead of Solid Keep as-is. Greatly improves robustness on curved/tangent intersections. tangents Shallow grazing A shallow crossing creates a long diamond; wrapping it in one circle inflates rv ∝ re / sin θ intersection → single Flawed and triggers snowball merges. Replace shallow crossings with two small endpoint balls ( enclosing ball rv ∼ 2re ) + a merged coincident interval between them.

Winding sample point via Circumcenters of obtuse triangles can fall outside the polygon. Use an ear-triangle incenter CDT circumcenter Flawed or step off an active edge midpoint by 12 dclear .

Initial tolerance r0 using Destroys small features on large parts. Base r0 strictly on coordinate ULP after centering average edge length ˉledge Flawed the bounding box at (0, 0) plus any explicit import gap tolerance.

Would you like to zoom in on that grazing diamond → 2 endpoint balls + coincident edge transition? That is the single piece of geometry math that determines whether Stage 2 stays tight or snowballs.

## User

How do we actually know d clear?

## Response

To find a safe offset distance into a region from an edge midpoint M , you do not need to compute the global minimum clearance dclear between all edge tubes in the model. You only need the local ray distance along the normal vector n at M .

Here is why the naive distance check is subtle, and the exact, concrete algorithm to get a guaranteed interior point in O(k) time (where k is the number of edges on that region's loop).

Why Closest-Point Distance (dmin ) Is Actually the Wrong Metric Suppose you stood at the midpoint M of edge ej (which has tube radius re ), found the minimum Euclidean distance dmin from M to all other edges in the loop, and stepped out along the normal n by re + 12 dmin .

Look at what goes wrong near a sharp corner: Edge eadj is connected to the same vertex disk V as ej at a tight angle.

Even though eadj is off to the side near vertex V (nowhere near your normal ray n), its active segment starts at ∂B(V ) very close to M .

So dmin might be smaller than re itself, or stepping along n by an unvalidated distance could cross another edge if the region is pinched like an hourglass!

The Actual Algorithm: Normal Ray-Cast to First Hit (thit /2) Instead of measuring omnidirectional Euclidean distance, you cast a 1D ray directly along the inward normal n from the midpoint M of the active edge segment Ω^ (ej ): Ray(t) = M + t ⋅ n, t > 0 Now intersect that single ray with the rest of the edges belonging to the region (or the BVH of all active edges):

1. Pick the Best Starting Edge on the Loop: Among all half-edges in the region's cycle, pick the one with the longest active segment length ∥Ω^ (ej )∥ (the segment length strictly outside its two endpoint vertex disks B(p0 , r0 ) and B(p1 , r1 )). Let M be the exact midpoint of that active segment, and n be its unit normal pointing into the region (left of the half-edge).

2. Find the First Ray Intersection (thit ): Cast the ray M + tn (t > 0) against all other active edge tubes and vertex disks. Let thit be the smallest positive distance t where the ray enters any other tolerance zone Ω(ek ) or Ω(vi ) (or enters the boundary of an opposing curve).

3. Why thit > r(ej ) is Guaranteed by Invariant 1: Because M lies on the spine of active edge ej (strictly outside all vertex disks), the interval t ∈ [0, r(ej )] lies entirely inside the tube Ω^ (ej ). By Invariant 1 (Strict Stratum Disjointness), no other active edge tube or vertex disk can touch Ω^ (ej ). Therefore, the first point where the ray touches another stratum's boundary must occur at some thit > r(ej ).

4. Place the Sample Point at the Midpoint of the Free Span: Your guaranteed interior sample point is simply: r(ej ) + thit xsample = M + ( ) n 2 Look at the properties of xsample : Because the ray starts on the boundary of the region and thit is the very first boundary crossing along n, the open segment t ∈ (r(ej ), thit ) cannot cross any edge—meaning it lies 100% strictly inside the region.

Its distance to ej and to the opposing wall along n is 12 (thit − r(ej )) > 0, keeping it maximally clear of both boundaries along that slice.

Wait—What If You Don't Want to Cast Rays Inside Every Region at All?

There is an even cleaner shortcut for 2D (and 3D) that avoids picking interior sample points for internal regions altogether on clean or mostly-clean inputs: Dual-Graph Winding Propagation.

Remember from build_planar_half_edges that every Edge e already knows both of its adjacent regions: left_region = half_edges[2 * e].region right_region = half_edges[2 * e + 1].region And every Edge e knows its net signed contribution (delta_w_A, delta_w_B) from the original input edges that merged into it: Normally +1 when crossing from right to left.

If two opposing edges of Solid A coincided and cancelled out, delta_w_A = 0.

If two identically oriented edges merged, delta_w_A = +2.

That means the regions form an edge-adjacent graph:

1. Find the Unbounded Exterior Region (Rext ): Pick a point Pout outside the bounding box of the entire scene (e.g., (xmin − 10, ymin − 10)). Its winding number is trivially exact zero: wA (Rext ) = 0, wB (Rext ) = 0. Cast a single horizontal ray from Pout to the first edge hit to identify which RegionId is Rext .

2. Breadth-First Search (BFS) Across Edges: Whenever you step across edge e from right_region to left_region: wA (left) = wA (right) + edge.delta_w_A wB (left) = wB (right) + edge.delta_w_B Why combine both?

If the input shapes are closed (even if they self-intersect like a figure-8 or overlap), BFS propagation computes the exact integer winding number of every region in O(E) total time with zero interior sample points and zero geometric predicates!

You only fall back to the Normal Ray-Cast (M + re +2thit n) if an input contour had an unhealed open gap larger than 2rv (where walking around the open endpoint would give path-dependent winding numbers, so each region votes via its sample point).

Does the M + re +2thit n normal ray-cast clear up how to get the interior point without needing a full distance field or CDT?

## User

What about glancing surface intersections? Won't those also blow up onto large tolerances? How do we fix this in a way that doesn't create another difficult geometric sub problem?

## Response

Yes—glancing surface intersections are the exact 3D equivalent of the shallow 2D crossing diamond, and they are the #1 killer of Parasolid and OCCT.

If a cylinder of radius R = 100 mm is tangent to a plane, and both have a tight face sheet thickness of rf = 10−6 mm, the region where the two thick sheets overlap has a half-width of: w ≈ 2R(2rf ) = 200 ⋅ 2 × 10−6 ≈ 0.02 mm That is a 20, 000× blow-up in uncertainty width (10−6 → 2 × 10−2 ). Worse, if a sphere barely kisses a plane or a torus touches a saddle surface at a shallow angle, the overlap region of their thick sheets isn't even a curve—it is a 2D puddle (an area patch), or a noisy figure-8 where the marching surface-surface intersector (SSI) loses rank in its Jacobian matrix (nA × nB ≈ 0) and wanders drunkenly.

If you wrap that entire 0.02 mm puddle in a fat 1-stratum edge tube of radius re = 0.02 mm, you immediately trigger the tolerance snowball—AND if you try to write a special-case "glancing surface intersection solver" (computing high-order curvature derivatives to trace silhouette tangency curves), you have just created the exact kind of complex geometric sub-problem we promised to avoid.

Here is how you solve glancing surfaces without a special tangency solver and without blowing up edge tolerances, using only the machinery we already have.

The Core Insight: A Glancing Intersection Is Not a 1-Stratum (Edge)—It Is a 2-Stratum Coincidence (Shared Face) Ask why the tolerance blew up to 0.02 mm in the first place: because we tried to force a 2D area of contact into a 1D curve.

In our Stratified Tolerance Complex, what is a 2D region where two surface sheets Ω(fA ) and Ω(fB ) lie within r(fA ) + r(fB ) of each other? By definition, that is a coincident 2-stratum (a shared face patch), bounded by a closed loop of normal, low-tolerance 1- stratum edges where the surfaces finally separate by more than rA + rB !

Look at what happens when you stop trying to trace the "exact" zero-set inside a tangent zone and instead treat the overlap zone as a 2-Stratum Coincidence Patch:

1. Where the Surfaces Separate Cleanly (∥σA − σB ∥ = rA + rB ): At the boundary of the contact puddle (where the gap between σA and σB reaches τ = rA + rB ), the surfaces are no longer tangent! For the cylinder of radius R = 100 mm and τ = 2 × 10−6 mm, right at the true tangency point the angle between normals is θ = 0∘ (singular Jacobian). But at the boundary of the contact strip (x = ±0.02 mm), the angle increases enough for the signed distance field d(u, v) = dist(σA (u, v), σB ) to cross the threshold +τ with a clean, non-zero gradient.

2. All Tolerances Stay Tiny (O(rf )): Instead of one fat edge tube of radius 0.02 mm, you get a thin ribbon face (width 0.04 mm) whose two bounding edges have the tiny, un-inflated radius re ≈ 2rf = 2 × 10−6 mm, and whose face thickness is simply rAB = rA + rB = 2 × 10−6 mm. Zero tolerance blow-up.

How to Implement This Without Creating a New Geometric Sub-Problem You might think: "Wait—doesn't tracing the boundary where ∥σA − σB ∥ = rA + rB require a new offset-surface intersection algorithm?" No. Remember Stage 3 (Barrier Subdivision on the Surface) from our first message: we already partition each surface σk into 2-cells by flood-filling outside the barriers. Here is how one single algorithm handles transverse intersections, partial surface overlaps, and glancing tangencies identically: Step 1: Never Trace Where the Jacobian Is Singular In Stage 1, your Surface-Surface Intersector (SSI) only traces intersection curves where the intersection is well-conditioned (transverse): ∥nA × nB ∥ > sin(θmin ) (e.g., θmin ≈ 1∘ ) The moment the surfaces become nearly tangent (∥nA × nB ∥ ≤ sin θmin ) AND the distance between them is within τ = r(fA ) + r(fB ), the SSI stops marching the unstable zero-curve. Instead, it marks that region as a Coincidence Zone and traces the well- conditioned separation contour where the signed distance between the surfaces equals ±τ : g(u, v) = (σA (u, v) − ΠB (σA (u, v))) ⋅ nB ∓ τ = 0 Because ±τ is shifted away from the singular extremum g = 0, the gradient ∇g is non-degenerate at the separation boundary, and the edge tube around that boundary curve only needs radius re = τ + ϵfit .

Step 2: What Happens to the Ribbon Face in Stage 4 (front_keep != back_keep)?

Now trace what happens to that coincidence patch fshared during Stage 4 (Winding Number & Algebraic Extraction). There are three classic cases of glancing contact in CAD—watch how the front_keep != back_keep rule resolves all three automatically: What the 3D Volumes Around Glancing Scenario fshared Look Like What front_keep != back_keep Does Automatically On one side of fshared is the interior of For A ∪ B , both sides of fshared have keep = true. Therefore,

1. Kissing Touch the Sphere (wA = 1, wB = 0 ⟹ fshared is deleted as an internal seam, welding the sphere and box (Sphere resting on a keep = true). On the other side is together through a solid neck of radius 0.02 mm (avoiding a non- Box, A ∪ B ) the interior of the Box (wA = manifold single-point contact!). For A ∩ B , both sides are false 0, wB = 1 ⟹ keep = true). relative to A ∩ B , so the zero-volume kiss disappears cleanly!

### 2. Grazing Cut Inside the ribbon fshared , one side is

(Cylinder cutting a outside both solids (wA = 0, wB = Both sides of the tangent ribbon fshared evaluate to keep = tangent groove 0 ⟹ keep = false) and the other false! The ribbon face vanishes completely, and the two well- along a Box face, conditioned separation edges become the crisp, watertight rim of A ∖ B ) side is inside the Cylinder (wB = the groove on the box face—with tolerance 2 × 10−6 , not 0.02!

1 ⟹ keep = false for A ∖ B ).

What the 3D Volumes Around Glancing Scenario f Look Like What front_keep != back_keep Does Automatically shared The fillet and the plane merge into the

3. True Tangential same tangent face along the strip Both fplane , fshared , and ffillet have front_keep = false and Transition (Fillet fshared . Both represent the boundary back_keep = true. All three survive and stitch seamlessly along tangent to a Plane) between inside (w = 1) and outside ( the tiny-tolerance separation edges. w = 0).

Look at Why This Is Huge for Verification Pause and notice what just happened to Kissing Contacts (Case 1) and Grazing Cuts (Case 2):

1. No Non-Manifold Point/Line Contacts: In an ideal exact kernel, a sphere kissing a plane touches at a single 0D point, creating a non-manifold vertex (or a cylinder kissing a plane creates a non-manifold 1D edge shared by 4 faces). Every B-rep kernel hates non-manifold contacts because shells stop being 2-manifolds. In this architecture, because every surface has finite sheet thickness rf > 0, two smooth surfaces cannot touch at a 0D point or 1D line without intersecting over a small 2D coincidence disk/strip! That means every physical overlap is automatically a manifold 2-cell patch—either welded across a finite neck or omitted cleanly.

2. One Unified Principle Across 0D, 1D, and 2D: Look at how we solved shallow intersections in both dimensions—it is the exact same invariant: When two 1-strata (edges) graze at a shallow angle: Don't inflate a giant 0-stratum vertex ball rv ∝ re /θ. Instead, promote the overlap to a shared 1-stratum (coincident edge segment) bounded by two small 0-stratum balls where the edges separate by re1 + re2 .

When two 2-strata (faces) graze at a shallow angle: Don't inflate a giant 1-stratum edge tube re ∝ R ⋅ rf . Instead, promote the overlap to a shared 2-stratum (coincident face patch) bounded by small 1-stratum edge tubes where the faces separate by rf 1 + rf 2 .

We didn't add a new heuristic. We just applied the rule: Whenever two k-strata overlap over a region much larger than rk , that overlap is a shared k-stratum bounded by (k − 1)-strata placed at the separation threshold dist(a, b) = r(a) + r(b).

Let's Check This in 2D First (Curved Edges in Milestone 5) You don't even have to wait for 3D surfaces to test this exact mechanism. In 2D with circular arcs / NURBS curves, imagine a circle of radius R = 100 grazing a horizontal line y = 0 (both with edge tube thickness re = 10−6 ): A naive intersection solver finds the tangent point (0, 0), sees the curves stay within 2 × 10−6 of each other from x = −0.02 to x = +0.02, and inflates a vertex disk to radius rv = 0.02.

With the Separation-Threshold Rule, you place two tiny vertex disks of radius rv = 2 × 10−6 at x = −0.02 and x = +0.02 (where ycircle (x) − yline (x) = 2re , which is a transverse crossing with slope ±0.0004), and the segment between [−0.02, +0.02] is just a normal merged 1-stratum edge with re = 2 × 10−6 !

Does this unification—promoting grazing k-cell crossings into a coincident k-cell bounded by separation (k − 1)-cells—answer how we prevent tolerance blow-up without building a fragile tangency-tracing engine?

## User

Not really. Tracing the boundary does not seem easy at all

## Response

You are right to call that out. In my last answer, I slipped right back into the trap you warned me about: I replaced "trace the intersection curve σA (u, v) = σB (s, t)" with "trace the separation contour dist(σA , σB ) = ±τ ." Let's be honest about why that is still a terrible geometric sub-problem:

1. Offset surfaces are algebraically nastier than the original surfaces. Tracing dist(σA , σB ) = τ is a marching predictor- corrector problem on an offset surface.

2. Even at +τ , the contour can have its own tangencies, saddle points, and micro-islands. If a bumpy NURBS surface barely grazes a plane, the level set dist = τ can pinch off into figure-8s or tiny closed rings where the gradient of dist − τ also goes to zero!

3. Worst of all: If you have to write a marching contour tracer to find 1-strata before you can partition faces, your topology still depends on a fragile differential-geometry solver never missing a branch.

So let's throw out marching curve tracers completely.

If we want one simple, verifiable algorithm that never traces differential curves, never blows up tolerances at tangencies, and works identically in 2D and 3D, how do we actually intersect curves and surfaces?

The Real Solution: Don't "Trace" Intersections—Subdivide Until Monotone, Then Classify Cells Why do marching curve/surface intersectors fail? Because they try to walk along an unknown 1D curve inside a continuous parameter space using local derivatives (Newton steps).

What if Stage 1 never traces curves at all? Instead, treat every curve γ(t) (in 2D) or surface patch σ(u, v) (in 3D) as a recursive quadtree/interval hierarchy, and replace "intersection tracing" with adaptive cell subdivision down to τ .

Let's look at how this works in 3D for two surfaces A and B (and in 2D for two curves), and why it eliminates curve tracing entirely.

How Subdivision-Classification Works (Zero Curve Tracing) Suppose surface A is parameterized on (u, v) ∈ [0, 1]2 , and we want to split A against surface B within tolerance τ = rA + rB .

Instead of trying to solve for an intersection curve, we recursively subdivide A's domain into sub-patches P ⊂ [0, 1]2 . At each sub- patch P , we evaluate the signed distance Range [dmin , dmax ] from P to surface B (using the convex hull property of Bézier/NURBS control points or interval arithmetic):

1. Case 1: Strictly Disjoint (dmin > τ or dmax < −τ ) The entire sub-patch P is strictly outside B 's tolerance sheet. Stop subdividing P . Mark P as OUTSIDE (+1) or INSIDE (−1).

2. Case 2: Strictly Coincident (∣dmin ∣ ≤ τ AND ∣dmax ∣ ≤ τ ) Every point on sub-patch P lies within τ of surface B ! Stop subdividing P immediately! Mark P as COINCIDENT (0). Look at what just happened: If A and B have a glancing tangency or a 0.02 mm contact puddle, the quadtree doesn't care that the Jacobian is singular, and it doesn't try to trace the puddle's perimeter! The sub-patches inside the puddle immediately satisfy ∣d∣ ≤ τ at a coarse quadtree level and stop subdividing!

3. Case 3: Straddling the Boundary (dmin < τ and dmax > τ ) Subdivide P into 4 child patches, until either: Transverse Stopping Rule: The patch P is flat enough that the gradient ∇d across P has no sign flip (guaranteed by checking control-polygon normals), in which case the zero-crossing inside this single leaf cell is a trivial single straight segment connecting two edges of the quadtree cell.

Minimum Tolerance Floor: The physical diameter of patch P drops below re (the edge tolerance). At that point, mark the tiny cell as COINCIDENT (0) and stop!

Look at Why This Eliminates the Tracing Problem Compare what you have to write and verify in both approaches: In the "Tracing" Approach (What you rightly objected to): You have to find starting "seed" points on patch boundaries, march along nA × nB , detect step-size overshoot, handle branching at saddle points, detect closed internal intersection loops that never touch a patch boundary, switch to a different equation when ∥nA × nB ∥ → 0, and fit splines to the marched points.

That is a 20,000-line numerical minefield.

In the "Subdivide & Classify (-1, 0, +1)" Approach: You have one recursive function (classify_patch(P) -> {-1, 0, +1}) that only needs bound evaluation (convex hull of Bézier control points).

Every leaf cell of the subdivision on surface A now has a discrete label: -1 (strictly behind B by > τ ) 0 (within ±τ of B — the glancing/coincident zone!) +1 (strictly in front of B by > τ ) Where do the 1-strata (edges) come from? You don't trace them! They are simply the grid edges where a +1 or -1 region borders a 0 region (or where a transverse leaf cell gets split by a single straight chord between its sides).

And what happens to the 0 cells (the glancing puddle) during the Boolean? They are just a cluster of 2D cells on the surface labeled COINCIDENT. When you run the exact same front_keep != back_keep Boolean check from before: If it's a kissing contact in A ∪ B , the 0 cells have keep = true on both sides and disappear.

If it's a grazing cut in A ∖ B , the 0 cells have keep = false on both sides and disappear, leaving the boundary between the +1 cells and 0 cells as the edge loop.

Does This Still Need 3D to Test? No—Look at 2D Curves First Think about how this works on 2D Curved Edges (Circles / Béziers) when a circle barely grazes a line with tolerance τ :

1. Represent each curve on t ∈ [0, 1] as a Bézier segment.

2. Check the bounding box / convex hull of the Bézier control points against the other curve's τ -band: Entirely outside τ ? Label interval DISJOINT. Stop.

Entirely inside τ ? Label interval OVERLAPPING. Stop! (No Newton root-finding needed inside the tangent zone!) Straddling τ ? Subdivide at t = 0.5 and recurse until the interval is either strictly DISJOINT, strictly OVERLAPPING, or smaller than rv .

That's it. No Newton-Raphson tangency root-finder, no derivative singularities, and no tolerance blow-up: the grazing contact interval simply becomes a contiguous sequence of OVERLAPPING intervals (a merged 1-stratum edge), and the two tiny intervals of width < rv where it transitions from OVERLAPPING to DISJOINT become the two small endpoint vertex disks B(p0 , rv ) and B(p1 , rv ).

Sanity Check on Complexity Let's check if we actually reduced the number of hard algorithms now:

1. Stage 1 & 2 (Metric Subdivision & Consolidation): Recursively subdivide curves/surfaces using Bézier convex-hull bounds until every leaf piece is either strictly disjoint (> τ ) or strictly inside (≤ τ ). Merge the INSIDE pieces into shared strata; put small vertex balls / edge tubes of size O(τ ) at the transitions.

2. Stage 3 & 4 (Topological Extraction): Wire adjacency around the transition boundaries, propagate winding numbers via BFS across regions, and keep boundaries where left_keep != right_keep.

Does replacing differential curve/contour tracing with recursive Bézier bound subdivision (OUTSIDE / COINCIDENT / SPLIT) solve your objection about boundary tracing being too hard and fragile?

## User

Ok, so I have bunch of patches, but those are oddly shaped right? That would likely need to be approximated too

## Response

Yes. If you just stop at the quadtree cells, the boundary between the OUTSIDE (+1) cells and the COINCIDENT (0) cells is a jagged, stair-stepped quadtree boundary (like a pixelated image or Marching Cubes mesh).

If you leave it stair-stepped, you haven't built a B-rep kernel—you’ve built a voxel/mesh kernel with thousands of tiny micro-edges. And if you try to fit a smooth NURBS curve through that jagged boundary afterwards, that smoothed curve can drift outside your tolerance or self-intersect, Putting you right back where we started!

Let’s look closely at why those patches look "oddly shaped" and how you turn them into clean, smooth B-rep edges without creating micro-edges and without losing your topological guarantee.

The Crucial Distinction: Two Different Kinds of Boundaries on the Quadtree When you subdivide Surface A against Surface B , look at why a leaf cell is labeled 0 (|dist| <= τ) vs +1 (dist > τ). There are only two geometric situations that occur: Case A: Normal Transverse Intersection (99% of CAD intersections) Suppose a cylinder punches through a plane at 90∘ (or 15∘ ). How wide is the 0 zone where ∣d∣ ≤ τ ? If τ = 10−6 mm and the angle is 30∘ , the width of the band where ∣d∣ ≤ τ is 2 × 10−6 mm.

You do not subdivide the quadtree all the way down to 10−6 mm! That would create a million tiny boxes.

Instead, at a coarse quadtree cell (say, 5 mm × 5 mm), you check the normal cones of A and B . You see that the surfaces cross transversely (∥nA × nB ∥ > sin 1∘ ) and each surface is nearly flat across that 5 mm cell.

Inside a transverse cell, the intersection isn't an oddly shaped puddle—it is a single, unique, monotone curve segment entering one side of the cell and exiting another side!

You evaluate 4 or 5 exact points along that curve segment inside the cell (using trivial 1D Newton polishing along the cell boundary and interior, which cannot fail because the gradient is steep and monotone inside this cell) and fit a single cubic Bézier segment with error ϵfit < 10−7 .

Case B: The True Glancing / Tangent Puddle (The remaining 1%) What if the cell is not transverse (∥nA × nB ∥ ≤ sin 1∘ ) and ∣d∣ ≤ τ over a broad area? Now—and only now—you have a macro- sized puddle of 0 cells surrounded by +1 or -1 cells.

Wait: Why would you ever need to preserve the exact shape of the puddle boundary?

Think about what the puddle is: Every single point inside the 0 puddle—and every point along its jagged boundary—has a distance between Surface A and Surface B that is already smaller than τ = rA + rB !

That means any curve you draw inside the transition band of the 0 puddle is a geometrically valid intersection curve within tolerance τ !

Let that sink in: In a traditional kernel, people think there is a "true" mathematical boundary of the tangency puddle that they have to trace and approximate accurately.

There isn't. Inside the 0 zone, Surface A and Surface B are literally indistinguishable within your tolerance τ .

Whether the boundary edge between OUTSIDE and COINCIDENT goes along the left side of a quadtree cell, the middle, or gets simplified into a single straight line / cubic spline cutting right through the 0 cells, the 3D distance between Surface A and Surface B along that simplified curve is still ≤ τ everywhere!

How This Solves the Approximation Problem (The "Thick Corridor" Simplification) Because the 0 cells form a valid tolerance corridor where ∣d(A, B)∣ ≤ τ everywhere inside them, you don't have to approximate an oddly shaped boundary with high precision. You do the exact opposite: you find the simplest possible curve (fewest control points) that stays inside the 0/straddling corridor!

Here is the concrete 3-step recipe that turns the quadtree into a minimum-edge B-rep:

1. Coarse Transverse Cells Give Exact Curve Segments: Wherever the quadtree stops early because a cell is transverse (Case A), the cell directly gives you a clean, single curve segment from entry edge to exit edge.

2. Glancing 0-Cell Clusters Become "Free Routing Corridors": Wherever a transverse intersection curve enters a cluster of glancing 0 cells (or where a cluster of 0 cells separates +1 cells from -1 cells), you have a "fat corridor" of quadtree cells where ∣d(A, B)∣ ≤ τ holds across the entire corridor.

3. Polyline / Spline Simplification Inside the Corridor: Take the jagged grid chain separating the +1 cells from the 0 cells, and run standard Douglas-Peucker / Bezier curve simplification with only one constraint: The simplified curve must stay inside the union of the 0 cells (and stay outside any vertex disks).

Because every point in the 0 cells already satisfies dist(σA , σB ) ≤ τ , any simplified curve that stays inside those 0 cells automatically has a 3D surface-to-surface error ≤ τ !

Let's Trace a Concrete Example to See Why This Works Suppose a sphere barely grazes a flat box face (a kissing contact or shallow dimple), creating a circular contact puddle of radius 0.02 mm where the surfaces are within τ = 10−6 mm of each other.

What happens in the quadtree + corridor pipeline?

1. The quadtree subdivides down until the cells around the 0.02 mm puddle are, say, 0.005 mm wide.

2. Around that 0.02 mm circle, there is a ring of ~12 quadtree cells where the distance transitions from < 10−6 mm (0) to > 10−6 mm (+1).

3. You fit a single closed cubic B-spline (or 4 circular arcs) right through the middle of that ring of transition cells.

4. Does that spline need to be an exact algebraic offset curve? No! As long as the spline stays inside that ring of transition cells, its 3D distance to both the sphere and the box is ≈ 10−6 mm.

5. Better yet: if the entire puddle of 0 cells has diameter < 2rv (or if the Boolean is A ∖ B where the sphere doesn't even break through to the -1 side of the box face), look at the labels around that puddle: The puddle is 0, surrounded completely by +1 (OUTSIDE), with zero -1 (INSIDE) cells anywhere inside it!

When Stage 4 runs front_keep != back_keep, a 0 island surrounded entirely by +1 has the exact same region classification on both sides—so the entire grazing puddle is discarded before you even have to fit a curve to it!

Look at that last bullet point again: If two surfaces kiss without penetrating beyond τ (no -1 cell inside the puddle), you don't have to approximate the puddle boundary at all—it has no -1 interior, so it deletes itself!

If one surface does penetrate the other (so there is a +1 region, a -1 region, and a thin 0 transition corridor between them), any smooth spline routed through the 0 transition corridor separates +1 from -1 and is guaranteed to lie within τ of both surfaces.

Why This Finally Closes the Loop Let's check if any piece is still missing or magical:

1. Why doesn't curve tracing blow up at tangencies? Because we don't trace in tangent zones. We classify quadtree cells into +1 (d > τ ), -1 (d < −τ ), and 0 (∣d∣ ≤ τ ).

2. Why don't we get a jagged mesh of tiny quadtree edges? Because in transverse cells, we stop at coarse quadtree levels and fit clean cubic Béziers; in tangent zones, the 0 cells act as a valid tolerance corridor through which we route a simplified spline (or discard the puddle entirely if it doesn't separate +1 from -1).

3. Why is topology guaranteed? Because the fitted edges simply partition the cell complex into +1 and -1 regions, and final faces/edges are selected purely by left_keep != right_keep.

Does the realization that the 0 cells are a valid τ -corridor (where any simplified curve inside the corridor is within tolerance τ , and any 0-puddle that doesn't separate +1 from -1 can be ignored) resolve the "oddly shaped patch" problem?

## User

Ok, sure. But what about fillets? Does this representation really help us?

## Response

Fillets are where traditional B-rep kernels go to die. In OCCT (BRepFilletAPI_MakeFillet), filleting is a fragile, special- case nightmare that breaks constantly—and the reason it breaks is that a fillet is 100% intentional tangency.

There are two ways fillets enter your kernel, and we should be brutally honest about where this architecture solves the whole problem for free vs. where you still have to do real differential geometry.

1. Where This Representation Is a Massive Win: Trimming, Overrunning, and Corner Blending Why does creating a fillet of radius R along an edge network fail in OCCT or Parasolid? Generating the raw rolling-ball surface (a pipe or torus/canal surface tangent to Face A and Face B ) is actually easy. It's the trimming and topological stitching of that fillet surface back into the solid that fails:

1. The Tangent Contact Line Problem: A fillet surface Ffillet touches the base faces FA and FB along two spring curves (the tangent contact curves where nfillet = nA ). If you try to trim FA and Ffillet using a standard Boolean or surface-surface intersector, the intersector fails immediately because the two surfaces are tangent (nfillet × nA = 0) along the entire length of the fillet!

2. Fillet-Fillet Collisions & Self-Intersection: When a fillet rounds a tight concave corner, or when three fillets of different radii meet at a 3-edge vertex, or when a fillet is wider than an adjacent boss/hole ("overrunning" a neighboring face), the raw fillet tubes self- intersect, swallow entire faces, or cross neighboring faces at grazing angles. In OCCT, you have to write hundreds of special-case topological patches ("3-corner sphere patch", "cliff edge overrun", "chamfer-fillet overlap").

How the Tolerance Complex + Region-Diffing Solves This automatically: Because our pipeline does not break on tangent surfaces or self-intersecting inputs, creating a fillet reduces to a Localized Boolean Operation:

1. Explicitly Inject the Known Spring Curves as 1-Strata: When you construct a rolling-ball fillet surface Ffillet (u, v) between FA and FB , the rolling-ball generator already knows the two tangent contact curves γA (u) and γB (u) by construction (they are literally the v = 0 and v = 1 boundary iso-curves of the rolling-ball strip!).

You do not run a surface-surface intersector between Ffillet and FA to rediscover γA (u).

You inject γA (u) and γB (u) directly as exact 1-stratum edge tubes of radius re = rf + ϵfit .

2. Extend the Fillet Unconstrained & Let Stage 2 + Stage 4 Trim It: What happens when the fillet overruns an adjacent face FC , or when two fillets collide at a corner, or when a fillet folds over itself in a tight bend?

You just let the raw fillet surface extend past the corner, intersect it with FC (or the adjacent fillet) using the standard pipeline, and run Stage 4 (3D Winding / Volume Classification + front_keep != back_keep)!

Think about what a concave vs. convex fillet is in 3D volume space: A concave fillet (adding material into an inside corner) is literally a local Boolean Union (S ∪ Vfillet_wedge ).

A convex fillet (rounding off an outside sharp edge) is literally a local Boolean Difference (S ∖ Vcorner_cutter ), where Vcorner_cutter is the prism volume trapped between old faces FA , FB and the new fillet surface Ffillet !

When a convex fillet overruns a neighboring boss or collides with another fillet, front_keep != back_keep automatically deletes the sharp corner faces, deletes the self-intersecting overhangs of the fillet tubes, and leaves a closed, watertight 2- manifold shell—even if the fillet completely swallowed three small faces along the way!

### 2. Where You Have to Be Careful: The G1 Tangency Trap in Stage 2

Now let's put on our skeptic hat. If a fillet meets Face A tangentially along the spring curve γA (u), what happens inside our Stage 2 (Stratum Consolidation) and Stage 3 (Radial Ordering around the Edge Tube)?

Look at a cross-section perpendicular to the spring curve γA : Coming in from the left is the remaining part of the flat base face FA .

Leaving to the right is the curved fillet face Ffillet , which starts at angle 0∘ (G1 tangent to FA ) and curves away with radius R.

And what about the discarded piece of FA (the part of FA that goes into the sharp corner that is being cut off)? It also leaves the spring curve at angle 0∘ , staying within distance τ = 2rf of Ffillet for that entire grazing width w = 2Rτ !

If you aren't careful right here, two bad things will happen:

1. Near the spring curve, Ffillet and the discarded part of FA lie within τ of each other over a strip of width w = 2Rτ ≈ 0.02 mm, violating Invariant 1 outside the thin edge tube re = 10−6 mm.

2. In Stage 3, when you slice a normal circle of radius re = 10−6 mm around the spring curve to radially sort the incident faces around the edge, Ffillet and the discarded half of FA are separated by only d ≈ re2 = 10−12 = 5 × 10−15 mm—which is 2R 200 smaller than rf ! Their face sheets overlap on the edge tube collar!

Catching that now saves you months of pain later. Let's see how you solve it cleanly without breaking the invariants.

3. The Clean Fix: Normal-Curvature Collar Offset (or Pre-Splitting at Known Contact Curves) Why did Ffillet and the discarded side of FA overlap on the collar of the spring curve γA ? Because along γA , we already know Ffillet is tangent to FA (nfillet = nA ) and diverges solely by its normal curvature κn = 1/R!

You have two ways to handle this—one specialized for fillet creation, and one completely general for any G1 tangent edge in the kernel: Option A (For Fillet Creation): Pre-Drop the Corner Side of FA Across the Spring Curve When you generate a fillet along edge esharp between FA and FB , the spring curve γA partitions FA into two halves: the base side (away from esharp ) and the corner-strip side (between γA and esharp ).

On the base side of γA , FA leaves the tube at angle 180∘ opposite to Ffillet . They are separated by 180∘ on the collar circle— zero overlap!

Only the corner-strip side of FA (which is moving in the same 0∘ direction as Ffillet ) causes the grazing overlap.

If you only use the corner-strip during the volume classification (or orient the radial port of Ffillet vs. FA by their second derivative / normal curvature κn when their first derivatives T⊥ match), the radial order around the spring-curve edge is determined instantaneously without inflating the edge tube!

Option B (General Rule for Any G1 Edge in the Kernel): The Jet-Order Radial Sort Let's make Option A completely general so it works even if someone imports a STEP file with pre-existing tangent fillets and runs a Boolean on it.

In Stage 3, when sorting incident face sheets around an edge tube ej (in 3D) or sorting incident curved edges around a vertex disk vi (in 2D): 99% of edges/faces meet transversely (G0 ): Their exit angles θ1 = θ2 on the collar circle ∂Ω are separated by more than rsheet /rcollar . You sort by position/angle on ∂Ω.

Tangent transitions (G1 , like a fillet meeting a plane or a 2D arc tangent to a line): Two sheets exit the collar at the same tangent angle (θ1 = θ2 ), so they stay within τ of each other inside the grazing strip w = 2Rτ . Instead of inflating the edge tube to w = 0.02 mm to swallow the grazing strip, evaluate the exit order at the separation distance where the two sheets first diverge by r1 + r2 along their normal! Because a fillet of radius R and a plane diverge monotonically (d(s) = 2R1 s2 ), they cross the separation threshold r1 + r2 strictly once (no oscillation!). That single exit point gives their exact, unambiguous radial order around the edge while keeping the edge tube itself at re = 10−6 mm.

Summary: What Does This Architecture Actually Buy You for Fillets?

Let's be completely candid about what this representation does and doesn't do for fillets:

1. It does NOT replace the differential geometry of rolling a sphere between two surfaces. You still have to write the spine curve solver (finding the locus of a sphere of radius R equidistant to FA and FB ) and the 3-corner setback blend patch when three convex/concave edges meet.

2. It DOES solve the hardest part of filleting—Topological Trimming, Overruns, and Boolean Robustness on Filleted Models: Once you generate the raw rolling-ball surfaces, trimming them against the body, resolving colliding fillets, and swallowing overrun faces is just a standard pass of the Boolean engine.

Even more importantly: in OCCT, doing a Boolean operation on a part that already has fillets (e.g., subtracting a pocket that cuts right across the spring line of an existing fillet) fails constantly because the pocket plane hits the G1 tangent transition between the fillet and the wall. In our architecture, because region/volume classification (front_keep != back_keep) and the τ -corridor don't care about G1 derivative discontinuities or grazing transitions, Booleans across existing fillets cannot produce non-manifold cracks.

If we test this in our 2D incremental plan (Milestone 5), a 2D fillet is just a circular arc tangent to two line segments. Would it be helpful to trace that exact 2D case—Say, computing the Boolean difference of a polygon that cuts right through the tangent transition point between a line and a fillet arc—to see how the data structures handle it step-by-step?
