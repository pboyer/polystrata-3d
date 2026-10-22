# polystrata-3d

**[Interactive 3D Planar B-Rep Lab](https://polystrata3d.ptrbyr.com)** &bull; **[Live 3D Benchmark Suite](https://polystrata3d.ptrbyr.com/benchmark)** &bull; **[npm](https://www.npmjs.com/package/polystrata-3d)** &bull; **[crates.io](https://crates.io/crates/polystrata-3d)** &bull; **[docs.rs](https://docs.rs/polystrata-3d)**

`polystrata-3d` is a high-performance, topologically robust **3D Planar Boundary-Representation (B-Rep) Boolean & Healing kernel** implemented in Rust and compiled to WebAssembly.

Unlike triangle-mesh CSG libraries (`manifold-3d`, `three-bvh-csg`), `polystrata-3d` operates directly on **exact planar polygon faces with interior holes** in IEEE-754 `f64` precision, unifying coplanar regions, resolving 3D T-junctions conformally, and stripping non-manifold dangling fins to guarantee closed, watertight 2-manifold (`∂₁ · ∂₂ = 0`) output.

- **Zero dependencies**, self-contained dual ESM + CommonJS build (including embedded WASM) and native Rust crate ([`polystrata-3d` on crates.io](https://crates.io/crates/polystrata-3d)).
- **Faster than `manifold-3d` (v3.5.4 C++/WASM) on 7 of 10 3D CAD, bicylinder, and glancing workloads** (`1.04×–2.17×` faster), while emitting **3×–4× fewer polygonal faces** and **10⁸× lower volume error** (`~10⁻¹³` vs. `~10⁻⁵`).
- **100% watertight 2-manifold pass rate (`24/24`)** across the 3D adversarial glancing & degeneracy stress suite (`manifold-3d`: `23/24` with 1 crash on non-manifold input; `three-bvh-csg`: `3/24`; `@jscad/modeling`: `4/24`).
- Looking for 2D polygon clipping? See [**`polystrata`**](https://github.com/pboyer/polystrata).

---

## Installation

**JavaScript / TypeScript ([npm](https://www.npmjs.com/package/polystrata-3d)):**
```bash
npm install polystrata-3d
```

**Rust ([crates.io](https://crates.io/crates/polystrata-3d) / [docs.rs](https://docs.rs/polystrata-3d)):**
```bash
cargo add polystrata-3d
```

---

## Quick Start (TypeScript / JavaScript)

```ts
import {
  union,
  intersect,
  difference,
  xor,
  heal,
  createBox,
  extrudePolygon,
  transformSolid,
  initPolystrata3D,
} from "polystrata-3d";

// Optional on Node.js / Bun / Deno / Web Workers; recommended once at startup on browser main threads:
await initPolystrata3D();

const boxA = createBox([-5, -5, -5], [5, 5, 5]);
const boxB = createBox([0, 0, 0], [8, 8, 8]);

const diff = difference(boxA, boxB);
console.log(diff.solid.faces);          // Exact unified planar faces ({ outer, holes, normal })
console.log(diff.triangles);            // Ready-to-render 3D surface triangles
console.log(diff.totalVolume);          // 875.0
console.log(diff.unmatchedHalfEdgeCount); // 0 (watertight 2-manifold)
```

### Quick Start (Rust)

```rust
use polystrata_3d::{union, intersect, difference, xor, heal, KernelOptions, Point3D, Face3D, Solid3D};

let opts = KernelOptions::default();
let res = difference(&solid_a, &solid_b, opts);
assert_eq!(res.unmatched_half_edge_count, 0);
```

---

## Benchmarks vs. `manifold-3d`, `three-bvh-csg`, and `@jscad/modeling`

Run the 3D benchmark suite locally with `npm run bench`:

| 3D Workload | Input Faces | `polystrata-3d` (Planar B-Rep) | `manifold-3d` v3.5.4 (C++/WASM) | `three-bvh-csg` v0.0.18 | `@jscad/modeling` v2.13.0 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Multi-Hole Flange & Tilted Pillar** | 14 + 6 | **5,063 op/s** (`22F/92T`, PASS) | 4,882 op/s (`92F/92T`, PASS) | 1,672 op/s (NON-MANIFOLD) | 4,952 op/s (NON-MANIFOLD) |
| **Coplanar Overlapping Hexahedra** | 6 + 6 | **25,779 op/s** (`6F/12T`, PASS) | 16,360 op/s (`28F/28T`, PASS) | 4,305 op/s (NON-MANIFOLD) | 40,097 op/s (`6F/12T`, PASS) |
| **8-Tooth Star Gear & Hex Cutter** | 18 + 8 | **3,574 op/s** (`39F/136T`, PASS) | 3,315 op/s (`156F/156T`, PASS) | 1,303 op/s (NON-MANIFOLD) | 3,353 op/s (NON-MANIFOLD) |
| **Steinmetz Bicylinder (Tangent)** | 18 + 18 | **3,250 op/s** (`68F/184T`, PASS) | 2,440 op/s (`184F/184T`, PASS) | 1,250 op/s (NON-MANIFOLD) | 1,717 op/s (`68F/184T`, PASS) |
| **UV-Spheres 3.5° Glancing Lens** | 144 + 144 | **2,124 op/s** (`288F/552T`, PASS) | 5,141 op/s (`576F/576T`, PASS) | 2,996 op/s (NON-MANIFOLD) | 696 op/s (NON-MANIFOLD) |
| **Torus Saddle Cut by 2° Cylinder** | 160 + 18 | **1,219 op/s** (`157F/494T`, PASS) | 1,972 op/s (`558F/558T`, PASS) | 850 op/s (NON-MANIFOLD) | 413 op/s (NON-MANIFOLD) |
| **Nearly Parallel Cylinders (1.75°)** | 18 + 18 | **3,440 op/s** (`50F/180T`, PASS) | 1,583 op/s (`222F/222T`, PASS) | 922 op/s (NON-MANIFOLD) | 1,669 op/s (NON-MANIFOLD) |
| **Pointed Cone Grazing Box (0.8°)** | 6 + 17 | **10,857 op/s** (`23F/50T`, PASS) | 16,375 op/s (`50F/50T`, PASS) | 5,155 op/s (NON-MANIFOLD) | 12,695 op/s (NON-MANIFOLD) |
| **Intermeshing Comb Teeth (1.1°)** | 14 + 14 | **4,588 op/s** (`36F/132T`, PASS) | 3,243 op/s (`134F/134T`, PASS) | 1,357 op/s (NON-MANIFOLD) | 3,154 op/s (NON-MANIFOLD) |
| **Box with Non-Manifold Dangling Fin** | 7 + 6 | **16,569 op/s** (`12F/36T`, PASS) | **CRASH** (`NotManifold`) | 7,240 op/s (NON-MANIFOLD) | 17,623 op/s (NON-MANIFOLD) |
| **24-Scenario 3D Stress Suite (72 ops)** | — | **1,737 op/s — 24/24 (100%) 2-Manifold** | 1,950 op/s — 23/24 (95.8%) (1 crash) | 1,015 op/s — 3/24 (12.5%) | 942 op/s — 4/24 (16.7%) |

---

## License

[MIT](./LICENSE)
