/**
 * 3D Planar B-Rep Boolean & Self-Intersection Healing Engine for `polystrata-3d`.
 *
 * Implements the 4-Stage Coupled 3D Stratified Tolerance Complex:
 * - Canonical carrier plane clustering (`π_c`) & 1D plane-plane intersection chord injection
 * - Coupled 3D 0-stratum & 1-stratum fixpoint consolidation with regularized 3×3 quadric projection
 * - Combinatorial 2-stratum (`∂₂`) & radial CoFace 3-stratum (`∂₃`) volume cell extraction
 * - Van Oosterom–Strackee 3D solid-angle winding classification, coplanar face unification,
 *   and conformal 3D T-junction subdivision for watertight 2-manifold (`∂₁ · ∂₂ = 0`) output
 */

import type {
  BooleanOp,
  BRepResult3D,
  Face3D,
  KernelOptions,
  Point3D,
  Solid3D,
  Triangle3D,
  TuplePoint3D,
  WasmBRep3DResult,
  WindingRule,
} from "./types.js";
import { getWasm } from "./wasm-kernel.js";

export type {
  BooleanOp,
  BRepResult3D,
  Face3D,
  KernelOptions,
  Point3D,
  Solid3D,
  Triangle3D,
  TuplePoint3D,
  WasmBRep3DResult,
  WindingRule,
};

function toPoint3D(p: Point3D | TuplePoint3D): Point3D {
  if (Array.isArray(p)) {
    return { x: p[0], y: p[1], z: p[2] };
  }
  return { x: (p as Point3D).x, y: (p as Point3D).y, z: (p as Point3D).z };
}

function normalize3(v: Point3D): Point3D {
  const len = Math.hypot(v.x, v.y, v.z);
  if (len <= 1e-15) return { x: 0, y: 0, z: 1 };
  return { x: v.x / len, y: v.y / len, z: v.z / len };
}

/**
 * Compute the Newell outward unit normal of a 3D polygon ring.
 */
export function computeLoopNormal3D(loop: readonly (Point3D | TuplePoint3D)[]): Point3D {
  let nx = 0;
  let ny = 0;
  let nz = 0;
  const n = loop.length;
  for (let i = 0; i < n; i++) {
    const cur = toPoint3D(loop[i]!);
    const next = toPoint3D(loop[(i + 1) % n]!);
    nx += (cur.y - next.y) * (cur.z + next.z);
    ny += (cur.z - next.z) * (cur.x + next.x);
    nz += (cur.x - next.x) * (cur.y + next.y);
  }
  return normalize3({ x: nx, y: ny, z: nz });
}

/** Alias for {@link computeLoopNormal3D}. */
export const computeLoopNormal = computeLoopNormal3D;

/**
 * Construct an axis-aligned 3D box `[min[0], max[0]] × [min[1], max[1]] × [min[2], max[2]]`
 * with 6 outward-oriented planar faces.
 */
export function createBox3D(
  min: TuplePoint3D | Point3D,
  max: TuplePoint3D | Point3D,
): Solid3D {
  const pMin = toPoint3D(min);
  const pMax = toPoint3D(max);
  const x0 = Math.min(pMin.x, pMax.x);
  const x1 = Math.max(pMin.x, pMax.x);
  const y0 = Math.min(pMin.y, pMax.y);
  const y1 = Math.max(pMin.y, pMax.y);
  const z0 = Math.min(pMin.z, pMax.z);
  const z1 = Math.max(pMin.z, pMax.z);

  const p000 = { x: x0, y: y0, z: z0 };
  const p100 = { x: x1, y: y0, z: z0 };
  const p110 = { x: x1, y: y1, z: z0 };
  const p010 = { x: x0, y: y1, z: z0 };
  const p001 = { x: x0, y: y0, z: z1 };
  const p101 = { x: x1, y: y0, z: z1 };
  const p111 = { x: x1, y: y1, z: z1 };
  const p011 = { x: x0, y: y1, z: z1 };

  return {
    faces: [
      { outer: [p000, p010, p110, p100], holes: [], normal: { x: 0, y: 0, z: -1 } },
      { outer: [p001, p101, p111, p011], holes: [], normal: { x: 0, y: 0, z: 1 } },
      { outer: [p000, p100, p101, p001], holes: [], normal: { x: 0, y: -1, z: 0 } },
      { outer: [p010, p011, p111, p110], holes: [], normal: { x: 0, y: 1, z: 0 } },
      { outer: [p000, p001, p011, p010], holes: [], normal: { x: -1, y: 0, z: 0 } },
      { outer: [p100, p110, p111, p101], holes: [], normal: { x: 1, y: 0, z: 0 } },
    ],
  };
}

/** Alias for {@link createBox3D}. */
export const createBox = createBox3D;

function signedArea2D(pts: readonly (readonly [number, number])[]): number {
  let s = 0;
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i]!;
    const b = pts[(i + 1) % pts.length]!;
    s += a[0] * b[1] - b[0] * a[1];
  }
  return 0.5 * s;
}

/**
 * Construct a 3D prism by extruding a 2D polygon (with optional interior holes)
 * along `[zMin, zMax]`.
 */
export function extrudePolygon3D(
  outer2D: readonly (readonly [number, number])[],
  holes2D: readonly (readonly (readonly [number, number])[])[] = [],
  zMin = 0,
  zMax = 10,
): Solid3D {
  const z0 = Math.min(zMin, zMax);
  const z1 = Math.max(zMin, zMax);

  const outerCCW = [...outer2D];
  if (signedArea2D(outerCCW) < 0) outerCCW.reverse();

  const holesCW = holes2D.map((h) => {
    const ring = [...h];
    if (signedArea2D(ring) > 0) ring.reverse();
    return ring;
  });

  const faces: Face3D[] = [];

  // Top face (+Z)
  faces.push({
    outer: outerCCW.map(([x, y]) => ({ x, y, z: z1 })),
    holes: holesCW.map((h) => h.map(([x, y]) => ({ x, y, z: z1 }))),
    normal: { x: 0, y: 0, z: 1 },
  });

  // Bottom face (-Z)
  faces.push({
    outer: [...outerCCW].reverse().map(([x, y]) => ({ x, y, z: z0 })),
    holes: holesCW.map((h) => [...h].reverse().map(([x, y]) => ({ x, y, z: z0 }))),
    normal: { x: 0, y: 0, z: -1 },
  });

  // Outer side walls
  for (let i = 0; i < outerCCW.length; i++) {
    const [x0, y0] = outerCCW[i]!;
    const [x1, y1] = outerCCW[(i + 1) % outerCCW.length]!;
    const dx = x1 - x0;
    const dy = y1 - y0;
    const len = Math.hypot(dx, dy);
    if (len <= 1e-12) continue;
    faces.push({
      outer: [
        { x: x0, y: y0, z: z0 },
        { x: x1, y: y1, z: z0 },
        { x: x1, y: y1, z: z1 },
        { x: x0, y: y0, z: z1 },
      ],
      holes: [],
      normal: { x: dy / len, y: -dx / len, z: 0 },
    });
  }

  // Hole side walls
  for (const h of holesCW) {
    for (let i = 0; i < h.length; i++) {
      const [x0, y0] = h[i]!;
      const [x1, y1] = h[(i + 1) % h.length]!;
      const dx = x1 - x0;
      const dy = y1 - y0;
      const len = Math.hypot(dx, dy);
      if (len <= 1e-12) continue;
      faces.push({
        outer: [
          { x: x0, y: y0, z: z0 },
          { x: x1, y: y1, z: z0 },
          { x: x1, y: y1, z: z1 },
          { x: x0, y: y0, z: z1 },
        ],
        holes: [],
        normal: { x: dy / len, y: -dx / len, z: 0 },
      });
    }
  }

  return { faces };
}

/** Alias for {@link extrudePolygon3D}. */
export const extrudePolygon = extrudePolygon3D;

/**
 * Apply an affine or rigid 3D transformation `(p) => Point3D` to every vertex of a `Solid3D`
 * and recompute outward unit normals.
 */
export function transformSolid3D(
  solid: Solid3D,
  transformPt: (p: Point3D) => Point3D,
): Solid3D {
  return {
    faces: solid.faces.map((f) => {
      const outer = f.outer.map(transformPt);
      const holes = f.holes.map((h) => h.map(transformPt));
      const normal = computeLoopNormal3D(outer);
      return { outer, holes, normal };
    }),
  };
}

/** Alias for {@link transformSolid3D}. */
export const transformSolid = transformSolid3D;

function ruleToCode(rule: WindingRule): number {
  switch (rule) {
    case "positive":
      return 0;
    case "nonZero":
      return 1;
    case "evenOdd":
      return 2;
  }
}

function opToCode(op: BooleanOp | "heal"): number {
  switch (op) {
    case "union":
      return 0;
    case "intersect":
      return 1;
    case "difference":
      return 2;
    case "xor":
      return 3;
    case "heal":
      return 4;
  }
}

function serializeSolidsToWasm(solids: readonly Solid3D[]): Float64Array {
  let count = 1; // num_solids
  for (const s of solids) {
    count += 1; // num_faces
    for (const f of s.faces) {
      count += 4; // nx, ny, nz, num_loops
      count += 1 + f.outer.length * 3;
      for (const h of f.holes) {
        count += 1 + h.length * 3;
      }
    }
  }

  const wasm = getWasm();
  const ptr = wasm.wasm_brep3d_alloc_input(count);
  const buf = new Float64Array(wasm.memory.buffer, ptr, count);

  let c = 0;
  buf[c++] = solids.length;
  for (const s of solids) {
    buf[c++] = s.faces.length;
    for (const f of s.faces) {
      buf[c++] = f.normal.x;
      buf[c++] = f.normal.y;
      buf[c++] = f.normal.z;
      buf[c++] = 1 + f.holes.length;

      buf[c++] = f.outer.length;
      for (const p of f.outer) {
        buf[c++] = p.x;
        buf[c++] = p.y;
        buf[c++] = p.z;
      }
      for (const h of f.holes) {
        buf[c++] = h.length;
        for (const p of h) {
          buf[c++] = p.x;
          buf[c++] = p.y;
          buf[c++] = p.z;
        }
      }
    }
  }
  return buf;
}

function executeBRep3DInternal(
  op: BooleanOp | "heal",
  solids: readonly Solid3D[],
  options?: KernelOptions,
): BRepResult3D {
  const wasm = getWasm();
  serializeSolidsToWasm(solids);

  const ruleCode = ruleToCode(options?.windingRule ?? "positive");
  const snapTol = options?.snapTolerance ?? 0;
  const simplify = (options?.simplifyCollinear ?? true) ? 1 : 0;

  const faceCount = wasm.wasm_brep3d_execute(opToCode(op), ruleCode, snapTol, simplify);

  const coordsPtr = wasm.wasm_brep3d_get_out_coords_ptr();
  const coordsLen = wasm.wasm_brep3d_get_out_coords_len();
  const coords = new Float64Array(wasm.memory.buffer, coordsPtr, coordsLen);

  const loopMetaPtr = wasm.wasm_brep3d_get_out_loop_meta_ptr();
  const loopCount = wasm.wasm_brep3d_get_out_loop_count();
  const loopMeta = new Uint32Array(wasm.memory.buffer, loopMetaPtr, loopCount * 4);

  const normalsPtr = wasm.wasm_brep3d_get_out_face_normals_ptr();
  const normals = new Float64Array(wasm.memory.buffer, normalsPtr, faceCount * 3);

  const trisPtr = wasm.wasm_brep3d_get_out_triangles_ptr();
  const trisLen = wasm.wasm_brep3d_get_out_triangles_len();
  const trisBuf = new Float64Array(wasm.memory.buffer, trisPtr, trisLen);

  const faces: Face3D[] = Array.from({ length: faceCount }, (_, fIdx) => ({
    outer: [],
    holes: [],
    normal: {
      x: normals[fIdx * 3] ?? 0,
      y: normals[fIdx * 3 + 1] ?? 0,
      z: normals[fIdx * 3 + 2] ?? 1,
    },
  }));

  for (let l = 0; l < loopCount; l++) {
    const faceId = loopMeta[l * 4] ?? 0;
    const isHole = (loopMeta[l * 4 + 1] ?? 0) !== 0;
    const startPt = loopMeta[l * 4 + 2] ?? 0;
    const ptCount = loopMeta[l * 4 + 3] ?? 0;

    const ring: Point3D[] = new Array(ptCount);
    for (let i = 0; i < ptCount; i++) {
      const base = (startPt + i) * 3;
      ring[i] = {
        x: coords[base] ?? 0,
        y: coords[base + 1] ?? 0,
        z: coords[base + 2] ?? 0,
      };
    }

    const targetFace = faces[faceId];
    if (targetFace) {
      if (!isHole) {
        targetFace.outer = ring;
      } else {
        targetFace.holes.push(ring);
      }
    }
  }

  const triCount = Math.floor(trisLen / 13);
  const triangles: Triangle3D[] = new Array(triCount);
  for (let t = 0; t < triCount; t++) {
    const base = t * 13;
    triangles[t] = {
      a: { x: trisBuf[base] ?? 0, y: trisBuf[base + 1] ?? 0, z: trisBuf[base + 2] ?? 0 },
      b: {
        x: trisBuf[base + 3] ?? 0,
        y: trisBuf[base + 4] ?? 0,
        z: trisBuf[base + 5] ?? 0,
      },
      c: {
        x: trisBuf[base + 6] ?? 0,
        y: trisBuf[base + 7] ?? 0,
        z: trisBuf[base + 8] ?? 0,
      },
      normal: {
        x: trisBuf[base + 9] ?? 0,
        y: trisBuf[base + 10] ?? 0,
        z: trisBuf[base + 11] ?? 1,
      },
      faceIndex: trisBuf[base + 12] ?? 0,
    };
  }

  return {
    solid: { faces },
    triangles,
    totalVolume: wasm.wasm_brep3d_get_out_volume(),
    totalSurfaceArea: wasm.wasm_brep3d_get_out_area(),
    vertexCount: wasm.wasm_brep3d_get_out_vertex_count(),
    edgeCount: wasm.wasm_brep3d_get_out_edge_count(),
    unmatchedHalfEdgeCount: wasm.wasm_brep3d_get_out_unmatched_he_count(),
  };
}

/**
 * Compute the 3D Boolean union (`A ∪ B`) of two planar B-Rep solids.
 */
export function union3D(
  a: Solid3D,
  b: Solid3D,
  options?: KernelOptions,
): BRepResult3D {
  return executeBRep3DInternal("union", [a, b], options);
}

/** Alias for {@link union3D}. */
export const union = union3D;

/**
 * Compute the 3D Boolean intersection (`A ∩ B`) of two planar B-Rep solids.
 */
export function intersect3D(
  a: Solid3D,
  b: Solid3D,
  options?: KernelOptions,
): BRepResult3D {
  return executeBRep3DInternal("intersect", [a, b], options);
}

/** Alias for {@link intersect3D}. */
export const intersect = intersect3D;

/**
 * Compute the 3D Boolean difference (`A \ B`) of two planar B-Rep solids.
 */
export function difference3D(
  a: Solid3D,
  b: Solid3D,
  options?: KernelOptions,
): BRepResult3D {
  return executeBRep3DInternal("difference", [a, b], options);
}

/** Alias for {@link difference3D}. */
export const difference = difference3D;

/**
 * Compute the 3D Boolean symmetric difference (`A ⊕ B`) of two planar B-Rep solids.
 */
export function xor3D(
  a: Solid3D,
  b: Solid3D,
  options?: KernelOptions,
): BRepResult3D {
  return executeBRep3DInternal("xor", [a, b], options);
}

/** Alias for {@link xor3D}. */
export const xor = xor3D;

/**
 * Heal a single 3D planar face soup or self-intersecting polyhedron into a closed 2-manifold B-Rep.
 */
export function heal3D(solid: Solid3D, options?: KernelOptions): BRepResult3D {
  return executeBRep3DInternal("heal", [solid], options);
}

/** Alias for {@link heal3D}. */
export const heal = heal3D;

/**
 * Dispatch a 3D Boolean operation by {@link BooleanOp}.
 */
export function clip3D(
  op: BooleanOp,
  a: Solid3D,
  b: Solid3D,
  options?: KernelOptions,
): BRepResult3D {
  return executeBRep3DInternal(op, [a, b], options);
}

/** Alias for {@link clip3D}. */
export const clip = clip3D;

/**
 * Pre-load two {@link Solid3D} bodies into WASM linear memory for repeated or benchmarked
 * 3D Boolean execution, returning flat TypedArray views of the output B-Rep and triangle mesh.
 */
export function prepareBooleanTask3D(
  a: Solid3D,
  b: Solid3D,
  options?: KernelOptions,
): { run: (op: BooleanOp) => WasmBRep3DResult } {
  const wasm = getWasm();
  serializeSolidsToWasm([a, b]);

  const ruleCode = ruleToCode(options?.windingRule ?? "positive");
  const snapTol = options?.snapTolerance ?? 0;
  const simplify = (options?.simplifyCollinear ?? true) ? 1 : 0;

  return {
    run: (op: BooleanOp): WasmBRep3DResult => {
      const faceCount = wasm.wasm_brep3d_execute(opToCode(op), ruleCode, snapTol, simplify);
      const coordsPtr = wasm.wasm_brep3d_get_out_coords_ptr();
      const coordsLen = wasm.wasm_brep3d_get_out_coords_len();
      const loopMetaPtr = wasm.wasm_brep3d_get_out_loop_meta_ptr();
      const loopCount = wasm.wasm_brep3d_get_out_loop_count();
      const normalsPtr = wasm.wasm_brep3d_get_out_face_normals_ptr();
      const trisPtr = wasm.wasm_brep3d_get_out_triangles_ptr();
      const trisLen = wasm.wasm_brep3d_get_out_triangles_len();

      const coords = new Float64Array(wasm.memory.buffer, coordsPtr, coordsLen);
      const loopMeta = new Uint32Array(wasm.memory.buffer, loopMetaPtr, loopCount * 4);
      const faceNormals = new Float64Array(wasm.memory.buffer, normalsPtr, faceCount * 3);
      const triangles = new Float64Array(wasm.memory.buffer, trisPtr, trisLen);

      return {
        faceCount,
        loopCount,
        triangleCount: Math.floor(trisLen / 13),
        totalVolume: wasm.wasm_brep3d_get_out_volume(),
        totalSurfaceArea: wasm.wasm_brep3d_get_out_area(),
        vertexCount: wasm.wasm_brep3d_get_out_vertex_count(),
        edgeCount: wasm.wasm_brep3d_get_out_edge_count(),
        unmatchedHalfEdgeCount: wasm.wasm_brep3d_get_out_unmatched_he_count(),
        coords,
        loopMeta,
        faceNormals,
        triangles,
      };
    },
  };
}

/** Alias for {@link prepareBooleanTask3D}. */
export const prepareBooleanTask = prepareBooleanTask3D;
