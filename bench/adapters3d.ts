import jscad from "@jscad/modeling";
import * as THREE from "three";
import { ADDITION, Brush, Evaluator, INTERSECTION, SUBTRACTION } from "three-bvh-csg";
import {
  heal3D,
  type Point3D,
  prepareBooleanTask3D,
  type Solid3D,
  type WasmBRep3DResult,
} from "../src/index.js";
import { getManifoldWasm } from "./manifold-wasm.js";
import type { BenchOp3D } from "./workloads3d.js";

const { booleans, geometries } = jscad;
const { geom3 } = geometries;

export interface InspectedResult3D {
  volume: number;
  faceCount: number;
  triangleCount: number;
  unmatchedHalfEdges: number;
}

export interface PreparedTask3D {
  run: (op: BenchOp3D) => unknown;
  inspect: (result: unknown) => InspectedResult3D;
  dispose?: () => void;
}

export interface LibraryAdapter3D {
  id: string;
  name: string;
  version: string;
  algorithm: string;
  representation: string;
  prepare: (solidA: Solid3D, solidB: Solid3D) => PreparedTask3D;
}

function earClip3DRing(ring: readonly Point3D[], normal: Point3D): [Point3D, Point3D, Point3D][] {
  const n = ring.length;
  if (n < 3) return [];
  if (n === 3) return [[ring[0]!, ring[1]!, ring[2]!]];
  if (n === 4) {
    return [
      [ring[0]!, ring[1]!, ring[2]!],
      [ring[0]!, ring[2]!, ring[3]!],
    ];
  }

  // Project onto 2D tangent frame of `normal`
  const ax = Math.abs(normal.x);
  const ay = Math.abs(normal.y);
  const az = Math.abs(normal.z);
  const ref: Point3D =
    ax < ay && ax < az ? { x: 1, y: 0, z: 0 } : ay < az ? { x: 0, y: 1, z: 0 } : { x: 0, y: 0, z: 1 };
  const uxRaw = normal.y * ref.z - normal.z * ref.y;
  const uyRaw = normal.z * ref.x - normal.x * ref.z;
  const uzRaw = normal.x * ref.y - normal.y * ref.x;
  const uLen = Math.hypot(uxRaw, uyRaw, uzRaw) || 1;
  const ux = uxRaw / uLen;
  const uy = uyRaw / uLen;
  const uz = uzRaw / uLen;
  const vx = normal.y * uz - normal.z * uy;
  const vy = normal.z * ux - normal.x * uz;
  const vz = normal.x * uy - normal.y * ux;

  const pts2d = ring.map((p) => ({
    x: p.x * ux + p.y * uy + p.z * uz,
    y: p.x * vx + p.y * vy + p.z * vz,
  }));

  const indices = ring.map((_, i) => i);
  const tris: [Point3D, Point3D, Point3D][] = [];

  const cross2 = (ia: number, ib: number, ic: number): number => {
    const a = pts2d[ia]!;
    const b = pts2d[ib]!;
    const c = pts2d[ic]!;
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  };

  const ptInTri = (ip: number, ia: number, ib: number, ic: number): boolean => {
    const p = pts2d[ip]!;
    const a = pts2d[ia]!;
    const b = pts2d[ib]!;
    const c = pts2d[ic]!;
    const c1 = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
    const c2 = (c.x - b.x) * (p.y - b.y) - (c.y - b.y) * (p.x - b.x);
    const c3 = (a.x - c.x) * (p.y - c.y) - (a.y - c.y) * (p.x - c.x);
    return c1 >= -1e-10 && c2 >= -1e-10 && c3 >= -1e-10;
  };

  let guard = 0;
  while (indices.length > 3 && guard < n * n) {
    guard++;
    let clipped = false;
    const m = indices.length;
    for (let i = 0; i < m; i++) {
      const iPrev = indices[(i + m - 1) % m]!;
      const iCurr = indices[i]!;
      const iNext = indices[(i + 1) % m]!;
      if (cross2(iPrev, iCurr, iNext) <= 1e-10) continue;
      let hasInside = false;
      for (let j = 0; j < m; j++) {
        const idx = indices[j]!;
        if (idx === iPrev || idx === iCurr || idx === iNext) continue;
        if (ptInTri(idx, iPrev, iCurr, iNext)) {
          hasInside = true;
          break;
        }
      }
      if (!hasInside) {
        tris.push([ring[iPrev]!, ring[iCurr]!, ring[iNext]!]);
        indices.splice(i, 1);
        clipped = true;
        break;
      }
    }
    if (!clipped) {
      const i0 = indices[0]!;
      const i1 = indices[1]!;
      const i2 = indices[2]!;
      tris.push([ring[i0]!, ring[i1]!, ring[i2]!]);
      indices.splice(1, 1);
    }
  }
  if (indices.length === 3) {
    tris.push([ring[indices[0]!]!, ring[indices[1]!]!, ring[indices[2]!]!]);
  }
  return tris;
}

/**
 * Convert a `Solid3D` into a list of oriented triangles `[a, b, c]`.
 * - If any face has interior holes, uses `heal3D` to bridge holes and triangulate watertightly.
 * - Otherwise ear-clips each face's outer ring directly so non-manifold fins or raw vertices are preserved as-is.
 */
export function solidToTriangles3D(solid: Solid3D): [Point3D, Point3D, Point3D][] {
  const hasHoles = solid.faces.some((f) => f.holes.length > 0);
  if (hasHoles) {
    const healed = heal3D(solid, { simplifyCollinear: false });
    return healed.triangles.map((t) => [t.a, t.b, t.c]);
  }
  const tris: [Point3D, Point3D, Point3D][] = [];
  for (const face of solid.faces) {
    const faceTris = earClip3DRing(face.outer, face.normal);
    for (const t of faceTris) tris.push(t);
  }
  return tris;
}

/**
 * Weld a triangle soup into an indexed mesh `(vertProperties: Float32Array, triVerts: Uint32Array)`
 * suitable for `manifold-3d` and `three-bvh-csg`.
 */
export function weldTrianglesToIndexedMesh(tris: readonly [Point3D, Point3D, Point3D][]): {
  vertProperties: Float32Array;
  triVerts: Uint32Array;
} {
  const vertMap = new Map<string, number>();
  const coords: number[] = [];
  const indices: number[] = [];

  const getIdx = (p: Point3D): number => {
    // Quantize to f32 precision so shared vertices weld cleanly in Float32Array
    const fx = Math.fround(p.x);
    const fy = Math.fround(p.y);
    const fz = Math.fround(p.z);
    const key = `${Math.round(fx * 1e5)},${Math.round(fy * 1e5)},${Math.round(fz * 1e5)}`;
    const existing = vertMap.get(key);
    if (existing !== undefined) return existing;
    const idx = coords.length / 3;
    coords.push(fx, fy, fz);
    vertMap.set(key, idx);
    return idx;
  };

  for (const [a, b, c] of tris) {
    const ia = getIdx(a);
    const ib = getIdx(b);
    const ic = getIdx(c);
    if (ia !== ib && ib !== ic && ic !== ia) {
      indices.push(ia, ib, ic);
    }
  }

  return {
    vertProperties: new Float32Array(coords),
    triVerts: new Uint32Array(indices),
  };
}

/**
 * Audit an indexed triangle mesh `(vertProperties, triVerts)` for unmatched half-edges
 * (`∂₁ · ∂₂ = 0`) and compute its divergence-theorem volume.
 */
function auditIndexedMesh(
  vertProperties: ArrayLike<number>,
  triVerts: ArrayLike<number>,
): { volume: number; triangleCount: number; unmatchedHalfEdges: number } {
  const triCount = Math.floor(triVerts.length / 3);
  const edgeBalance = new Map<string, number>();
  let vol6 = 0;

  const vKey = (idx: number): string => {
    const x = vertProperties[idx * 3] ?? 0;
    const y = vertProperties[idx * 3 + 1] ?? 0;
    const z = vertProperties[idx * 3 + 2] ?? 0;
    return `${Math.round(x * 1e4)},${Math.round(y * 1e4)},${Math.round(z * 1e4)}`;
  };

  const addHe = (u: string, v: string): void => {
    if (u === v) return;
    const [k, s] = u < v ? [`${u}|${v}`, 1] : [`${v}|${u}`, -1];
    edgeBalance.set(k, (edgeBalance.get(k) ?? 0) + s);
  };

  for (let t = 0; t < triCount; t++) {
    const ia = triVerts[t * 3]!;
    const ib = triVerts[t * 3 + 1]!;
    const ic = triVerts[t * 3 + 2]!;

    const ax = vertProperties[ia * 3] ?? 0;
    const ay = vertProperties[ia * 3 + 1] ?? 0;
    const az = vertProperties[ia * 3 + 2] ?? 0;
    const bx = vertProperties[ib * 3] ?? 0;
    const by = vertProperties[ib * 3 + 1] ?? 0;
    const bz = vertProperties[ib * 3 + 2] ?? 0;
    const cx = vertProperties[ic * 3] ?? 0;
    const cy = vertProperties[ic * 3 + 1] ?? 0;
    const cz = vertProperties[ic * 3 + 2] ?? 0;

    vol6 += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);

    const ka = vKey(ia);
    const kb = vKey(ib);
    const kc = vKey(ic);
    addHe(ka, kb);
    addHe(kb, kc);
    addHe(kc, ka);
  }

  let unmatchedHalfEdges = 0;
  for (const bal of edgeBalance.values()) {
    unmatchedHalfEdges += Math.abs(bal);
  }

  return {
    volume: vol6 / 6,
    triangleCount: triCount,
    unmatchedHalfEdges,
  };
}

export const LIBRARY_ADAPTERS_3D: LibraryAdapter3D[] = [
  {
    id: "polystrata-3d",
    name: "polystrata-3d",
    version: "0.1.0",
    algorithm: "3D Stratified Tolerance Complex + Solid-Angle Winding + Planar Unification",
    representation: "Planar B-Rep (Polygons + Holes)",
    prepare: (solidA, solidB) => {
      const wasmTask = prepareBooleanTask3D(solidA, solidB);
      return {
        run: (op) => wasmTask.run(op),
        inspect: (res) => {
          const r = res as WasmBRep3DResult;
          return {
            volume: r.totalVolume,
            faceCount: r.faceCount,
            triangleCount: r.triangleCount,
            unmatchedHalfEdges: r.unmatchedHalfEdgeCount,
          };
        },
      };
    },
  },

  {
    id: "manifold-3d",
    name: "manifold-3d",
    version: "3.5.4",
    algorithm: "C++/WASM Symbolic Perturbation + Half-Edge Mesh CSG (Emmett Lalish)",
    representation: "Indexed Triangle Mesh",
    prepare: (solidA, solidB) => {
      const wasm = getManifoldWasm();
      const meshDataA = weldTrianglesToIndexedMesh(solidToTriangles3D(solidA));
      const meshDataB = weldTrianglesToIndexedMesh(solidToTriangles3D(solidB));
      const manA = new wasm.Manifold(new wasm.Mesh({ numProp: 3, ...meshDataA }));
      const manB = new wasm.Manifold(new wasm.Mesh({ numProp: 3, ...meshDataB }));

      return {
        run: (op) => {
          let out;
          switch (op) {
            case "union":
              out = wasm.Manifold.union(manA, manB);
              break;
            case "intersect":
              out = wasm.Manifold.intersection(manA, manB);
              break;
            case "difference":
              out = wasm.Manifold.difference(manA, manB);
              break;
          }
          const mesh = out.getMesh();
          const volume = out.volume();
          out.delete();
          return { mesh, volume };
        },
        inspect: (res) => {
          const { mesh, volume } = res as {
            mesh: { vertProperties: Float32Array; triVerts: Uint32Array };
            volume: number;
          };
          const audit = auditIndexedMesh(mesh.vertProperties, mesh.triVerts);
          return {
            volume,
            faceCount: audit.triangleCount,
            triangleCount: audit.triangleCount,
            unmatchedHalfEdges: audit.unmatchedHalfEdges,
          };
        },
        dispose: () => {
          manA.delete();
          manB.delete();
        },
      };
    },
  },

  {
    id: "three-bvh-csg",
    name: "three-bvh-csg",
    version: "0.0.18",
    algorithm: "BVH-Accelerated Triangle-Splitting CSG (three-mesh-bvh)",
    representation: "THREE.BufferGeometry",
    prepare: (solidA, solidB) => {
      const toBrush = (solid: Solid3D): Brush => {
        const { vertProperties, triVerts } = weldTrianglesToIndexedMesh(solidToTriangles3D(solid));
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(vertProperties, 3));
        geom.setIndex(new THREE.BufferAttribute(triVerts, 1));
        geom.computeVertexNormals();
        const brush = new Brush(geom);
        brush.updateMatrixWorld();
        brush.prepareGeometry();
        return brush;
      };

      const brushA = toBrush(solidA);
      const brushB = toBrush(solidB);
      const evaluator = new Evaluator();
      evaluator.useGroups = false;
      evaluator.attributes = ["position"];

      return {
        run: (op) => {
          const csgOp =
            op === "union" ? ADDITION : op === "intersect" ? INTERSECTION : SUBTRACTION;
          return evaluator.evaluate(brushA, brushB, csgOp);
        },
        inspect: (res) => {
          const mesh = res as Brush;
          const geom = mesh.geometry;
          const posAttr = geom.getAttribute("position");
          const indexAttr = geom.getIndex();
          if (!posAttr || posAttr.count === 0) {
            return { volume: 0, faceCount: 0, triangleCount: 0, unmatchedHalfEdges: 0 };
          }
          const verts = posAttr.array as Float32Array;
          const indices = indexAttr
            ? (indexAttr.array as Uint32Array | Uint16Array)
            : Uint32Array.from({ length: posAttr.count }, (_, i) => i);
          const audit = auditIndexedMesh(verts, indices);
          return {
            volume: audit.volume,
            faceCount: audit.triangleCount,
            triangleCount: audit.triangleCount,
            unmatchedHalfEdges: audit.unmatchedHalfEdges,
          };
        },
        dispose: () => {
          brushA.geometry.dispose();
          brushB.geometry.dispose();
        },
      };
    },
  },

  {
    id: "jscad-modeling",
    name: "@jscad/modeling",
    version: "2.13.0",
    algorithm: "3D Binary Space Partitioning (BSP) Convex-Polygon CSG",
    representation: "Convex Polygon Soup (geom3)",
    prepare: (solidA, solidB) => {
      const toGeom3 = (solid: Solid3D) => {
        const tris = solidToTriangles3D(solid);
        const polys = tris.map(([a, b, c]) => [
          [a.x, a.y, a.z] as [number, number, number],
          [b.x, b.y, b.z] as [number, number, number],
          [c.x, c.y, c.z] as [number, number, number],
        ]);
        return geom3.fromPoints(polys);
      };

      const gA = toGeom3(solidA);
      const gB = toGeom3(solidB);

      return {
        run: (op) => {
          switch (op) {
            case "union":
              return booleans.union(gA, gB);
            case "intersect":
              return booleans.intersect(gA, gB);
            case "difference":
              return booleans.subtract(gA, gB);
          }
        },
        inspect: (res) => {
          const g = res as ReturnType<typeof geom3.fromPoints>;
          const polys = geom3.toPolygons(g);
          const tris: [Point3D, Point3D, Point3D][] = [];
          for (const poly of polys) {
            const v = poly.vertices;
            for (let i = 1; i + 1 < v.length; i++) {
              const p0 = v[0]!;
              const p1 = v[i]!;
              const p2 = v[i + 1]!;
              tris.push([
                { x: p0[0], y: p0[1], z: p0[2] },
                { x: p1[0], y: p1[1], z: p1[2] },
                { x: p2[0], y: p2[1], z: p2[2] },
              ]);
            }
          }
          const { vertProperties, triVerts } = weldTrianglesToIndexedMesh(tris);
          const audit = auditIndexedMesh(vertProperties, triVerts);
          return {
            volume: audit.volume,
            faceCount: polys.length,
            triangleCount: audit.triangleCount,
            unmatchedHalfEdges: audit.unmatchedHalfEdges,
          };
        },
      };
    },
  },
];
