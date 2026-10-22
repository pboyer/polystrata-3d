import { describe, expect, it } from "vitest";
import {
  type BRepResult3D,
  clip3D,
  computeLoopNormal3D,
  createBox3D,
  difference3D,
  extrudePolygon3D,
   type Face3D,
  heal3D,
  intersect3D,
  type Point3D,
  type Solid3D,
  transformSolid3D,
  union3D,
  xor3D,
} from "../src/index.js";

describe("3D Planar B-Rep Boolean & Healing Engine (WASM)", () => {
  it("satisfies Inclusion-Exclusion volume identities and 2-manifold edge pairing on overlapping boxes", () => {
    const boxA = createBox3D([0, 0, 0], [10, 10, 10]);
    const boxB = createBox3D([5, 5, 5], [15, 15, 15]);

    const u = union3D(boxA, boxB);
    const i = intersect3D(boxA, boxB);
    const dAB = difference3D(boxA, boxB);
    const dBA = difference3D(boxB, boxA);
    const x = xor3D(boxA, boxB);

    expect(u.totalVolume).toBeCloseTo(1875, 4);
    expect(i.totalVolume).toBeCloseTo(125, 4);
    expect(dAB.totalVolume).toBeCloseTo(875, 4);
    expect(dBA.totalVolume).toBeCloseTo(875, 4);
    expect(x.totalVolume).toBeCloseTo(1750, 4);

    // Every output solid must be a closed watertight 2-manifold (0 unmatched half-edges)
    expect(u.unmatchedHalfEdgeCount).toBe(0);
    expect(i.unmatchedHalfEdgeCount).toBe(0);
    expect(dAB.unmatchedHalfEdgeCount).toBe(0);
    expect(dBA.unmatchedHalfEdgeCount).toBe(0);
    expect(x.unmatchedHalfEdgeCount).toBe(0);

    // Surface triangles should also be populated for 3D rendering
    expect(u.triangles.length).toBeGreaterThan(12);
    expect(i.triangles.length).toBe(12);
  });

  it("unifies coplanar faces on kissing boxes and punches interior holes on through-cuts", () => {
    const left = createBox3D([0, 0, 0], [10, 10, 10]);
    const right = createBox3D([10, 0, 0], [20, 10, 10]);

    const u = union3D(left, right);
    expect(u.totalVolume).toBeCloseTo(2000, 4);
    expect(u.solid.faces.length).toBe(6);
    expect(u.unmatchedHalfEdgeCount).toBe(0);

    const pillar = createBox3D([3, 3, -2], [7, 7, 12]);
    const holed = difference3D(left, pillar);
    expect(holed.totalVolume).toBeCloseTo(840, 4);
    expect(holed.unmatchedHalfEdgeCount).toBe(0);

    const facesWithHoles = holed.solid.faces.filter((f) => f.holes.length > 0);
    expect(facesWithHoles.length).toBe(2);
    // Exact 16 canonical vertices (8 outer + 8 hole) and 24 edges, with 0 spurious diagonal splits
    expect(holed.vertexCount).toBe(16);
    expect(holed.edgeCount).toBe(24);

    // Verify triangulated mesh area and volume match exact B-Rep surface area and volume
    let meshArea = 0;
    let meshVol6 = 0;
    for (const t of holed.triangles) {
      const ux = t.b.x - t.a.x;
      const uy = t.b.y - t.a.y;
      const uz = t.b.z - t.a.z;
      const vx = t.c.x - t.a.x;
      const vy = t.c.y - t.a.y;
      const vz = t.c.z - t.a.z;
      const cx = uy * vz - uz * vy;
      const cy = uz * vx - ux * vz;
      const cz = ux * vy - uy * vx;
      expect(cx * t.normal.x + cy * t.normal.y + cz * t.normal.z).toBeGreaterThan(0);
      meshArea += 0.5 * Math.hypot(cx, cy, cz);
      meshVol6 += t.a.x * cx + t.a.y * cy + t.a.z * cz;
    }
    expect(meshArea).toBeCloseTo(holed.totalSurfaceArea, 4);
    expect(meshVol6 / 6).toBeCloseTo(holed.totalVolume, 4);
  });

  it("handles extruded 2D polygons with holes, rotated 3D solids, and heal3D", () => {
    const outerL: [number, number][] = [
      [0, 0],
      [14, 0],
      [14, 5],
      [5, 5],
      [5, 14],
      [0, 14],
    ];
    const lSolid = extrudePolygon3D(outerL, [], 0, 10);
    // Area = 14*5 + 5*9 = 115; height = 10 => vol = 1150
    const healedL = heal3D(lSolid);
    expect(healedL.totalVolume).toBeCloseTo(1150, 3);
    expect(healedL.unmatchedHalfEdgeCount).toBe(0);

    // Rotated box cutting diagonally across the L-bracket corner
    const rawCut = createBox3D([-4, -4, 2], [4, 4, 8]);
    const angle = Math.PI / 6;
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    const rotatedCut = transformSolid3D(rawCut, (p) => ({
      x: 5 + p.x * cosA - p.y * sinA,
      y: 5 + p.x * sinA + p.y * cosA,
      z: p.z,
    }));

    const u = clip3D("union", lSolid, rotatedCut);
    const i = clip3D("intersect", lSolid, rotatedCut);
    const d = clip3D("difference", lSolid, rotatedCut);

    expect(u.totalVolume + i.totalVolume).toBeCloseTo(1150 + 8 * 8 * 6, 2);
    expect(d.totalVolume + i.totalVolume).toBeCloseTo(1150, 2);
    expect(u.unmatchedHalfEdgeCount).toBe(0);
    expect(i.unmatchedHalfEdgeCount).toBe(0);
    expect(d.unmatchedHalfEdgeCount).toBe(0);
  });

  it("preserves closed cap triangulations without missing teeth at high snapTolerance (r0 = 0.282)", () => {
    const gearPts: [number, number][] = [];
    const teeth = 8;
    for (let i = 0; i < teeth * 2; i++) {
      const r = i % 2 === 0 ? 6.8 : 4.6;
      const a = (i * Math.PI) / teeth;
      gearPts.push([r * Math.cos(a), r * Math.sin(a)]);
    }
    const solidA = extrudePolygon3D(gearPts, [], -3, 3);

    const hexPts: [number, number][] = [];
    for (let i = 0; i < 6; i++) {
      const a = (i * Math.PI) / 3;
      hexPts.push([2.8 * Math.cos(a), 2.8 * Math.sin(a)]);
    }
    const rawB = extrudePolygon3D(hexPts, [], -5, 5);

    const poses: [number, number][] = [
      [0, 0],
      [2.4, -1.5],
      [2.0, 1.0],
      [-2.0, -1.0],
      [2.2, 0.0],
      [1.8, -1.5],
    ];

    for (const [dx, dy] of poses) {
      const solidB = transformSolid3D(rawB, (p) => ({
        x: p.x + dx,
        y: p.y + dy,
        z: p.z,
      }));
      const res = clip3D("difference", solidA, solidB, {
        snapTolerance: Math.pow(10, -0.55), // ~0.282
      });
      expect(res.unmatchedHalfEdgeCount).toBe(0);

      for (let fi = 0; fi < res.solid.faces.length; fi++) {
        const f = res.solid.faces[fi];
        if (!f || Math.abs(f.normal.z) < 0.9) continue;

        let sx = 0,
          sy = 0,
          sz = 0;
        for (let i = 0; i < f.outer.length; i++) {
          const a = f.outer[i]!;
          const b = f.outer[(i + 1) % f.outer.length]!;
          sx += a.y * b.z - a.z * b.y;
          sy += a.z * b.x - a.x * b.z;
          sz += a.x * b.y - a.y * b.x;
        }
        let facePolyArea = 0.5 * Math.hypot(sx, sy, sz);
        for (const h of f.holes) {
          let hsx = 0,
            hsy = 0,
            hsz = 0;
          for (let i = 0; i < h.length; i++) {
            const a = h[i]!;
            const b = h[(i + 1) % h.length]!;
            hsx += a.y * b.z - a.z * b.y;
            hsy += a.z * b.x - a.x * b.z;
            hsz += a.x * b.y - a.y * b.x;
          }
          facePolyArea -= 0.5 * Math.hypot(hsx, hsy, hsz);
        }

        const faceTris = res.triangles.filter((t) => t.faceIndex === fi);
        let faceTriArea = 0;
        for (const t of faceTris) {
          const ux = t.b.x - t.a.x,
            uy = t.b.y - t.a.y,
            uz = t.b.z - t.a.z;
          const vx = t.c.x - t.a.x,
            vy = t.c.y - t.a.y,
            vz = t.c.z - t.a.z;
          const cx = uy * vz - uz * vy,
            cy = uz * vx - ux * vz,
            cz = ux * vy - uy * vx;
          faceTriArea += 0.5 * Math.hypot(cx, cy, cz);
        }

        expect(faceTriArea).toBeCloseTo(facePolyArea, 3);
      }
    }
  });

  it("maintains watertight 2-manifold topology (unmatchedHalfEdgeCount = 0) and planar faces under extreme snapTolerance (up to 0.5)", () => {
    const boxA = createBox3D([-5, -5, -5], [5, 5, 5]);
    const boxB = transformSolid3D(createBox3D([-4.5, -4.5, -4.5], [4.5, 4.5, 4.5]), (p) => {
      const rx = (-22 * 0.6 * Math.PI) / 180;
      const ry = (-22 * Math.PI) / 180;
      const rz = (-22 * 0.35 * Math.PI) / 180;
      const cx = Math.cos(rx),
        sx = Math.sin(rx);
      const cy = Math.cos(ry),
        sy = Math.sin(ry);
      const cz = Math.cos(rz),
        sz = Math.sin(rz);
      const y1 = p.y * cx - p.z * sx,
        z1 = p.y * sx + p.z * cx;
      const x2 = p.x * cy + z1 * sy,
        z2 = -p.x * sy + z1 * cy;
      const x3 = x2 * cz - y1 * sz,
        y3 = x2 * sz + y1 * cz;
      return { x: x3 + 3.1, y: y3 + 0.8, z: z2 - 0.4 };
    });

    for (const snapTolerance of [1e-4, 0.05, Math.pow(10, -0.55), 0.5]) {
      for (const op of ["union", "intersect", "difference", "xor"] as const) {
        const res = clip3D(op, boxA, boxB, { snapTolerance });
        expect(res.unmatchedHalfEdgeCount).toBe(0);
        expect(res.totalVolume).toBeGreaterThan(0);

        // Verify every output face is strictly planar (Option B triangulates any tolerance-warped polygon)
        for (const f of res.solid.faces) {
          if (f.outer.length > 3 || f.holes.length > 0) {
            const p0 = f.outer[0]!;
            const d0 = f.normal.x * p0.x + f.normal.y * p0.y + f.normal.z * p0.z;
            for (const p of f.outer) {
              const dist = Math.abs(f.normal.x * p.x + f.normal.y * p.y + f.normal.z * p.z - d0);
              expect(dist).toBeLessThan(1e-3);
            }
          }
        }
      }
    }
  });

  it("preserves 100% watertight 2-manifold faces and triangles (0 holes, Euler chi = 2) on Extruded L-Bracket & Diagonal Slot Cutter up to r0 = 0.631", () => {
    const lProfile: [number, number][] = [
      [-6, -6],
      [6, -6],
      [6, -1.5],
      [-1.5, -1.5],
      [-1.5, 6],
      [-6, 6],
    ];
    const solidA = extrudePolygon3D(lProfile, [], -4, 4);
    const rawB = createBox3D([-4.5, -4.5, -2.5], [2.5, 2.5, 5.5]);
    const rad = (30 * Math.PI) / 180;
    const c = Math.cos(rad),
      s = Math.sin(rad);
    const solidB = transformSolid3D(rawB, (p) => {
      const rx = p.x + 1,
        ry = p.y + 1;
      return {
        x: -1 + rx * c - ry * s,
        y: -1 + rx * s + ry * c,
        z: p.z,
      };
    });

    const keyPt = (p: { x: number; y: number; z: number }) =>
      `${p.x.toFixed(7)},${p.y.toFixed(7)},${p.z.toFixed(7)}`;

    for (const snapTolerance of [1e-4, 0.01, 0.1, 0.3162, 0.5, Math.pow(10, -0.2)]) {
      for (const op of ["union", "intersect", "difference", "xor"] as const) {
        const res = clip3D(op, solidA, solidB, {
          snapTolerance,
          simplifyCollinear: true,
        });
        expect(res.unmatchedHalfEdgeCount).toBe(0);

        // Verify 0 unmatched half-edges in B-Rep faces
        const faceHE = new Map<string, number>();
        for (const f of res.solid.faces) {
          for (const ring of [f.outer, ...f.holes]) {
            for (let i = 0; i < ring.length; i++) {
              const u = keyPt(ring[i]!);
              const v = keyPt(ring[(i + 1) % ring.length]!);
              if (u === v) continue;
              const [a, b, sgn] = u < v ? [u, v, 1] : [v, u, -1];
              const k = `${a}|${b}`;
              faceHE.set(k, (faceHE.get(k) ?? 0) + sgn);
            }
          }
        }
        let faceUm = 0;
        for (const v of faceHE.values()) faceUm += Math.abs(v);
        expect(faceUm).toBe(0);

        // Verify 0 unmatched half-edges in triangulated mesh (no missing wall triangles / holes)
        const triHE = new Map<string, number>();
        for (const t of res.triangles) {
          const pts = [t.a, t.b, t.c];
          for (let i = 0; i < 3; i++) {
            const u = keyPt(pts[i]!);
            const v = keyPt(pts[(i + 1) % 3]!);
            if (u === v) continue;
            const [a, b, sgn] = u < v ? [u, v, 1] : [v, u, -1];
            const k = `${a}|${b}`;
            triHE.set(k, (triHE.get(k) ?? 0) + sgn);
          }
        }
        let triUm = 0;
        for (const v of triHE.values()) triUm += Math.abs(v);
        expect(triUm).toBe(0);

        if (op === "difference") {
          expect(res.totalVolume).toBeGreaterThan(450);
          expect(res.totalVolume).toBeLessThan(500);
          const holesCount = res.solid.faces.reduce((acc, f) => acc + f.holes.length, 0);
          const eulerChi =
            res.vertexCount - res.edgeCount + res.solid.faces.length - holesCount;
          expect(eulerChi).toBe(2);
        }
      }
    }
  });

  it("stress-tests open geometry, spheres, cylinders, cones, and toroids in touching and glancing-angle configurations", () => {
    function makeFace(outer: Point3D[], holes: Point3D[][] = []): Face3D {
      return {
        outer,
        holes,
        normal: computeLoopNormal3D(outer),
      };
    }

    function rotateAndShift(
      solid: Solid3D,
      rxDeg: number,
      ryDeg: number,
      rzDeg: number,
      dx: number,
      dy: number,
      dz: number,
    ): Solid3D {
      const rx = (rxDeg * Math.PI) / 180;
      const ry = (ryDeg * Math.PI) / 180;
      const rz = (rzDeg * Math.PI) / 180;
      const cx = Math.cos(rx),
        sx = Math.sin(rx);
      const cy = Math.cos(ry),
        sy = Math.sin(ry);
      const cz = Math.cos(rz),
        sz = Math.sin(rz);
      return transformSolid3D(solid, (p) => {
        const y1 = p.y * cx - p.z * sx;
        const z1 = p.y * sx + p.z * cx;
        const x2 = p.x * cy + z1 * sy;
        const z2 = -p.x * sy + z1 * cy;
        const x3 = x2 * cz - y1 * sz;
        const y3 = x2 * sz + y1 * cz;
        return { x: x3 + dx, y: y3 + dy, z: z2 + dz };
      });
    }

    function createCylinder3D(
      radius: number,
      zMin: number,
      zMax: number,
      segments = 16,
    ): Solid3D {
      const ring: [number, number][] = [];
      for (let i = 0; i < segments; i++) {
        const a = (2 * Math.PI * i) / segments;
        ring.push([radius * Math.cos(a), radius * Math.sin(a)]);
      }
      return extrudePolygon3D(ring, [], zMin, zMax);
    }

    function createCone3D(
      rBottom: number,
      rTop: number,
      zMin: number,
      zMax: number,
      segments = 16,
    ): Solid3D {
      const faces: Face3D[] = [];
      const botPts: Point3D[] = [];
      const topPts: Point3D[] = [];
      for (let i = 0; i < segments; i++) {
        const a = (2 * Math.PI * i) / segments;
        const c = Math.cos(a),
          s = Math.sin(a);
        botPts.push({ x: rBottom * c, y: rBottom * s, z: zMin });
        if (rTop > 1e-9) {
          topPts.push({ x: rTop * c, y: rTop * s, z: zMax });
        }
      }
      // Bottom cap (-Z)
      faces.push(makeFace([...botPts].reverse()));
      if (rTop > 1e-9) {
        // Top cap (+Z)
        faces.push(makeFace(topPts));
        for (let i = 0; i < segments; i++) {
          const j = (i + 1) % segments;
          faces.push(makeFace([botPts[i]!, botPts[j]!, topPts[j]!, topPts[i]!]));
        }
      } else {
        const apex: Point3D = { x: 0, y: 0, z: zMax };
        for (let i = 0; i < segments; i++) {
          const j = (i + 1) % segments;
          faces.push(makeFace([botPts[i]!, botPts[j]!, apex]));
        }
      }
      return { faces };
    }

    function createSphere3D(
      center: Point3D,
      radius: number,
      stacks = 8,
      slices = 12,
    ): Solid3D {
      const faces: Face3D[] = [];
      const ringPt = (st: number, sl: number): Point3D => {
        const phi = -0.5 * Math.PI + (Math.PI * st) / stacks;
        const theta = (2 * Math.PI * (sl % slices)) / slices;
        const cp = Math.cos(phi),
          sp = Math.sin(phi);
        return {
          x: center.x + radius * cp * Math.cos(theta),
          y: center.y + radius * cp * Math.sin(theta),
          z: center.z + radius * sp,
        };
      };
      const southPole: Point3D = { x: center.x, y: center.y, z: center.z - radius };
      const northPole: Point3D = { x: center.x, y: center.y, z: center.z + radius };

      for (let st = 0; st < stacks; st++) {
        for (let sl = 0; sl < slices; sl++) {
          const slNext = (sl + 1) % slices;
          if (st === 0) {
            const p1 = ringPt(1, sl);
            const p2 = ringPt(1, slNext);
            faces.push(makeFace([southPole, p2, p1]));
          } else if (st === stacks - 1) {
            const p0 = ringPt(st, sl);
            const p1 = ringPt(st, slNext);
            faces.push(makeFace([p0, p1, northPole]));
          } else {
            const p00 = ringPt(st, sl);
            const p10 = ringPt(st, slNext);
            const p11 = ringPt(st + 1, slNext);
            const p01 = ringPt(st + 1, sl);
            // Iso-latitude quad on UV sphere has parallel top/bottom edges -> exact planar trapezoid
            faces.push(makeFace([p00, p10, p11, p01]));
          }
        }
      }
      return { faces };
    }

    function createTorus3D(
      majorR: number,
      minorR: number,
      majorSegs = 14,
      minorSegs = 8,
    ): Solid3D {
      const faces: Face3D[] = [];
      const pt = (i: number, j: number): Point3D => {
        const u = (2 * Math.PI * (i % majorSegs)) / majorSegs;
        const v = (2 * Math.PI * (j % minorSegs)) / minorSegs;
        const r = majorR + minorR * Math.cos(v);
        return {
          x: r * Math.cos(u),
          y: r * Math.sin(u),
          z: minorR * Math.sin(v),
        };
      };
      for (let i = 0; i < majorSegs; i++) {
        const iNext = (i + 1) % majorSegs;
        for (let j = 0; j < minorSegs; j++) {
          const jNext = (j + 1) % minorSegs;
          const p00 = pt(i, j);
          const p10 = pt(iNext, j);
          const p11 = pt(iNext, jNext);
          const p01 = pt(i, jNext);
          // Parallel major-circle chords at v and vNext form an exact planar trapezoid
          faces.push(makeFace([p00, p10, p11, p01]));
        }
      }
      return { faces };
    }

    function createOpenBox3D(
      min: [number, number, number],
      max: [number, number, number],
      omitTopAndBottom = false,
    ): Solid3D {
      const full = createBox3D(min, max);
      // Remove +Z top face (and optionally -Z bottom face) to create an open surface sheet/trough
      return {
        faces: full.faces.filter((f) =>
          omitTopAndBottom ? Math.abs(f.normal.z) < 0.5 : f.normal.z < 0.5,
        ),
      };
    }

    function createOpenCylinderShell3D(
      radius: number,
      zMin: number,
      zMax: number,
      segments = 16,
    ): Solid3D {
      const full = createCylinder3D(radius, zMin, zMax, segments);
      // Remove both top and bottom caps so only the open cylindrical tube wall remains
      return {
        faces: full.faces.filter((f) => Math.abs(f.normal.z) < 0.5),
      };
    }

    function createOpenHemisphereDish3D(
      center: Point3D,
      radius: number,
      stacks = 8,
      slices = 12,
    ): Solid3D {
      const full = createSphere3D(center, radius, stacks, slices);
      // Keep only lower hemisphere facets (open bowl/dish with an uncapped rim at z = center.z)
      return {
        faces: full.faces.filter((f) => {
          const avgZ =
            f.outer.reduce((acc, p) => acc + p.z, 0) / Math.max(1, f.outer.length);
          return avgZ < center.z;
        }),
      };
    }

    function computeSolidInputVolume(solid: Solid3D): number {
      let vol6 = 0;
      for (const f of solid.faces) {
        for (const ring of [f.outer, ...f.holes]) {
          if (ring.length < 3) continue;
          const p0 = ring[0]!;
          for (let i = 1; i + 1 < ring.length; i++) {
            const a = ring[i]!;
            const b = ring[i + 1]!;
            vol6 +=
              p0.x * (a.y * b.z - a.z * b.y) +
              p0.y * (a.z * b.x - a.x * b.z) +
              p0.z * (a.x * b.y - a.y * b.x);
          }
        }
      }
      return vol6 / 6;
    }

    function auditResult(res: BRepResult3D) {
      const keyPt = (p: Point3D) =>
        `${p.x.toFixed(6)},${p.y.toFixed(6)},${p.z.toFixed(6)}`;

      const faceHE = new Map<string, number>();
      let maxPlanarityDev = 0;
      let holeLoops = 0;

      for (const f of res.solid.faces) {
        holeLoops += f.holes.length;
        if (f.outer.length > 0) {
          const p0 = f.outer[0]!;
          const d0 = f.normal.x * p0.x + f.normal.y * p0.y + f.normal.z * p0.z;
          for (const ring of [f.outer, ...f.holes]) {
            for (const p of ring) {
              const dev = Math.abs(
                f.normal.x * p.x + f.normal.y * p.y + f.normal.z * p.z - d0,
              );
              if (dev > maxPlanarityDev) maxPlanarityDev = dev;
            }
          }
        }
        for (const ring of [f.outer, ...f.holes]) {
          for (let i = 0; i < ring.length; i++) {
            const u = keyPt(ring[i]!);
            const v = keyPt(ring[(i + 1) % ring.length]!);
            if (u === v) continue;
            const [a, b, sgn] = u < v ? [u, v, 1] : [v, u, -1];
            const k = `${a}|${b}`;
            faceHE.set(k, (faceHE.get(k) ?? 0) + sgn);
          }
        }
      }
      let faceUm = 0;
      for (const v of faceHE.values()) faceUm += Math.abs(v);

      const triHE = new Map<string, number>();
      let meshVol6 = 0;
      let invertedTris = 0;
      for (const t of res.triangles) {
        const pts = [t.a, t.b, t.c];
        for (let i = 0; i < 3; i++) {
          const u = keyPt(pts[i]!);
          const v = keyPt(pts[(i + 1) % 3]!);
          if (u === v) continue;
          const [a, b, sgn] = u < v ? [u, v, 1] : [v, u, -1];
          const k = `${a}|${b}`;
          triHE.set(k, (triHE.get(k) ?? 0) + sgn);
        }
        const ux = t.b.x - t.a.x,
          uy = t.b.y - t.a.y,
          uz = t.b.z - t.a.z;
        const vx = t.c.x - t.a.x,
          vy = t.c.y - t.a.y,
          vz = t.c.z - t.a.z;
        const cx = uy * vz - uz * vy,
          cy = uz * vx - ux * vz,
          cz = ux * vy - uy * vx;
        if (cx * t.normal.x + cy * t.normal.y + cz * t.normal.z <= 0) {
          invertedTris++;
        }
        meshVol6 += t.a.x * cx + t.a.y * cy + t.a.z * cz;
      }
      let triUm = 0;
      for (const v of triHE.values()) triUm += Math.abs(v);

      const meshVol = meshVol6 / 6;
      const meshVolErr = Math.abs(meshVol - res.totalVolume);
      const eulerChi =
        res.vertexCount - res.edgeCount + (res.solid.faces.length - holeLoops);

      return {
        reportedUm: res.unmatchedHalfEdgeCount,
        faceUm,
        triUm,
        maxPlanarityDev,
        invertedTris,
        meshVolErr,
        eulerChi,
        holeLoops,
      };
    }

    interface StressScenario {
      name: string;
      category: string;
      solidA: Solid3D;
      solidB: Solid3D;
      isOpenInput?: boolean;
    }

    // Helper: align Solid B's minX to Solid A's maxX + gap (negative gap = grazing overlap depth)
    function placeTouchingX(solidA: Solid3D, solidB: Solid3D, gap: number): Solid3D {
      let maxXA = -Infinity;
      for (const f of solidA.faces) {
        for (const p of f.outer) if (p.x > maxXA) maxXA = p.x;
      }
      let minXB = Infinity;
      for (const f of solidB.faces) {
        for (const p of f.outer) if (p.x < minXB) minXB = p.x;
      }
      const dx = maxXA - minXB + gap;
      return transformSolid3D(solidB, (p) => ({ x: p.x + dx, y: p.y, z: p.z }));
    }

    // Helper: align Solid B's minZ to Solid A's maxZ + gap (negative gap = grazing overlap depth)
    function placeTouchingZ(solidA: Solid3D, solidB: Solid3D, gap: number): Solid3D {
      let maxZA = -Infinity;
      for (const f of solidA.faces) {
        for (const p of f.outer) if (p.z > maxZA) maxZA = p.z;
      }
      let minZB = Infinity;
      for (const f of solidB.faces) {
        for (const p of f.outer) if (p.z < minZB) minZB = p.z;
      }
      const dz = maxZA - minZB + gap;
      return transformSolid3D(solidB, (p) => ({ x: p.x, y: p.y, z: p.z + dz }));
    }

    const baseSphere = createSphere3D({ x: 0, y: 0, z: 0 }, 4, 8, 12);
    // Rotate sphere by half-slice (15°) and half-stack (11.25°) so a flat quad facet faces +X and +Z
    const facetAlignedSphere = rotateAndShift(baseSphere, 11.25, 0, 15, 0, 0, 0);

    const scenarios: StressScenario[] = [
      // --- 1. SPHERES: Vertex-Kiss, Facet-Kiss, & Shallow Glancing ---
      {
        name: "S1. Two UV-Spheres Externally Kissing at Equator Vertex (gap = 0)",
        category: "Sphere-Sphere Vertex Kiss",
        solidA: createSphere3D({ x: -4, y: 0, z: 0 }, 4, 8, 12),
        solidB: createSphere3D({ x: 4, y: 0, z: 0 }, 4, 8, 12),
      },
      {
        name: "S2. Two Facet-Aligned Spheres Kissing Flush on Shared Tangent Quad (gap = 0)",
        category: "Sphere-Sphere Facet Kiss",
        solidA: facetAlignedSphere,
        solidB: placeTouchingX(facetAlignedSphere, facetAlignedSphere, 0),
      },
      {
        name: "S3. Two UV-Spheres Glancing Overlap (shallow lens, tilt 2.5°)",
        category: "Sphere-Sphere Glancing",
        solidA: baseSphere,
        solidB: placeTouchingX(
          baseSphere,
          rotateAndShift(baseSphere, 2.5, -1.8, 3.2, 0, 0, 0),
          -0.35,
        ),
      },
      {
        name: "S4. Sphere Skimming Planar Box Top Face & Edge at Glancing Angle (tilt 1.8°)",
        category: "Sphere-Box Glancing",
        solidA: createBox3D([-5, -5, -4], [5, 5, 4]),
        solidB: placeTouchingZ(
          createBox3D([-5, -5, -4], [5, 5, 4]),
          rotateAndShift(baseSphere, 1.8, -2.2, 7.5, 2.5, 0, 0),
          -0.35,
        ),
      },
      {
        name: "S5. Hollow Spherical Shell (Inner Cavity) Sliced by 1.2° Glancing Box",
        category: "HollowSphere-Box Glancing",
        solidA: difference3D(
          createSphere3D({ x: 0, y: 0, z: 0 }, 4.5, 8, 12),
          createSphere3D({ x: 0, y: 0, z: 0 }, 2.8, 6, 10),
        ).solid,
        solidB: rotateAndShift(createBox3D([-5, -5, -0.2], [5, 5, 5]), 1.2, -0.9, 4.0, 0, 0, 0),
      },
      {
        name: "S6. Sphere Wedged in Dihedral Corner of L-Bracket (2-Wall Simultaneous Grazing, 1.4°)",
        category: "Sphere-LBracket Dihedral",
        solidA: extrudePolygon3D(
          [
            [-5, -5],
            [5, -5],
            [5, -1],
            [-1, -1],
            [-1, 5],
            [-5, 5],
          ],
          [],
          -4,
          4,
        ),
        solidB: rotateAndShift(createSphere3D({ x: 0, y: 0, z: 0 }, 3.2, 8, 12), 1.4, -1.1, 7.5, 2.05, 2.05, 0),
      },

      // --- 2. CYLINDERS: Generator-Line Kiss, Facet-Kiss, Steinmetz, & Glancing Angle ---
      {
        name: "C1. Two Cylinders Kissing Along Vertical Generator Edge (gap = 0)",
        category: "Cyl-Cyl Edge Kiss",
        solidA: createCylinder3D(3, -5, 5, 16),
        solidB: placeTouchingX(
          createCylinder3D(3, -5, 5, 16),
          createCylinder3D(3, -5, 5, 16),
          0,
        ),
      },
      {
        name: "C2. Two Cylinders Nearly Parallel at 1.75° Glancing Angle",
        category: "Cyl-Cyl Glancing",
        solidA: createCylinder3D(3.5, -6, 6, 16),
        solidB: rotateAndShift(createCylinder3D(3.5, -6, 6, 16), 1.75, 0.8, 7.5, 0.4, 0, 0),
      },
      {
        name: "C3. Orthogonal Equi-Radius Steinmetz Cylinders (Tangent Crown at z = ±3.5)",
        category: "Cyl-Cyl Steinmetz",
        solidA: createCylinder3D(3.5, -6, 6, 16),
        solidB: rotateAndShift(createCylinder3D(3.5, -6, 6, 16), 90, 0, 0, 0, 0, 0),
      },
      {
        name: "C4. Orthogonal Cylinders Grazing Tangentially at Outer Rim (tilt 2°)",
        category: "Cyl-Cyl Rim Grazing",
        solidA: createCylinder3D(3.5, -6, 6, 16),
        solidB: rotateAndShift(createCylinder3D(3.5, -6, 6, 16), 90, 2.0, 1.5, 6.65, 0, 0),
      },
      {
        name: "C5. Hollow Cylindrical Pipe (Coaxial Bore) Pierced by 1.4° Glancing Cylinder",
        category: "HollowPipe-Cyl Glancing",
        solidA: difference3D(
          createCylinder3D(4.2, -5, 5, 16),
          createCylinder3D(2.6, -5, 5, 16),
        ).solid,
        solidB: rotateAndShift(createCylinder3D(2.6, -5, 5, 16), 1.4, -0.8, 11.25, 1.55, 0, 0),
      },
      {
        name: "C6. Cylinder Internally Tangent to Box Wall & Flush with Top/Bottom Caps",
        category: "Cyl-Box Multi-Tangent",
        solidA: createBox3D([-4, -4, -5], [4, 4, 5]),
        solidB: rotateAndShift(createCylinder3D(4.0, -5, 5, 16), 0.9, -0.6, 0, 0, 0, 0),
      },

      // --- 3. CONES: Apex Kissing, Generator Tangency, & Glancing Slant Cuts ---
      {
        name: "K1. Pointed Cone Apex Exactly Kissing Box Bottom Face (z = 0, gap = 0)",
        category: "Cone-Box Apex Kiss",
        solidA: createBox3D([-5, -5, 0], [5, 5, 6]),
        solidB: createCone3D(4, 0, -6, 0, 16),
      },
      {
        name: "K2. Pointed Cone Apex Piercing Box Face by δ = 0.25 at 2.5° Glancing Tilt",
        category: "Cone-Box Apex Glancing",
        solidA: createBox3D([-5, -5, 0], [5, 5, 6]),
        solidB: rotateAndShift(createCone3D(4, 0, -6, 0.25, 16), 2.5, -1.5, 0, 0, 0, 0),
      },
      {
        name: "K3. Cone & Cylinder Coaxial Base-to-Cap Kissing & Glancing Slant Wall (2.2°)",
        category: "Cone-Cyl Glancing",
        solidA: createCylinder3D(4, -4, 4, 16),
        solidB: rotateAndShift(createCone3D(5, 1.5, -4, 4, 16), 2.2, 0, 11.25, 0.15, 0, 0),
      },
      {
        name: "K4. Two Cones Glancing Along Slant Generators (1.5° off-tangent)",
        category: "Cone-Cone Slant Glancing",
        solidA: createCone3D(4, 0, -4, 4, 16),
        solidB: rotateAndShift(createCone3D(4, 0, -4, 4, 16), 180 + 1.5, 1.0, 11.25, 3.75, 0, 0),
      },
      {
        name: "K5. Apex-to-Apex Double Cone (Hourglass Singularity) Sliced by Glancing Cylinder",
        category: "Hourglass-Cyl Singularity",
        solidA: createCone3D(4, 0, -5, 0.05, 16),
        solidB: rotateAndShift(createCone3D(4, 0, -5, 0.05, 16), 178.8, 1.1, 11.25, 0, 0, 0),
      },
      {
        name: "K6. Frustum & Inverted Frustum Kissing on Mid-Plane Ring at 0.9° Glancing Tilt",
        category: "Frustum-Frustum Ring Glancing",
        solidA: createCone3D(4.5, 2.5, -4, 0, 16),
        solidB: rotateAndShift(createCone3D(2.5, 4.5, 0, 4, 16), 0.9, -0.7, 11.25, 0.05, 0, -0.02),
      },

      // --- 4. TOROIDS: Genus-1 Tangent Rings, Interlocking & Glancing ---
      {
        name: "T1. Torus Sitting Tangent on Planar Box Top (Bottom Ring Kissing z = 0)",
        category: "Torus-Box Ring Kiss",
        solidA: createBox3D([-8, -8, -4], [8, 8, 0]),
        solidB: rotateAndShift(createTorus3D(4.5, 1.5, 14, 8), 0, 0, 0, 0, 0, 1.5),
      },
      {
        name: "T2. Torus Grazing Planar Box Top at 1.8° Glancing Tilt (δ = 0.04)",
        category: "Torus-Box Glancing",
        solidA: createBox3D([-8, -8, -4], [8, 8, 0]),
        solidB: rotateAndShift(createTorus3D(4.5, 1.5, 14, 8), 1.8, -1.2, 0, 0, 0, 1.46),
      },
      {
        name: "T3. Cylinder Snugly Threading Torus Inner Hole (r_cyl = R - r = 3.0, Glancing 1.5°)",
        category: "Torus-Cyl Tangent Bore",
        solidA: createTorus3D(4.5, 1.5, 14, 8),
        solidB: rotateAndShift(createCylinder3D(3.15, -4, 4, 14), 1.5, 0, 0, 0, 0, 0),
      },
      {
        name: "T4. Two Toroids Grazing Externally at 3° Glancing Angle",
        category: "Torus-Torus Glancing",
        solidA: createTorus3D(4.0, 1.4, 12, 8),
        solidB: rotateAndShift(createTorus3D(4.0, 1.4, 12, 8), 3.0, -2.0, 15.0, 10.35, 0, 0),
      },
      {
        name: "T5. Sphere Seated in Torus Saddle (Grazing Inner Gorge at Glancing Angle)",
        category: "Torus-Sphere Saddle",
        solidA: createTorus3D(4.5, 1.5, 14, 8),
        solidB: rotateAndShift(createSphere3D({ x: 0, y: 0, z: 0 }, 3.25, 8, 14), 2.0, -1.5, 5.0, 0.1, 0, 0),
      },
      {
        name: "T6. Two Interlocking Hopf-Link Toroids Grazing Inner Rings at 88° Glancing Angle",
        category: "Torus-Torus Hopf Grazing",
        solidA: createTorus3D(4.5, 1.4, 14, 8),
        solidB: rotateAndShift(createTorus3D(4.5, 1.4, 14, 8), 88.0, 1.5, 0, 4.5, 0, 0),
      },
      {
        name: "T7. Coaxial Stacked Toroids Kissing Along Poloidal Rings at 1.1° Glancing Tilt",
        category: "Torus-Torus Coaxial Stack",
        solidA: createTorus3D(4.2, 1.4, 14, 8),
        solidB: rotateAndShift(createTorus3D(4.2, 1.4, 14, 8), 1.1, -0.8, 12.85, 0, 0, 2.62),
      },
      {
        name: "T8. Pointed Cone Threading & Grazing Torus Inner Gorge at 1.6° Glancing Angle",
        category: "Torus-Cone Gorge Grazing",
        solidA: createTorus3D(4.5, 1.5, 14, 8),
        solidB: rotateAndShift(createCone3D(4.2, 1.8, -4, 4, 14), 1.6, -1.1, 12.85, 0.1, 0, 0),
      },

      // --- 5. COMPLEX EXTRUSIONS, MULTI-HOLE FLANGES, & INTERMESHING GEARS ---
      {
        name: "P1. Extruded Multi-Hole Flange Plate Cut by 1.3° Glancing Frustum",
        category: "MultiHoleFlange-Cone",
        solidA: extrudePolygon3D(
          [
            [-6, -6],
            [6, -6],
            [6, 6],
            [-6, 6],
          ],
          [
            [
              [-4, -4],
              [-1, -4],
              [-1, -1],
              [-4, -1],
            ],
            [
              [1, 1],
              [4, 1],
              [4, 4],
              [1, 4],
            ],
          ],
          -2,
          2,
        ),
        solidB: rotateAndShift(createCone3D(4.5, 2.0, -4, 4, 16), 1.3, -0.9, 7.5, 0.2, 0.2, 0),
      },
      {
        name: "P2. Intermeshing Extruded Comb Teeth Grazing Along Flanks at 0.8° Glancing Tilt",
        category: "Comb-Comb Tooth Grazing",
        solidA: extrudePolygon3D(
          [
            [-6, -4],
            [6, -4],
            [6, 0],
            [4, 0],
            [4, 3],
            [2, 3],
            [2, 0],
            [-2, 0],
            [-2, 3],
            [-4, 3],
            [-4, 0],
            [-6, 0],
          ],
          [],
          -3,
          3,
        ),
        solidB: rotateAndShift(
          extrudePolygon3D(
            [
              [-5, 0],
              [-4, 0],
              [-4, 4],
              [4, 4],
              [4, 0],
              [5, 0],
              [5, 5],
              [-5, 5],
            ],
            [],
            -3,
            3,
          ),
          0.8,
          -0.6,
          0.5,
          0,
          -1.0,
          0,
        ),
      },
      {
        name: "P3. Non-Convex Star Prism Grazing Torus at 1.9° Glancing Angle",
        category: "StarPrism-Torus Glancing",
        solidA: (() => {
          const pts: [number, number][] = [];
          const n = 10;
          for (let i = 0; i < n; i++) {
            const a = (2 * Math.PI * i) / n;
            const r = i % 2 === 0 ? 5.0 : 2.5;
            pts.push([r * Math.cos(a), r * Math.sin(a)]);
          }
          return extrudePolygon3D(pts, [], -3, 3);
        })(),
        solidB: rotateAndShift(createTorus3D(3.8, 1.3, 12, 8), 1.9, -1.4, 9.0, 0.2, 0, 0),
      },
      {
        name: "P4. Chained CSG Assembly ((Box ∪ Cyl) \\ Cone) Intersected with Glancing Sphere",
        category: "ChainedCSG-Sphere",
        solidA: difference3D(
          union3D(createBox3D([-4, -4, -3], [4, 4, 3]), createCylinder3D(2.8, -5, 5, 14)).solid,
          createCone3D(2.2, 0.8, -5.5, 5.5, 12),
        ).solid,
        solidB: rotateAndShift(createSphere3D({ x: 0, y: 0, z: 0 }, 4.2, 8, 12), 1.5, -1.2, 6.0, 1.2, 0.5, 0),
      },

      // --- 6. OPEN GEOMETRY: Uncapped Shells, Open Boxes, Dishes, & Flaps ---
      {
        name: "O1. Open Uncapped Box (5 faces, missing +Z top) Intersecting Closed Sphere",
        category: "OpenBox-Sphere",
        solidA: createOpenBox3D([-4, -4, -4], [4, 4, 4], false),
        solidB: createSphere3D({ x: 0, y: 0, z: 4 }, 3.5, 8, 12),
        isOpenInput: true,
      },
      {
        name: "O2. Open Cylindrical Tube Wall (no caps) Grazing Closed Box at Glancing Angle",
        category: "OpenCyl-Box Glancing",
        solidA: createOpenCylinderShell3D(3.5, -5, 5, 16),
        solidB: rotateAndShift(createBox3D([-4, -4, -3], [4, 4, 3]), 2.0, 1.5, 10, 3.4, 0, 0),
        isOpenInput: true,
      },
      {
        name: "O3. Open Hemisphere Dish Kissing & Slicing Closed Cone",
        category: "OpenDish-Cone",
        solidA: createOpenHemisphereDish3D({ x: 0, y: 0, z: 0 }, 4.5, 8, 12),
        solidB: rotateAndShift(createCone3D(3.5, 0, -4.5, 3.0, 14), 1.5, 0, 0, 0, 0, 0),
        isOpenInput: true,
      },
      {
        name: "O4. Two Open Half-Boxes Kissing Along Rim (z = 0) Forming Closed Cube on Union/Heal",
        category: "OpenBox-OpenBox Kiss",
        solidA: createOpenBox3D([-4, -4, -4], [4, 4, 0], false),
        solidB: rotateAndShift(createOpenBox3D([-4, -4, -4], [4, 4, 0], false), 180, 0, 0, 0, 0, 0),
        isOpenInput: true,
      },
      {
        name: "O5. Open Conical Funnel (No Caps) Slicing Closed Torus at 1.7° Glancing Angle",
        category: "OpenFunnel-Torus",
        solidA: {
          faces: createCone3D(4.8, 1.8, -4, 4, 16).faces.filter((f) => Math.abs(f.normal.z) < 0.8),
        },
        solidB: rotateAndShift(createTorus3D(3.8, 1.4, 12, 8), 1.7, -1.1, 8.0, 0.3, 0, 0),
        isOpenInput: true,
      },
      {
        name: "O6. Closed Box with Dangling External Fin Flap Unioned with Glancing Cylinder",
        category: "FlapPruning-BoxCyl",
        solidA: {
          faces: [
            ...createBox3D([-4, -4, -4], [4, 4, 4]).faces,
            // Single dangling unclosed quad flap protruding from the +X face
            {
              outer: [
                { x: 4, y: -2, z: -2 },
                { x: 7, y: -2, z: -2 },
                { x: 7, y: 2, z: 2 },
                { x: 4, y: 2, z: 2 },
              ],
              holes: [],
              normal: { x: 0, y: -0.70710678, z: 0.70710678 },
            },
          ],
        },
        solidB: rotateAndShift(createCylinder3D(3.2, -5, 5, 14), 1.6, -1.0, 5.0, 3.5, 0, 0),
        isOpenInput: true,
      },
    ];

    const tolerances: { label: string; snapTolerance?: number }[] = [
      { label: "Auto" },
      { label: "1e-4", snapTolerance: 1e-4 },
      { label: "1e-2", snapTolerance: 1e-2 },
      { label: "0.10", snapTolerance: 0.1 },
    ];

    const reportLines: string[] = [];
    let totalOps = 0;
    let totalExceptions = 0;
    let totalFaceNonManifold = 0;
    let totalTriNonManifold = 0;
    let totalReportedMismatch = 0;
    let totalNonPlanar = 0;
    let totalInvertedTris = 0;
    let totalIEViolations = 0;
    let totalEmptyOutputsOnOpen = 0;

    for (const sc of scenarios) {
      const volA = computeSolidInputVolume(sc.solidA);
      const volB = computeSolidInputVolume(sc.solidB);

      for (const tolCfg of tolerances) {
        const opts =
          tolCfg.snapTolerance !== undefined
            ? { snapTolerance: tolCfg.snapTolerance }
            : undefined;
        const t0 = performance.now();

        try {
          const u = union3D(sc.solidA, sc.solidB, opts);
          const i = intersect3D(sc.solidA, sc.solidB, opts);
          const d = difference3D(sc.solidA, sc.solidB, opts);
          const x = xor3D(sc.solidA, sc.solidB, opts);
          const hAB = heal3D({ faces: [...sc.solidA.faces, ...sc.solidB.faces] }, opts);
          const elapsed = performance.now() - t0;
          totalOps += 5;

          const au = auditResult(u);
          const ai = auditResult(i);
          const ad = auditResult(d);
          const ax = auditResult(x);
          const ah = auditResult(hAB);

          for (const [opName, res, aud] of [
            ["U", u, au],
            ["I", i, ai],
            ["D", d, ad],
            ["X", x, ax],
            ["H", hAB, ah],
          ] as const) {
            if (aud.faceUm > 0) totalFaceNonManifold++;
            if (aud.triUm > 0) totalTriNonManifold++;
            if ((aud.reportedUm === 0) !== (aud.faceUm === 0)) totalReportedMismatch++;
            if (aud.maxPlanarityDev > 1e-3) totalNonPlanar++;
            if (aud.invertedTris > 0) totalInvertedTris++;
            if (sc.isOpenInput && res.solid.faces.length === 0) {
              totalEmptyOutputsOnOpen++;
            }
          }

          const ieErr = Math.abs(u.totalVolume + i.totalVolume - (volA + volB));
          const diffErr = Math.abs(d.totalVolume + i.totalVolume - volA);
          const tolScale = Math.max(0.5, (tolCfg.snapTolerance ?? 1e-4) * 250);
          if (!sc.isOpenInput && ieErr > tolScale) {
            totalIEViolations++;
          }

          const anyIssue =
            au.faceUm > 0 ||
            ai.faceUm > 0 ||
            ad.faceUm > 0 ||
            ax.faceUm > 0 ||
            ah.faceUm > 0 ||
            au.triUm > 0 ||
            ai.triUm > 0 ||
            ad.triUm > 0 ||
            ax.triUm > 0 ||
            ah.triUm > 0 ||
            au.invertedTris > 0 ||
            ai.invertedTris > 0 ||
            ad.invertedTris > 0 ||
            ax.invertedTris > 0 ||
            (!sc.isOpenInput && ieErr > tolScale) ||
            (sc.isOpenInput && u.solid.faces.length === 0);

          reportLines.push(
            `${anyIssue ? "[ISSUE]" : "[ OK  ]"} ${sc.name.slice(0, 42).padEnd(42)} | tol=${tolCfg.label.padEnd(4)} | ` +
              `um(F/T): U=${au.faceUm}/${au.triUm} I=${ai.faceUm}/${ai.triUm} D=${ad.faceUm}/${ad.triUm} X=${ax.faceUm}/${ax.triUm} H=${ah.faceUm}/${ah.triUm} | ` +
              `vol(U/I/D/H)=${u.totalVolume.toFixed(1)}/${i.totalVolume.toFixed(1)}/${d.totalVolume.toFixed(1)}/${hAB.totalVolume.toFixed(1)} | ` +
              `ieErr=${ieErr.toExponential(1)} diffErr=${diffErr.toExponential(1)} | ` +
              `chi(U/D)=${au.eulerChi}/${ad.eulerChi} | invTri=${au.invertedTris + ai.invertedTris + ad.invertedTris + ax.invertedTris} | ${elapsed.toFixed(0)}ms`,
          );
        } catch (err) {
          totalExceptions++;
          reportLines.push(
            `[THROW] ${sc.name.slice(0, 42).padEnd(42)} | tol=${tolCfg.label.padEnd(4)} | ERROR: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
      }
    }

    console.log("\n=== 3D ADVERSARIAL GEOMETRY STRESS TEST REPORT ===");
    console.log(
      `Scenarios: ${scenarios.length} | Tolerances: ${tolerances.length} | Total 3D Ops: ${totalOps}`,
    );
    console.log(
      `Exceptions: ${totalExceptions} | Face Non-Manifold (faceUm>0): ${totalFaceNonManifold} | Tri Mesh Holes (triUm>0): ${totalTriNonManifold}`,
    );
    console.log(
      `Reported-vs-Actual Um Mismatch: ${totalReportedMismatch} | Non-Planar Faces (>1e-3): ${totalNonPlanar} | Inverted Tris: ${totalInvertedTris}`,
    );
    console.log(
      `Closed-Input IE Volume Violations: ${totalIEViolations} | Open-Input Empty Results: ${totalEmptyOutputsOnOpen}`,
    );
    console.log("--------------------------------------------------");
    for (const line of reportLines) {
      console.log(line);
    }
    console.log("==================================================\n");

    expect(scenarios.length).toBe(36);
    expect(totalExceptions).toBe(0);
    expect(totalFaceNonManifold).toBe(0);
    expect(totalTriNonManifold).toBe(0);
    expect(totalReportedMismatch).toBe(0);
    expect(totalNonPlanar).toBe(0);
    expect(totalInvertedTris).toBe(0);
    expect(totalIEViolations).toBe(0);
  });
});
