import {
  type BooleanOp,
  type BRepResult3D,
  clip3D,
  createBox3D,
  difference3D,
  extrudePolygon3D,
   type Face3D,
  type KernelOptions,
  intersect3D,
  type Point3D,
  type Solid3D,
  transformSolid3D,
  union3D,
} from "../src/index.js";

export type Scenario3DCategory =
  | "prisms"
  | "spheres"
  | "cylinders"
  | "cones"
  | "toroids"
  | "open";

export interface Transform3DParams {
  dx: number;
  dy: number;
  dz: number;
  rotDeg: number;
  rxDeg?: number;
  ryDeg?: number;
  detailScale?: number;
}

export interface Scenario3D {
  id: string;
  title: string;
  badge: string;
  category: Scenario3DCategory;
  description: string;
  defaultOp: BooleanOp | "heal";
  buildSolids: (offset: Transform3DParams) => { solidA: Solid3D; solidB: Solid3D };
}

function computePolygonNormal(pts: readonly Point3D[]): Point3D {
  let nx = 0;
  let ny = 0;
  let nz = 0;
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    const curr = pts[i]!;
    const next = pts[(i + 1) % n]!;
    nx += (curr.y - next.y) * (curr.z + next.z);
    ny += (curr.z - next.z) * (curr.x + next.x);
    nz += (curr.x - next.x) * (curr.y + next.y);
  }
  const len = Math.hypot(nx, ny, nz);
  if (len <= 1e-15) return { x: 0, y: 0, z: 1 };
  return { x: nx / len, y: ny / len, z: nz / len };
}

function makeQuadOrTriFaces(
  p00: Point3D,
  p10: Point3D,
  p11: Point3D,
  p01: Point3D,
): Face3D[] {
  const cop0 = p00.x - p11.x;
  const cop1 = p00.y - p11.y;
  const cop2 = p00.z - p11.z;
  const e1x = p10.x - p00.x;
  const e1y = p10.y - p00.y;
  const e1z = p10.z - p00.z;
  const e2x = p01.x - p00.x;
  const e2y = p01.y - p00.y;
  const e2z = p01.z - p00.z;
  const cx = e1y * e2z - e1z * e2y;
  const cy = e1z * e2x - e1x * e2z;
  const cz = e1x * e2y - e1y * e2x;
  const clen = Math.hypot(cx, cy, cz);
  const nonPlanar =
    clen > 1e-12 ? Math.abs(cop0 * cx + cop1 * cy + cop2 * cz) / clen : 0;

  if (nonPlanar <= 1e-9) {
    const outer = [p00, p10, p11, p01];
    return [{ outer, holes: [], normal: computePolygonNormal(outer) }];
  }
  const t1 = [p00, p10, p11];
  const t2 = [p00, p11, p01];
  return [
    { outer: t1, holes: [], normal: computePolygonNormal(t1) },
    { outer: t2, holes: [], normal: computePolygonNormal(t2) },
  ];
}

export function createCylinder3D(
  radius: number,
  zMin: number,
  zMax: number,
  segments = 16,
  center: [number, number] = [0, 0],
): Solid3D {
  const ring: [number, number][] = [];
  for (let i = 0; i < segments; i++) {
    const a = (2 * Math.PI * i) / segments;
    ring.push([
      center[0] + radius * Math.cos(a),
      center[1] + radius * Math.sin(a),
    ]);
  }
  return extrudePolygon3D(ring, [], zMin, zMax);
}

export function createCone3D(
  rBottom: number,
  rTop: number,
  zMin: number,
  zMax: number,
  segments = 16,
  center: [number, number] = [0, 0],
): Solid3D {
  const faces: Face3D[] = [];
  const botRing: Point3D[] = [];
  const topRing: Point3D[] = [];
  for (let i = 0; i < segments; i++) {
    const a = (2 * Math.PI * i) / segments;
    const c = Math.cos(a);
    const s = Math.sin(a);
    botRing.push({ x: center[0] + rBottom * c, y: center[1] + rBottom * s, z: zMin });
    topRing.push({ x: center[0] + rTop * c, y: center[1] + rTop * s, z: zMax });
  }

  if (rBottom > 1e-9) {
    faces.push({
      outer: [...botRing].reverse(),
      holes: [],
      normal: { x: 0, y: 0, z: -1 },
    });
  }
  if (rTop > 1e-9) {
    faces.push({
      outer: [...topRing],
      holes: [],
      normal: { x: 0, y: 0, z: 1 },
    });
  }

  const apexTop: Point3D = { x: center[0], y: center[1], z: zMax };
  const apexBot: Point3D = { x: center[0], y: center[1], z: zMin };

  for (let i = 0; i < segments; i++) {
    const next = (i + 1) % segments;
    if (rTop <= 1e-9) {
      const outer = [botRing[i]!, botRing[next]!, apexTop];
      faces.push({ outer, holes: [], normal: computePolygonNormal(outer) });
    } else if (rBottom <= 1e-9) {
      const outer = [apexBot, topRing[next]!, topRing[i]!];
      faces.push({ outer, holes: [], normal: computePolygonNormal(outer) });
    } else {
      faces.push(
        ...makeQuadOrTriFaces(
          botRing[i]!,
          botRing[next]!,
          topRing[next]!,
          topRing[i]!,
        ),
      );
    }
  }
  return { faces };
}

export function createSphere3D(
  radius: number,
  center: [number, number, number] = [0, 0, 0],
  latBands = 8,
  lonSegments = 12,
): Solid3D {
  const faces: Face3D[] = [];
  const [cx, cy, cz] = center;

  const ringAt = (latIdx: number): Point3D[] => {
    const phi = -Math.PI / 2 + (Math.PI * latIdx) / latBands;
    const r = radius * Math.cos(phi);
    const z = cz + radius * Math.sin(phi);
    const pts: Point3D[] = [];
    for (let j = 0; j < lonSegments; j++) {
      const theta = (2 * Math.PI * j) / lonSegments;
      pts.push({
        x: cx + r * Math.cos(theta),
        y: cy + r * Math.sin(theta),
        z,
      });
    }
    return pts;
  };

  const southPole: Point3D = { x: cx, y: cy, z: cz - radius };
  const northPole: Point3D = { x: cx, y: cy, z: cz + radius };

  const firstRing = ringAt(1);
  for (let j = 0; j < lonSegments; j++) {
    const next = (j + 1) % lonSegments;
    const outer = [southPole, firstRing[next]!, firstRing[j]!];
    faces.push({ outer, holes: [], normal: computePolygonNormal(outer) });
  }

  for (let i = 1; i < latBands - 1; i++) {
    const r0 = ringAt(i);
    const r1 = ringAt(i + 1);
    for (let j = 0; j < lonSegments; j++) {
      const next = (j + 1) % lonSegments;
      faces.push(...makeQuadOrTriFaces(r0[j]!, r0[next]!, r1[next]!, r1[j]!));
    }
  }

  const lastRing = ringAt(latBands - 1);
  for (let j = 0; j < lonSegments; j++) {
    const next = (j + 1) % lonSegments;
    const outer = [lastRing[j]!, lastRing[next]!, northPole];
    faces.push({ outer, holes: [], normal: computePolygonNormal(outer) });
  }

  return { faces };
}

export function createTorus3D(
  majorRadius: number,
  minorRadius: number,
  center: [number, number, number] = [0, 0, 0],
  majorSegments = 14,
  minorSegments = 8,
): Solid3D {
  const [cx, cy, cz] = center;
  const grid: Point3D[][] = [];

  for (let i = 0; i < majorSegments; i++) {
    const u = (2 * Math.PI * i) / majorSegments;
    const cu = Math.cos(u);
    const su = Math.sin(u);
    const ring: Point3D[] = [];
    for (let j = 0; j < minorSegments; j++) {
      const v = (2 * Math.PI * j) / minorSegments;
      const cv = Math.cos(v);
      const sv = Math.sin(v);
      const r = majorRadius + minorRadius * cv;
      ring.push({
        x: cx + r * cu,
        y: cy + r * su,
        z: cz + minorRadius * sv,
      });
    }
    grid.push(ring);
  }

  const faces: Face3D[] = [];
  for (let i = 0; i < majorSegments; i++) {
    const nextI = (i + 1) % majorSegments;
    for (let j = 0; j < minorSegments; j++) {
      const nextJ = (j + 1) % minorSegments;
      const p00 = grid[i]![j]!;
      const p10 = grid[nextI]![j]!;
      const p11 = grid[nextI]![nextJ]!;
      const p01 = grid[i]![nextJ]!;
      faces.push(...makeQuadOrTriFaces(p00, p10, p11, p01));
    }
  }
  return { faces };
}

function rotateEulerAndTranslate(
  solid: Solid3D,
  rxDeg: number,
  ryDeg: number,
  rzDeg: number,
  tx: number,
  ty: number,
  tz: number,
  pivot: Point3D = { x: 0, y: 0, z: 0 },
): Solid3D {
  const rx = (rxDeg * Math.PI) / 180;
  const ry = (ryDeg * Math.PI) / 180;
  const rz = (rzDeg * Math.PI) / 180;
  const cx = Math.cos(rx);
  const sx = Math.sin(rx);
  const cy = Math.cos(ry);
  const sy = Math.sin(ry);
  const cz = Math.cos(rz);
  const sz = Math.sin(rz);

  return transformSolid3D(solid, (p) => {
    const px = p.x - pivot.x;
    const py = p.y - pivot.y;
    const pz = p.z - pivot.z;
    // Rotate X
    const y1 = py * cx - pz * sx;
    const z1 = py * sx + pz * cx;
    // Rotate Y
    const x2 = px * cy + z1 * sy;
    const z2 = -px * sy + z1 * cy;
    // Rotate Z
    const x3 = x2 * cz - y1 * sz;
    const y3 = x2 * sz + y1 * cz;
    return {
      x: pivot.x + x3 + tx,
      y: pivot.y + y3 + ty,
      z: pivot.z + z2 + tz,
    };
  });
}

function applyLiveTransform(
  solid: Solid3D,
  offset: Transform3DParams,
  pivot: Point3D = { x: 0, y: 0, z: 0 },
  baseEuler: [number, number, number] = [0, 0, 0],
): Solid3D {
  return rotateEulerAndTranslate(
    solid,
    baseEuler[0] + (offset.rxDeg ?? 0),
    baseEuler[1] + (offset.ryDeg ?? 0),
    baseEuler[2] + offset.rotDeg,
    offset.dx,
    offset.dy,
    offset.dz,
    pivot,
  );
}

export const DEMO_SCENARIOS_3D: Scenario3D[] = [
  // =========================================================================
  // PRISMS & EXTRUSIONS
  // =========================================================================
  {
    id: "through-hole-block",
    title: "1. Machined Block & Through-Pillar",
    badge: "Face Holes",
    category: "prisms",
    description:
      "A vertical prism punches cleanly through a machined base block, generating interior CW hole loops on the top and bottom planar faces with 0 unmatched half-edges.",
    defaultOp: "difference",
    buildSolids: (offset) => {
      const solidA = createBox3D([-6, -6, -4], [6, 6, 4]);
      const rawB = createBox3D([-2.5, -2.5, -6.5], [2.5, 2.5, 6.5]);
      const solidB = applyLiveTransform(rawB, offset);
      return { solidA, solidB };
    },
  },
  {
    id: "overlapping-cubes",
    title: "2. Corner-Overlapping Hexahedra",
    badge: "Classic 3D",
    category: "prisms",
    description:
      "Two 10×10×10 cubes intersecting in a 5×5×5 octant. Drag the Transform sliders to sweep Solid B through coplanar, edge-kissing, and glancing configurations.",
    defaultOp: "union",
    buildSolids: (offset) => {
      const solidA = createBox3D([-6, -6, -5], [4, 4, 5]);
      const rawB = createBox3D([-1, -1, -2], [7, 7, 6]);
      const solidB = applyLiveTransform(rawB, offset, { x: 3, y: 3, z: 2 });
      return { solidA, solidB };
    },
  },
  {
    id: "kissing-coplanar",
    title: "3. Coplanar Kissing Blocks (Face Unification)",
    badge: "Coplanar",
    category: "prisms",
    description:
      "Two 8×8×8 blocks sharing the exact plane x = 0. Union automatically dissolves the shared interior wall and unifies the 4 pairs of coplanar side faces into 6 clean faces.",
    defaultOp: "union",
    buildSolids: (offset) => {
      const solidA = createBox3D([-8, -4, -4], [0, 4, 4]);
      const rawB = createBox3D([0, -4, -4], [8, 4, 4]);
      const solidB = applyLiveTransform(rawB, offset, { x: 4, y: 0, z: 0 });
      return { solidA, solidB };
    },
  },
  {
    id: "extruded-l-bracket",
    title: "4. Extruded L-Bracket & Diagonal Slot Cutter",
    badge: "Extrusion",
    category: "prisms",
    description:
      "An extruded 2D L-bracket profile intersected by a rotated cutter box across its inner corner, testing non-orthogonal carrier planes and T-junction subdivision.",
    defaultOp: "difference",
    buildSolids: (offset) => {
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
      const solidB = applyLiveTransform(
        rawB,
        offset,
        { x: -1, y: -1, z: 0 },
        [0, 0, 30],
      );
      return { solidA, solidB };
    },
  },
  {
    id: "star-gear-prism",
    title: "5. 8-Tooth Star Gear & Hexagonal Bore",
    badge: "Gear B-Rep",
    category: "prisms",
    description:
      "A 16-vertex non-convex star gear extruded into 3D and Boolean-differenced with a hexagonal shaft bore.",
    defaultOp: "difference",
    buildSolids: (offset) => {
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
      const solidB = applyLiveTransform(rawB, offset);
      return { solidA, solidB };
    },
  },
  {
    id: "hollow-frame-beam",
    title: "6. Hollow Picture-Frame Prism & Cross-Beam",
    badge: "Multi-Hole",
    category: "prisms",
    description:
      "An extruded hollow rectangular tube (with an existing interior hole loop) intersected by a horizontal cross-beam.",
    defaultOp: "union",
    buildSolids: (offset) => {
      const outer: [number, number][] = [
        [-6, -6],
        [6, -6],
        [6, 6],
        [-6, 6],
      ];
      const hole: [number, number][] = [
        [-3.5, -3.5],
        [-3.5, 3.5],
        [3.5, 3.5],
        [3.5, -3.5],
      ];
      const solidA = extrudePolygon3D(outer, [hole], -3.5, 3.5);
      const rawB = createBox3D([-7.5, -1.8, -1.8], [7.5, 1.8, 1.8]);
      const solidB = applyLiveTransform(rawB, offset);
      return { solidA, solidB };
    },
  },
  {
    id: "flange-plate-frustum",
    title: "7. Multi-Hole Flange Plate & Glancing Frustum",
    badge: "Flange + Cone",
    category: "prisms",
    description:
      "An extruded flange plate with two square bolt holes cut by a 1.3° glancing conical frustum that simultaneously breaches both holes.",
    defaultOp: "difference",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const outer: [number, number][] = [
        [-6, -6],
        [6, -6],
        [6, 6],
        [-6, 6],
      ];
      const hole1: [number, number][] = [
        [-4.0, -4.0],
        [-1.0, -4.0],
        [-1.0, -1.0],
        [-4.0, -1.0],
      ];
      const hole2: [number, number][] = [
        [1.0, 1.0],
        [4.0, 1.0],
        [4.0, 4.0],
        [1.0, 4.0],
      ];
      const solidA = extrudePolygon3D(outer, [hole1, hole2], -2, 2);
      const rawB = createCone3D(4.5, 2.0, -4, 4, Math.round(16 * s));
      const solidB = applyLiveTransform(
        rawB,
        { ...offset, dx: offset.dx + 0.2, dy: offset.dy + 0.2 },
        { x: 0, y: 0, z: 0 },
        [1.3, -0.9, 7.5],
      );
      return { solidA, solidB };
    },
  },
  {
    id: "intermeshing-comb-teeth",
    title: "8. Intermeshing Comb Teeth (1.1° Glancing)",
    badge: "Comb Array",
    category: "prisms",
    description:
      "A 3-tooth extruded comb intermeshed with a counter-comb tilted by 1.1°, creating multiple simultaneous glancing cheek-to-cheek contacts.",
    defaultOp: "union",
    buildSolids: (offset) => {
      const combA: [number, number][] = [
        [-6, -4],
        [6, -4],
        [6, 4],
        [4, 4],
        [4, -1],
        [1, -1],
        [1, 4],
        [-1, 4],
        [-1, -1],
        [-4, -1],
        [-4, 4],
        [-6, 4],
      ];
      const solidA = extrudePolygon3D(combA, [], -2.5, 2.5);
      const rawB = createBox3D([-3.95, -0.95, -2.0], [3.95, 3.5, 2.0]);
      const solidB = applyLiveTransform(
        rawB,
        offset,
        { x: 0, y: 1, z: 0 },
        [1.1, 0.8, 0.5],
      );
      return { solidA, solidB };
    },
  },

  // =========================================================================
  // SPHERES
  // =========================================================================
  {
    id: "spheres-kissing-equator",
    title: "9. UV-Spheres Externally Kissing at Equator",
    badge: "Kissing Spheres",
    category: "spheres",
    description:
      "Two polyhedral UV-spheres touching externally at a single equatorial vertex (x = 0). Drag Δx negative to transition from point contact into a shallow lens intersection.",
    defaultOp: "union",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const lat = Math.max(6, Math.round(8 * s));
      const lon = Math.max(8, Math.round(12 * s));
      const solidA = createSphere3D(4.0, [-4.0, 0, 0], lat, lon);
      const rawB = createSphere3D(4.0, [4.0, 0, 0], lat, lon);
      const solidB = applyLiveTransform(rawB, offset, { x: 4, y: 0, z: 0 });
      return { solidA, solidB };
    },
  },
  {
    id: "spheres-glancing-overlap",
    title: "10. UV-Spheres Shallow Glancing Lens",
    badge: "Glancing Lens",
    category: "spheres",
    description:
      "Two UV-spheres overlapping by Δx = 0.25 with a 3.5° glancing axial tilt, producing a thin convex lens intersection with high-degree seam vertices.",
    defaultOp: "intersect",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const lat = Math.max(6, Math.round(8 * s));
      const lon = Math.max(8, Math.round(12 * s));
      const solidA = createSphere3D(4.0, [-3.875, 0, 0], lat, lon);
      const rawB = createSphere3D(4.0, [0, 0, 0], lat, lon);
      const solidB = applyLiveTransform(
        rawB,
        { ...offset, dx: offset.dx + 3.875, dy: offset.dy + 0.08, dz: offset.dz - 0.05 },
        { x: 0, y: 0, z: 0 },
        [3.5, -2.0, 7.5],
      );
      return { solidA, solidB };
    },
  },
  {
    id: "sphere-skimming-box",
    title: "11. UV-Sphere Skimming Planar Box Face",
    badge: "Sphere + Box",
    category: "spheres",
    description:
      "A UV-sphere grazing the top z = 4 plane of a machined block with a 0.15 shallow indentation and 1.2° tilt, carving a spherical dish into the planar face.",
    defaultOp: "difference",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const lat = Math.max(6, Math.round(8 * s));
      const lon = Math.max(8, Math.round(12 * s));
      const solidA = createBox3D([-5, -5, -4], [5, 5, 4]);
      const rawB = createSphere3D(4.0, [0, 0, 0], lat, lon);
      const solidB = applyLiveTransform(
        rawB,
        { ...offset, dx: offset.dx + 0.3, dy: offset.dy - 0.2, dz: offset.dz + 7.85 },
        { x: 0, y: 0, z: 0 },
        [1.2, 0.8, 0],
      );
      return { solidA, solidB };
    },
  },
  {
    id: "hollow-spherical-shell",
    title: "12. Hollow Spherical Shell & Diagonal Cutter",
    badge: "Hollow Cavity",
    category: "spheres",
    description:
      "A concentric hollow spherical shell (outer R=4.5, inner cavity R=3.0 with inverted normals) sliced open by a tilted box to expose the interior spherical wall.",
    defaultOp: "difference",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const lat = Math.max(6, Math.round(8 * s));
      const lon = Math.max(8, Math.round(12 * s));
      const outerSphere = createSphere3D(4.5, [0, 0, 0], lat, lon);
      const innerCavity = transformSolid3D(
        createSphere3D(3.0, [0, 0, 0], lat, lon),
        (p) => p,
      );
      const flippedInner: Face3D[] = innerCavity.faces.map((f) => ({
        outer: [...f.outer].reverse(),
        holes: f.holes.map((h) => [...h].reverse()),
        normal: { x: -f.normal.x, y: -f.normal.y, z: -f.normal.z },
      }));
      const solidA: Solid3D = { faces: [...outerSphere.faces, ...flippedInner] };
      const rawB = createBox3D([0, -5, -5], [6, 5, 5]);
      const solidB = applyLiveTransform(rawB, offset, { x: 2, y: 0, z: 0 }, [2.0, -1.5, 4.0]);
      return { solidA, solidB };
    },
  },
  {
    id: "sphere-wedged-l-bracket",
    title: "13. Sphere Wedged in L-Bracket Dihedral Corner",
    badge: "Dual Tangent",
    category: "spheres",
    description:
      "A UV-sphere simultaneously tangent to both inner orthogonal walls of an extruded L-bracket, creating dual circular contact patches.",
    defaultOp: "union",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const lat = Math.max(6, Math.round(8 * s));
      const lon = Math.max(8, Math.round(12 * s));
      const lProfile: [number, number][] = [
        [-6, -6],
        [6, -6],
        [6, -2],
        [-2, -2],
        [-2, 6],
        [-6, 6],
      ];
      const solidA = extrudePolygon3D(lProfile, [], -4, 4);
      const rawB = createSphere3D(3.2, [1.1, 1.1, 0], lat, lon);
      const solidB = applyLiveTransform(rawB, offset, { x: 1.1, y: 1.1, z: 0 });
      return { solidA, solidB };
    },
  },

  // =========================================================================
  // CYLINDERS
  // =========================================================================
  {
    id: "cylinders-kissing-generator",
    title: "14. Two Cylinders Kissing Along Generator",
    badge: "Line Contact",
    category: "cylinders",
    description:
      "Two vertical 16-gon cylinders touching along a shared vertical side edge at x = 0 with coplanar top and bottom caps (z = ±4).",
    defaultOp: "union",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const segs = Math.max(12, Math.round(16 * s));
      const solidA = createCylinder3D(3.5, -4, 4, segs, [-3.5, 0]);
      const rawB = createCylinder3D(3.5, -4, 4, segs, [3.5, 0]);
      const solidB = applyLiveTransform(rawB, offset, { x: 3.5, y: 0, z: 0 });
      return { solidA, solidB };
    },
  },
  {
    id: "cylinders-glancing-175deg",
    title: "15. Nearly Parallel Cylinders at 1.75° Glancing",
    badge: "1.75° Glancing",
    category: "cylinders",
    description:
      "Two coaxial cylinders where Cylinder B is tilted by a 1.75° glancing angle, producing long sliver-like quadric facet intersections across the side walls.",
    defaultOp: "difference",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const segs = Math.max(12, Math.round(16 * s));
      const solidA = createCylinder3D(3.5, -5, 5, segs, [0, 0]);
      const rawB = createCylinder3D(3.5, -5, 5, segs, [0, 0]);
      const solidB = applyLiveTransform(
        rawB,
        { ...offset, dx: offset.dx + 0.4, dy: offset.dy + 0.1 },
        { x: 0, y: 0, z: 0 },
        [1.75, -0.8, 3.0],
      );
      return { solidA, solidB };
    },
  },
  {
    id: "steinmetz-orthogonal-cylinders",
    title: "16. Orthogonal Equi-Radius Steinmetz Bicylinder",
    badge: "Steinmetz",
    category: "cylinders",
    description:
      "Classic Steinmetz solid: two equal-radius cylinders (R = 3.5) with perpendicular axes (Z and X) whose surfaces tangentially kiss at z = ±R, y = ±R.",
    defaultOp: "intersect",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const segs = Math.max(12, Math.round(16 * s));
      const solidA = createCylinder3D(3.5, -6, 6, segs, [0, 0]);
      const rawB = rotateEulerAndTranslate(
        createCylinder3D(3.5, -6, 6, segs, [0, 0]),
        0,
        90,
        0,
        0,
        0,
        0,
      );
      const solidB = applyLiveTransform(rawB, offset);
      return { solidA, solidB };
    },
  },
  {
    id: "orthogonal-cylinders-grazing",
    title: "17. Orthogonal Cylinders Grazing Tangentially",
    badge: "Saddle Graze",
    category: "cylinders",
    description:
      "Two perpendicular cylinders shifted so their curved outer walls barely graze by 0.12 units at a single saddle point.",
    defaultOp: "union",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const segs = Math.max(12, Math.round(16 * s));
      const solidA = createCylinder3D(3.5, -6, 6, segs, [0, 0]);
      const rawB = rotateEulerAndTranslate(
        createCylinder3D(3.5, -6, 6, segs, [0, 0]),
        1.5,
        90,
        0,
        0,
        6.88,
        0,
      );
      const solidB = applyLiveTransform(rawB, offset, { x: 0, y: 6.88, z: 0 });
      return { solidA, solidB };
    },
  },
  {
    id: "hollow-pipe-glancing-cylinder",
    title: "18. Hollow Pipe Cut by 2° Glancing Cylinder",
    badge: "Pipe + Bore",
    category: "cylinders",
    description:
      "An extruded hollow cylindrical pipe (outer R=4.5, inner bore R=2.5) intersected by a tilted cylinder that skims both the outer wall and inner bore.",
    defaultOp: "difference",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const segs = Math.max(12, Math.round(16 * s));
      const outerRing: [number, number][] = [];
      const innerHole: [number, number][] = [];
      for (let i = 0; i < segs; i++) {
        const a = (2 * Math.PI * i) / segs;
        outerRing.push([4.5 * Math.cos(a), 4.5 * Math.sin(a)]);
      }
      for (let i = segs - 1; i >= 0; i--) {
        const a = (2 * Math.PI * i) / segs;
        innerHole.push([2.5 * Math.cos(a), 2.5 * Math.sin(a)]);
      }
      const solidA = extrudePolygon3D(outerRing, [innerHole], -4, 4);
      const rawB = createCylinder3D(2.8, -5, 5, segs, [0, 0]);
      const solidB = applyLiveTransform(
        rawB,
        { ...offset, dx: offset.dx + 2.5, dy: offset.dy + 0.2 },
        { x: 0, y: 0, z: 0 },
        [18.0, 2.0, 0],
      );
      return { solidA, solidB };
    },
  },

  // =========================================================================
  // CONES
  // =========================================================================
  {
    id: "double-cone-hourglass-sphere",
    title: "19. Double-Cone Hourglass + Sphere at Waist",
    badge: "Apex Singularity",
    category: "cones",
    description:
      "Two cones joined into an hourglass waist intersected by a UV-sphere centered near the high-valence apex pinch.",
    defaultOp: "difference",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const segs = Math.max(12, Math.round(16 * s));
      const lat = Math.max(6, Math.round(8 * s));
      const lon = Math.max(8, Math.round(12 * s));
      const lower = createCone3D(4.0, 0.8, -4, 0, segs);
      const upper = createCone3D(0.8, 4.0, 0, 4, segs);
      const solidA = union3D(lower, upper).solid;
      const rawB = createSphere3D(2.6, [1.2, 0.2, 0.1], lat, lon);
      const solidB = applyLiveTransform(rawB, offset, { x: 1.2, y: 0.2, z: 0.1 });
      return { solidA, solidB };
    },
  },
  {
    id: "cone-glancing-box-face",
    title: "20. Pointed Cone Glancing Box Face",
    badge: "Slant Tangent",
    category: "cones",
    description:
      "A pointed cone tilted so its slant generator grazes the vertical x = 0 face of a block at a 0.8° glancing angle.",
    defaultOp: "union",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const segs = Math.max(12, Math.round(16 * s));
      const solidA = createBox3D([-6, -5, -4], [0, 5, 4]);
      const slantDeg = (Math.atan2(3.5, 8.0) * 180) / Math.PI;
      const rawB = createCone3D(3.5, 0, -4, 4, segs);
      const solidB = applyLiveTransform(
        rawB,
        { ...offset, dx: offset.dx + 1.65 },
        { x: 0, y: 0, z: 0 },
        [0, -(slantDeg - 0.8), 0],
      );
      return { solidA, solidB };
    },
  },
  {
    id: "cone-cylinder-coaxial-glancing",
    title: "21. Coaxial Frustum & Cylinder (2.2° Glancing)",
    badge: "Coaxial Glancing",
    category: "cones",
    description:
      "A conical frustum and cylinder sharing z = ±4 cap planes with a 2.2° glancing axial tilt.",
    defaultOp: "difference",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const segs = Math.max(12, Math.round(16 * s));
      const solidA = createCylinder3D(4.0, -4, 4, segs);
      const rawB = createCone3D(5.0, 1.5, -4, 4, segs);
      const solidB = applyLiveTransform(
        rawB,
        { ...offset, dx: offset.dx + 0.15 },
        { x: 0, y: 0, z: 0 },
        [2.2, 0, 11.25],
      );
      return { solidA, solidB };
    },
  },
  {
    id: "cones-slant-generator-glancing",
    title: "22. Counter-Tilted Cones Glancing at 1.5°",
    badge: "1.5° Cones",
    category: "cones",
    description:
      "Two pointed 16-gon cones arranged tip-to-base and tilted by 1.5° so their conical walls graze along a narrow generator strip.",
    defaultOp: "union",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const segs = Math.max(12, Math.round(16 * s));
      const solidA = createCone3D(4.0, 0, -4, 4, segs);
      const rawB = createCone3D(4.0, 0, -4, 4, segs);
      const solidB = applyLiveTransform(
        rawB,
        { ...offset, dx: offset.dx + 3.75 },
        { x: 0, y: 0, z: 0 },
        [181.5, 1.0, 11.25],
      );
      return { solidA, solidB };
    },
  },

  // =========================================================================
  // TOROIDS
  // =========================================================================
  {
    id: "torus-kissing-box",
    title: "23. Torus Resting Tangent on Box Face (1.5° Tilt)",
    badge: "Torus + Box",
    category: "toroids",
    description:
      "A genus-1 torus (R = 4.2, r = 1.6) resting on the z = 0 top face of a base block with a 1.5° glancing tilt, creating a crescent contact footprint.",
    defaultOp: "union",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const maj = Math.max(10, Math.round(14 * s));
      const min = Math.max(6, Math.round(8 * s));
      const solidA = createBox3D([-6, -6, -4], [6, 6, 0]);
      const rawB = createTorus3D(4.2, 1.6, [0, 0, 0], maj, min);
      const solidB = applyLiveTransform(
        rawB,
        { ...offset, dz: offset.dz + 1.45 },
        { x: 0, y: 0, z: 0 },
        [1.5, -1.0, 5.0],
      );
      return { solidA, solidB };
    },
  },
  {
    id: "cylinder-threading-torus-bore",
    title: "24. Cylinder Threading & Grazing Torus Bore",
    badge: "Saddle Bore",
    category: "toroids",
    description:
      "A vertical cylinder (R = 2.65) threading the central hole of a torus (inner radius R - r = 2.5) at a 2.0° glancing angle, grazing the hyperbolic inner saddle ring.",
    defaultOp: "difference",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const maj = Math.max(10, Math.round(14 * s));
      const min = Math.max(6, Math.round(8 * s));
      const segs = Math.max(12, Math.round(16 * s));
      const solidA = createTorus3D(4.2, 1.7, [0, 0, 0], maj, min);
      const rawB = createCylinder3D(2.65, -4, 4, segs);
      const solidB = applyLiveTransform(
        rawB,
        { ...offset, dx: offset.dx + 0.12 },
        { x: 0, y: 0, z: 0 },
        [2.0, -1.2, 8.0],
      );
      return { solidA, solidB };
    },
  },
  {
    id: "hopf-link-toroids",
    title: "25. Hopf-Link Interlocking Toroids (2° Graze)",
    badge: "Genus-2 Link",
    category: "toroids",
    description:
      "Two perpendicular interlocking toroids arranged in a Hopf link and shifted so their minor tubes tangentially graze each other.",
    defaultOp: "union",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const maj = Math.max(10, Math.round(14 * s));
      const min = Math.max(6, Math.round(8 * s));
      const solidA = createTorus3D(4.2, 1.4, [0, 0, 0], maj, min);
      const rawB = createTorus3D(4.2, 1.4, [0, 0, 0], maj, min);
      const solidB = applyLiveTransform(
        rawB,
        { ...offset, dx: offset.dx + 4.1, dy: offset.dy + 0.15 },
        { x: 0, y: 0, z: 0 },
        [88.0, 2.0, 5.0],
      );
      return { solidA, solidB };
    },
  },
  {
    id: "sphere-in-torus-saddle",
    title: "26. Sphere Nested in Torus Inner Saddle Ring",
    badge: "Saddle Kissing",
    category: "toroids",
    description:
      "A UV-sphere (R = 2.75) seated inside the central hole of a torus (inner radius = 2.6), grazing the entire inner equator in a continuous ring of contact.",
    defaultOp: "difference",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const maj = Math.max(10, Math.round(14 * s));
      const min = Math.max(6, Math.round(8 * s));
      const lat = Math.max(6, Math.round(8 * s));
      const lon = Math.max(8, Math.round(12 * s));
      const solidA = createTorus3D(4.2, 1.6, [0, 0, 0], maj, min);
      const rawB = createSphere3D(2.75, [0.1, -0.05, 0.15], lat, lon);
      const solidB = applyLiveTransform(rawB, offset, { x: 0.1, y: -0.05, z: 0.15 });
      return { solidA, solidB };
    },
  },
  {
    id: "coaxial-stacked-toroids",
    title: "27. Coaxial Stacked Toroids Kissing Along Ring",
    badge: "Stacked Toroids",
    category: "toroids",
    description:
      "Two coaxial toroids stacked vertically and tilted by 0.9° so their circular crests graze along an annular contact zone.",
    defaultOp: "union",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const maj = Math.max(10, Math.round(14 * s));
      const min = Math.max(6, Math.round(8 * s));
      const solidA = createTorus3D(4.0, 1.4, [0, 0, -1.35], maj, min);
      const rawB = createTorus3D(4.0, 1.4, [0, 0, 0], maj, min);
      const solidB = applyLiveTransform(
        rawB,
        { ...offset, dx: offset.dx + 0.08, dz: offset.dz + 1.35 },
        { x: 0, y: 0, z: 0 },
        [0.9, -0.6, 12.8],
      );
      return { solidA, solidB };
    },
  },
  {
    id: "star-prism-grazing-torus",
    title: "28. Non-Convex Star Prism Grazing Torus (2.1°)",
    badge: "Star + Torus",
    category: "toroids",
    description:
      "A 5-pointed non-convex extruded star prism slicing through a torus at a 2.1° glancing angle.",
    defaultOp: "difference",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const maj = Math.max(10, Math.round(14 * s));
      const min = Math.max(6, Math.round(8 * s));
      const starPts: [number, number][] = [];
      for (let i = 0; i < 10; i++) {
        const r = i % 2 === 0 ? 5.2 : 2.3;
        const a = (i * Math.PI) / 5;
        starPts.push([r * Math.cos(a), r * Math.sin(a)]);
      }
      const solidA = extrudePolygon3D(starPts, [], -2.5, 2.5);
      const rawB = createTorus3D(3.8, 1.3, [0, 0, 0], maj, min);
      const solidB = applyLiveTransform(
        rawB,
        { ...offset, dx: offset.dx + 1.2, dz: offset.dz + 0.3 },
        { x: 0, y: 0, z: 0 },
        [2.1, -1.4, 9.0],
      );
      return { solidA, solidB };
    },
  },

  // =========================================================================
  // OPEN GEOMETRY & HEALING
  // =========================================================================
  {
    id: "closed-box-dangling-fin",
    title: "29. Closed Box with Dangling Fin (Flap Pruning)",
    badge: "Flap Pruning",
    category: "open",
    description:
      "Solid A has a closed 8×8×8 cube plus a dangling non-manifold external fin protruding to x = 8. Stage 3b automatically prunes the zero-volume flap (d_f == d_b) while uniting the closed bodies.",
    defaultOp: "union",
    buildSolids: (offset) => {
      const box = createBox3D([-4, -4, -4], [4, 4, 4]);
      const finOuter: Point3D[] = [
        { x: 4, y: 0, z: -3 },
        { x: 8, y: 0, z: -3 },
        { x: 8, y: 0, z: 3 },
        { x: 4, y: 0, z: 3 },
      ];
      const solidA: Solid3D = {
        faces: [
          ...box.faces,
          { outer: finOuter, holes: [], normal: computePolygonNormal(finOuter) },
        ],
      };
      const rawB = createBox3D([1, -3, -3], [6, 3, 3]);
      const solidB = applyLiveTransform(
        rawB,
        offset,
        { x: 3.5, y: 0, z: 0 },
        [1.5, -1.0, 2.0],
      );
      return { solidA, solidB };
    },
  },
  {
    id: "open-uncapped-box-cutter",
    title: "30. Open Uncapped Box (5 Faces) & Diagonal Cutter",
    badge: "Open Shell",
    category: "open",
    description:
      "Solid A is an open 5-face container (missing its +Z roof) intersected by a tilted closed box. Stage 3b/3c extracts the closed manifold intersection pocket.",
    defaultOp: "union",
    buildSolids: (offset) => {
      const fullBox = createBox3D([-4, -4, -4], [4, 4, 4]);
      const solidA: Solid3D = {
        faces: fullBox.faces.filter((f) => f.normal.z <= 0.9),
      };
      const rawB = createBox3D([-3, -3, -1], [5, 5, 5]);
      const solidB = applyLiveTransform(
        rawB,
        offset,
        { x: 1, y: 1, z: 2 },
        [4.0, -3.0, 12.0],
      );
      return { solidA, solidB };
    },
  },
  {
    id: "open-hemisphere-dish-cylinder",
    title: "31. Open Hemisphere Dish Sliced by Cylinder",
    badge: "Open Dish",
    category: "open",
    description:
      "An open lower hemisphere dish (uncapped at the z = 0 equator) intersected by a tilted cylinder.",
    defaultOp: "intersect",
    buildSolids: (offset) => {
      const s = offset.detailScale ?? 1;
      const lat = Math.max(6, Math.round(8 * s));
      const lon = Math.max(8, Math.round(12 * s));
      const segs = Math.max(12, Math.round(16 * s));
      const sphere = createSphere3D(4.2, [0, 0, 0], lat, lon);
      const solidA: Solid3D = {
        faces: sphere.faces.filter((f) => {
          const avgZ = f.outer.reduce((acc, p) => acc + p.z, 0) / f.outer.length;
          return avgZ < 0;
        }),
      };
      const rawB = createCylinder3D(2.8, -5, 2, segs);
      const solidB = applyLiveTransform(
        rawB,
        { ...offset, dx: offset.dx + 1.1, dy: offset.dy - 0.4 },
        { x: 0, y: 0, z: 0 },
        [3.0, -1.5, 0],
      );
      return { solidA, solidB };
    },
  },
];

export interface Camera3D {
  yaw: number;
  pitch: number;
  distance: number;
  panX: number;
  panY: number;
}

export interface RenderOptions3D {
  showInputWireframe: boolean;
  showShadedFaces: boolean;
  showBRepEdges: boolean;
  showNormals: boolean;
  colorByFace: boolean;
}

const FACE_PALETTE = [
  "#38bdf8",
  "#34d399",
  "#a78bfa",
  "#fbbf24",
  "#f472b6",
  "#60a5fa",
  "#4ade80",
  "#fb923c",
  "#2dd4bf",
  "#c084fc",
];

export function projectPoint3D(
  p: Point3D,
  cam: Camera3D,
  width: number,
  height: number,
): { sx: number; sy: number; cz: number } {
  const cosY = Math.cos(cam.yaw);
  const sinY = Math.sin(cam.yaw);
  const cosP = Math.cos(cam.pitch);
  const sinP = Math.sin(cam.pitch);

  // Rotate around Z (yaw), then X (pitch)
  const x1 = p.x * cosY - p.y * sinY;
  const y1 = p.x * sinY + p.y * cosY;
  const z1 = p.z;

  const x2 = x1;
  const y2 = y1 * sinP + z1 * cosP;
  const cz = y1 * cosP - z1 * sinP + cam.distance;

  const fov = Math.min(width, height) * 1.15;
  const scale = fov / Math.max(2.0, cz);

  return {
    sx: width * 0.5 + cam.panX + x2 * scale,
    sy: height * 0.5 + cam.panY - y2 * scale,
    cz,
  };
}

function drawWireSolid(
  ctx: CanvasRenderingContext2D,
  solid: Solid3D,
  cam: Camera3D,
  w: number,
  h: number,
  strokeStyle: string,
  dash: number[],
): void {
  ctx.save();
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = 1.15;
  ctx.setLineDash(dash);

  const camPos: Point3D = {
    x: -cam.distance * Math.cos(cam.pitch) * Math.sin(cam.yaw),
    y: -cam.distance * Math.cos(cam.pitch) * Math.cos(cam.yaw),
    z: cam.distance * Math.sin(cam.pitch),
  };

  for (const face of solid.faces) {
    if (face.outer.length === 0) continue;
    const p0 = face.outer[0]!;
    // Cull back-facing input wireframes so ghost lines behind the solid don't clutter the front view
    const toCamX = camPos.x - p0.x;
    const toCamY = camPos.y - p0.y;
    const toCamZ = camPos.z - p0.z;
    const facing =
      face.normal.x * toCamX + face.normal.y * toCamY + face.normal.z * toCamZ;
    if (facing <= 0) continue;

    const rings = [face.outer, ...face.holes];
    for (const ring of rings) {
      if (ring.length < 2) continue;
      ctx.beginPath();
      for (let i = 0; i < ring.length; i++) {
        const pr = projectPoint3D(ring[i], cam, w, h);
        if (i === 0) ctx.moveTo(pr.sx, pr.sy);
        else ctx.lineTo(pr.sx, pr.sy);
      }
      ctx.closePath();
      ctx.stroke();
    }
  }
  ctx.restore();
}

function hexToRgb01(hex: string): [number, number, number] {
  return [
    parseInt(hex.slice(1, 3), 16) / 255,
    parseInt(hex.slice(3, 5), 16) / 255,
    parseInt(hex.slice(5, 7), 16) / 255,
  ];
}

interface WebGLBRepContext {
  canvas: HTMLCanvasElement;
  gl: WebGLRenderingContext;
  faceProg: WebGLProgram;
  lineProg: WebGLProgram;
  faceBuf: WebGLBuffer;
  lineBuf: WebGLBuffer;
}

let glCtxCache: WebGLBRepContext | null | undefined;

function compileShader(
  gl: WebGLRenderingContext,
  type: number,
  src: string,
): WebGLShader | null {
  const sh = gl.createShader(type);
  if (!sh) return null;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    gl.deleteShader(sh);
    return null;
  }
  return sh;
}

function createProgram(
  gl: WebGLRenderingContext,
  vsSrc: string,
  fsSrc: string,
): WebGLProgram | null {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  if (!vs || !fs) return null;
  const prog = gl.createProgram();
  if (!prog) return null;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    gl.deleteProgram(prog);
    return null;
  }
  return prog;
}

function getWebGLBRepContext(): WebGLBRepContext | null {
  if (glCtxCache !== undefined) return glCtxCache;
  if (typeof document === "undefined") {
    glCtxCache = null;
    return null;
  }
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl", {
    antialias: true,
    depth: true,
    alpha: true,
    premultipliedAlpha: false,
  });
  if (!gl) {
    glCtxCache = null;
    return null;
  }

  const commonTransformGLSL = `
    uniform vec2 u_resolution;
    uniform vec2 u_pan;
    uniform float u_fov;
    uniform float u_distance;
    uniform vec2 u_yawCS;   // (cos(yaw), sin(yaw))
    uniform vec2 u_pitchCS; // (cos(pitch), sin(pitch))

    vec4 projectWorld3D(vec3 p, out vec3 vEyePos) {
      float x1 = p.x * u_yawCS.x - p.y * u_yawCS.y;
      float y1 = p.x * u_yawCS.y + p.y * u_yawCS.x;
      float z1 = p.z;

      float cx = x1;
      float cy = y1 * u_pitchCS.y + z1 * u_pitchCS.x;
      float cz = max(1.5, y1 * u_pitchCS.x - z1 * u_pitchCS.y + u_distance);
      vEyePos = vec3(cx, cy, cz);

      float clipX = (2.0 * u_fov / u_resolution.x) * cx + (2.0 * u_pan.x / u_resolution.x) * cz;
      float clipY = (2.0 * u_fov / u_resolution.y) * cy - (2.0 * u_pan.y / u_resolution.y) * cz;

      float zNear = 1.5;
      float zFar = 220.0;
      float clipZ = ((zFar + zNear) / (zFar - zNear)) * cz - (2.0 * zFar * zNear / (zFar - zNear));
      return vec4(clipX, clipY, clipZ, cz);
    }
  `;

  const faceVS = `
    attribute vec3 a_pos;
    attribute vec3 a_normal;
    attribute vec3 a_color;
    varying vec3 v_normal;
    varying vec3 v_color;
    varying vec3 v_eyePos;
    ${commonTransformGLSL}
    void main() {
      v_normal = a_normal;
      v_color = a_color;
      gl_Position = projectWorld3D(a_pos, v_eyePos);
    }
  `;

  const faceFS = `
    precision highp float;
    varying vec3 v_normal;
    varying vec3 v_color;
    varying vec3 v_eyePos;
    uniform vec2 u_yawCS;
    uniform vec2 u_pitchCS;

    void main() {
      vec3 n = normalize(v_normal);
      // Key directional light + warm fill + hemisphere sky/ground ambient
      vec3 keyDir = normalize(vec3(0.42, -0.58, 0.69));
      vec3 fillDir = normalize(vec3(-0.62, 0.35, -0.45));

      float key = max(0.0, dot(n, keyDir));
      float fill = max(0.0, dot(n, fillDir));
      float hemi = 0.5 + 0.5 * n.z;

      // Rotate normal to camera space for subtle rim highlight
      float nx1 = n.x * u_yawCS.x - n.y * u_yawCS.y;
      float ny1 = n.x * u_yawCS.y + n.y * u_yawCS.x;
      float nzCam = -(ny1 * u_pitchCS.x - n.z * u_pitchCS.y);
      float rim = pow(clamp(1.0 - abs(nzCam), 0.0, 1.0), 2.4) * 0.14;

      float shade = 0.42 + 0.44 * key + 0.14 * fill + 0.10 * hemi;
      vec3 col = v_color * shade + vec3(rim * 0.7, rim * 0.85, rim);
      gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
    }
  `;

  const lineVS = `
    attribute vec3 a_pos;
    uniform float u_pointSize;
    ${commonTransformGLSL}
    void main() {
      vec3 eyePos;
      gl_Position = projectWorld3D(a_pos, eyePos);
      gl_PointSize = u_pointSize;
    }
  `;

  const lineFS = `
    precision mediump float;
    uniform vec4 u_color;
    uniform float u_isPoint;
    void main() {
      if (u_isPoint > 0.5) {
        vec2 c = gl_PointCoord * 2.0 - 1.0;
        if (dot(c, c) > 1.0) discard;
      }
      gl_FragColor = u_color;
    }
  `;

  const faceProg = createProgram(gl, faceVS, faceFS);
  const lineProg = createProgram(gl, lineVS, lineFS);
  const faceBuf = gl.createBuffer();
  const lineBuf = gl.createBuffer();

  if (!faceProg || !lineProg || !faceBuf || !lineBuf) {
    glCtxCache = null;
    return null;
  }

  glCtxCache = { canvas, gl, faceProg, lineProg, faceBuf, lineBuf };
  return glCtxCache;
}

function setTransformUniforms(
  gl: WebGLRenderingContext,
  prog: WebGLProgram,
  cam: Camera3D,
  width: number,
  height: number,
): void {
  const fov = Math.min(width, height) * 1.15;
  gl.uniform2f(gl.getUniformLocation(prog, "u_resolution"), width, height);
  gl.uniform2f(gl.getUniformLocation(prog, "u_pan"), cam.panX, cam.panY);
  gl.uniform1f(gl.getUniformLocation(prog, "u_fov"), fov);
  gl.uniform1f(gl.getUniformLocation(prog, "u_distance"), cam.distance);
  gl.uniform2f(
    gl.getUniformLocation(prog, "u_yawCS"),
    Math.cos(cam.yaw),
    Math.sin(cam.yaw),
  );
  gl.uniform2f(
    gl.getUniformLocation(prog, "u_pitchCS"),
    Math.cos(cam.pitch),
    Math.sin(cam.pitch),
  );
}

function renderWebGL3DPass(
  glCtx: WebGLBRepContext,
  width: number,
  height: number,
  dpr: number,
  result: BRepResult3D,
  cam: Camera3D,
  opts: RenderOptions3D,
): void {
  const { canvas, gl, faceProg, lineProg, faceBuf, lineBuf } = glCtx;
  const pw = Math.max(1, Math.round(width * dpr));
  const ph = Math.max(1, Math.round(height * dpr));
  if (canvas.width !== pw || canvas.height !== ph) {
    canvas.width = pw;
    canvas.height = ph;
  }

  gl.viewport(0, 0, pw, ph);
  gl.clearColor(0, 0, 0, 0);
  gl.clearDepth(1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  // 1. Hardware Z-buffered shaded B-Rep surface triangles
  if (opts.showShadedFaces && result.triangles.length > 0) {
    const triCount = result.triangles.length;
    const data = new Float32Array(triCount * 3 * 9);
    let c = 0;

    for (let i = 0; i < triCount; i++) {
      const t = result.triangles[i]!;
      let r: number;
      let g: number;
      let b: number;
      if (opts.colorByFace) {
        const hex = FACE_PALETTE[t.faceIndex % FACE_PALETTE.length]!;
        [r, g, b] = hexToRgb01(hex);
      } else {
        r = (110 + 75 * t.normal.x) / 255;
        g = (155 + 70 * t.normal.y) / 255;
        b = (205 + 45 * t.normal.z) / 255;
      }

      for (const pt of [t.a, t.b, t.c]) {
        data[c++] = pt.x;
        data[c++] = pt.y;
        data[c++] = pt.z;
        data[c++] = t.normal.x;
        data[c++] = t.normal.y;
        data[c++] = t.normal.z;
        data[c++] = r;
        data[c++] = g;
        data[c++] = b;
      }
    }

    gl.useProgram(faceProg);
    setTransformUniforms(gl, faceProg, cam, width, height);

    gl.bindBuffer(gl.ARRAY_BUFFER, faceBuf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);

    const stride = 9 * 4;
    const posLoc = gl.getAttribLocation(faceProg, "a_pos");
    const normLoc = gl.getAttribLocation(faceProg, "a_normal");
    const colLoc = gl.getAttribLocation(faceProg, "a_color");

    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(normLoc);
    gl.vertexAttribPointer(normLoc, 3, gl.FLOAT, false, stride, 3 * 4);
    gl.enableVertexAttribArray(colLoc);
    gl.vertexAttribPointer(colLoc, 3, gl.FLOAT, false, stride, 6 * 4);

    gl.depthMask(true);
    gl.enable(gl.POLYGON_OFFSET_FILL);
    gl.polygonOffset(1.0, 1.5);
    gl.drawArrays(gl.TRIANGLES, 0, triCount * 3);
    gl.disable(gl.POLYGON_OFFSET_FILL);

    gl.disableVertexAttribArray(posLoc);
    gl.disableVertexAttribArray(normLoc);
    gl.disableVertexAttribArray(colLoc);
  }

  // 2. Depth-tested B-Rep outer loops, interior hole loops, and canonical 0-strata vertices
  if (opts.showBRepEdges && result.solid.faces.length > 0) {
    const camPos: Point3D = {
      x: -cam.distance * Math.cos(cam.pitch) * Math.sin(cam.yaw),
      y: -cam.distance * Math.cos(cam.pitch) * Math.cos(cam.yaw),
      z: cam.distance * Math.sin(cam.pitch),
    };

    const outerSegs: number[] = [];
    const holeSegs: number[] = [];
    const outerPts: number[] = [];
    const holePts: number[] = [];

    for (const face of result.solid.faces) {
      if (face.outer.length === 0) continue;
      // Only emit boundary loops for front-facing faces so edges on back-facing walls
      // cannot poke through shared corner vertices or silhouettes!
      const p0 = face.outer[0]!;
      const facing =
        face.normal.x * (camPos.x - p0.x) +
        face.normal.y * (camPos.y - p0.y) +
        face.normal.z * (camPos.z - p0.z);
      if (facing <= 0) continue;

      const n = face.outer.length;
      for (let i = 0; i < n; i++) {
        const a = face.outer[i]!;
        const b = face.outer[(i + 1) % n]!;
        outerSegs.push(a.x, a.y, a.z, b.x, b.y, b.z);
        outerPts.push(a.x, a.y, a.z);
      }
      for (const hole of face.holes) {
        const hn = hole.length;
        for (let i = 0; i < hn; i++) {
          const a = hole[i]!;
          const b = hole[(i + 1) % hn]!;
          holeSegs.push(a.x, a.y, a.z, b.x, b.y, b.z);
          holePts.push(a.x, a.y, a.z);
        }
      }
    }

    gl.useProgram(lineProg);
    setTransformUniforms(gl, lineProg, cam, width, height);
    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);

    const posLoc = gl.getAttribLocation(lineProg, "a_pos");
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

    const colorLoc = gl.getUniformLocation(lineProg, "u_color");
    const ptSizeLoc = gl.getUniformLocation(lineProg, "u_pointSize");
    const isPtLoc = gl.getUniformLocation(lineProg, "u_isPoint");

    const drawBuffer = (
      arr: number[],
      mode: number,
      rgba: [number, number, number, number],
      depthFunc: number,
      isPoint = false,
      ptSize = 1.0,
    ) => {
      if (arr.length === 0) return;
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.DYNAMIC_DRAW);
      gl.depthFunc(depthFunc);
      gl.uniform4f(colorLoc, rgba[0], rgba[1], rgba[2], rgba[3]);
      gl.uniform1f(isPtLoc, isPoint ? 1.0 : 0.0);
      gl.uniform1f(ptSizeLoc, ptSize * dpr);
      gl.drawArrays(mode, 0, arr.length / 3);
    };

    gl.depthMask(false);

    // Crisp visible front B-Rep edges & hole loops (depth <= surface)
    drawBuffer(outerSegs, gl.LINES, [0.90, 0.96, 1.0, 1.0], gl.LEQUAL);
    drawBuffer(holeSegs, gl.LINES, [1.0, 0.36, 0.50, 1.0], gl.LEQUAL);

    // Depth-tested canonical 0-strata vertices
    drawBuffer(outerPts, gl.POINTS, [0.22, 0.74, 0.97, 1.0], gl.LEQUAL, true, 5.5);
    drawBuffer(holePts, gl.POINTS, [0.98, 0.44, 0.52, 1.0], gl.LEQUAL, true, 6.2);

    gl.depthMask(true);
    gl.depthFunc(gl.LEQUAL);
    gl.disableVertexAttribArray(posLoc);
  }
}

export function renderBRep3DScene(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  solidA: Solid3D,
  solidB: Solid3D,
  result: BRepResult3D,
  cam: Camera3D,
  opts: RenderOptions3D,
): void {
  ctx.clearRect(0, 0, width, height);

  // Subtle radial background glow
  const grad = ctx.createRadialGradient(
    width * 0.5,
    height * 0.5,
    20,
    width * 0.5,
    height * 0.5,
    Math.max(width, height) * 0.65,
  );
  grad.addColorStop(0, "#131b2e");
  grad.addColorStop(1, "#090d16");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);

  // Draw reference XY ground grid at z = -6
  ctx.save();
  ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
  ctx.lineWidth = 1;
  const gridZ = -6;
  for (let g = -10; g <= 10; g += 2) {
    const p0 = projectPoint3D({ x: g, y: -10, z: gridZ }, cam, width, height);
    const p1 = projectPoint3D({ x: g, y: 10, z: gridZ }, cam, width, height);
    ctx.beginPath();
    ctx.moveTo(p0.sx, p0.sy);
    ctx.lineTo(p1.sx, p1.sy);
    ctx.stroke();

    const q0 = projectPoint3D({ x: -10, y: g, z: gridZ }, cam, width, height);
    const q1 = projectPoint3D({ x: 10, y: g, z: gridZ }, cam, width, height);
    ctx.beginPath();
    ctx.moveTo(q0.sx, q0.sy);
    ctx.lineTo(q1.sx, q1.sy);
    ctx.stroke();
  }
  ctx.restore();

  // Ghosted input wireframes (Solid A & Solid B) — drawn behind the WebGL pass so they are naturally occluded by the solid
  if (opts.showInputWireframe) {
    drawWireSolid(ctx, solidA, cam, width, height, "rgba(56, 189, 248, 0.28)", [4, 4]);
    drawWireSolid(ctx, solidB, cam, width, height, "rgba(251, 146, 60, 0.42)", [4, 4]);
  }

  // Hardware Z-buffered WebGL rendering of shaded B-Rep faces and depth-tested B-Rep topology loops
  const glCtx = getWebGLBRepContext();
  if (glCtx) {
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    renderWebGL3DPass(glCtx, width, height, dpr, result, cam, opts);
    ctx.drawImage(glCtx.canvas, 0, 0, width, height);
  }

  // Draw outward face normal vectors at each face's outer centroid
  if (opts.showNormals) {
    ctx.save();
    ctx.strokeStyle = "#facc15";
    ctx.fillStyle = "#facc15";
    ctx.lineWidth = 1.5;
    for (const face of result.solid.faces) {
      if (face.outer.length === 0) continue;
      let cx = 0;
      let cy = 0;
      let cz = 0;
      for (const p of face.outer) {
        cx += p.x;
        cy += p.y;
        cz += p.z;
      }
      cx /= face.outer.length;
      cy /= face.outer.length;
      cz /= face.outer.length;

      const tip = {
        x: cx + face.normal.x * 1.4,
        y: cy + face.normal.y * 1.4,
        z: cz + face.normal.z * 1.4,
      };
      const p0 = projectPoint3D({ x: cx, y: cy, z: cz }, cam, width, height);
      const p1 = projectPoint3D(tip, cam, width, height);

      ctx.beginPath();
      ctx.moveTo(p0.sx, p0.sy);
      ctx.lineTo(p1.sx, p1.sy);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(p1.sx, p1.sy, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
}

export interface InvariantAudit3D {
  volA: number;
  volB: number;
  volUnion: number;
  volIntersect: number;
  volDiffAB: number;
  inclusionExclusionError: number;
  inclusionExclusionPass: boolean;
  manifoldPass: boolean;
  holeLoopCount: number;
  eulerCharacteristic: number;
}

function computeSolidVolume3D(solid: Solid3D): number {
  let vol6 = 0;
  for (const f of solid.faces) {
    for (const ring of [f.outer, ...f.holes]) {
      if (ring.length < 3) continue;
      const p0 = ring[0]!;
      for (let i = 1; i + 1 < ring.length; i++) {
        const a = ring[i]!;
        const b = ring[i + 1]!;
        const cx = a.y * b.z - a.z * b.y;
        const cy = a.z * b.x - a.x * b.z;
        const cz = a.x * b.y - a.y * b.x;
        vol6 += p0.x * cx + p0.y * cy + p0.z * cz;
      }
    }
  }
  return vol6 / 6.0;
}

export function audit3DBooleanInvariants(
  solidA: Solid3D,
  solidB: Solid3D,
  activeResult: BRepResult3D,
  activeOp?: BooleanOp,
  options?: KernelOptions,
): InvariantAudit3D {
  const u = activeOp === "union" ? activeResult : union3D(solidA, solidB, options);
  const i = activeOp === "intersect" ? activeResult : intersect3D(solidA, solidB, options);
  const d = activeOp === "difference" ? activeResult : difference3D(solidA, solidB, options);
  const volA = computeSolidVolume3D(solidA);
  const volB = computeSolidVolume3D(solidB);
  const ieErr = Math.abs(u.totalVolume + i.totalVolume - (volA + volB));
  const ieTol = options?.snapTolerance ? Math.max(1e-2, options.snapTolerance * 10) : 1e-2;

  let holeLoopCount = 0;
  for (const f of activeResult.solid.faces) {
    holeLoopCount += f.holes.length;
  }

  // Generalized Euler-Poincaré formula with face holes H: χ = V - E + (F - H)
  const eulerCharacteristic =
    activeResult.vertexCount -
    activeResult.edgeCount +
    (activeResult.solid.faces.length - holeLoopCount);

  return {
    volA,
    volB,
    volUnion: u.totalVolume,
    volIntersect: i.totalVolume,
    volDiffAB: d.totalVolume,
    inclusionExclusionError: ieErr,
    inclusionExclusionPass: ieErr < ieTol,
    manifoldPass:
      activeResult.unmatchedHalfEdgeCount === 0 &&
      u.unmatchedHalfEdgeCount === 0 &&
      i.unmatchedHalfEdgeCount === 0 &&
      d.unmatchedHalfEdgeCount === 0,
    holeLoopCount,
    eulerCharacteristic,
  };
}

export { clip3D };
