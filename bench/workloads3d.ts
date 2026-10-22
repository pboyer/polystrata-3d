import {
  computeLoopNormal3D,
  createBox3D,
  extrudePolygon3D,
  type Face3D,
  type Point3D,
  type Solid3D,
  transformSolid3D,
} from "../src/index.js";

export type BenchOp3D = "union" | "intersect" | "difference";

export interface BenchWorkload3D {
  id: string;
  name: string;
  category: "standard" | "quadrics" | "glancing" | "robustness";
  description: string;
  solidA: Solid3D;
  solidB: Solid3D;
  isClosedManifoldInput: boolean;
}

export function rotateAndTranslateSolid3D(
  solid: Solid3D,
  rxDeg: number,
  ryDeg: number,
  rzDeg: number,
  tx: number,
  ty: number,
  tz: number,
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
    const y1 = p.y * cx - p.z * sx;
    const z1 = p.y * sx + p.z * cx;
    const x2 = p.x * cy + z1 * sy;
    const z2 = -p.x * sy + z1 * cy;
    const x3 = x2 * cz - y1 * sz;
    const y3 = x2 * sz + y1 * cz;
    return { x: x3 + tx, y: y3 + ty, z: z2 + tz };
  });
}

export function createCylinder3D(
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

export function createConeFrustum3D(
  rBottom: number,
  rTop: number,
  zMin: number,
  zMax: number,
  segments = 16,
): Solid3D {
  const faces: Face3D[] = [];
  const botRing: Point3D[] = [];
  const topRing: Point3D[] = [];

  for (let i = 0; i < segments; i++) {
    const a = (2 * Math.PI * i) / segments;
    botRing.push({ x: rBottom * Math.cos(a), y: rBottom * Math.sin(a), z: zMin });
    if (rTop > 1e-9) {
      topRing.push({ x: rTop * Math.cos(a), y: rTop * Math.sin(a), z: zMax });
    }
  }

  faces.push({
    outer: [...botRing].reverse(),
    holes: [],
    normal: { x: 0, y: 0, z: -1 },
  });

  if (rTop > 1e-9) {
    faces.push({
      outer: [...topRing],
      holes: [],
      normal: { x: 0, y: 0, z: 1 },
    });
    for (let i = 0; i < segments; i++) {
      const b0 = botRing[i]!;
      const b1 = botRing[(i + 1) % segments]!;
      const t1 = topRing[(i + 1) % segments]!;
      const t0 = topRing[i]!;
      const outer = [b0, b1, t1, t0];
      faces.push({ outer, holes: [], normal: computeLoopNormal3D(outer) });
    }
  } else {
    const apex: Point3D = { x: 0, y: 0, z: zMax };
    for (let i = 0; i < segments; i++) {
      const b0 = botRing[i]!;
      const b1 = botRing[(i + 1) % segments]!;
      const outer = [b0, b1, apex];
      faces.push({ outer, holes: [], normal: computeLoopNormal3D(outer) });
    }
  }

  return { faces };
}

export function createUVSphere3D(
  radius: number,
  latBands = 12,
  lonBands = 12,
): Solid3D {
  const faces: Face3D[] = [];
  const ringPt = (latIdx: number, lonIdx: number): Point3D => {
    const theta = (Math.PI * latIdx) / latBands;
    const phi = (2 * Math.PI * (lonIdx % lonBands)) / lonBands;
    return {
      x: radius * Math.sin(theta) * Math.cos(phi),
      y: radius * Math.sin(theta) * Math.sin(phi),
      z: radius * Math.cos(theta),
    };
  };

  const northPole: Point3D = { x: 0, y: 0, z: radius };
  const southPole: Point3D = { x: 0, y: 0, z: -radius };

  for (let i = 0; i < latBands; i++) {
    for (let j = 0; j < lonBands; j++) {
      const jNext = (j + 1) % lonBands;
      if (i === 0) {
        const p1 = ringPt(1, j);
        const p2 = ringPt(1, jNext);
        const outer = [northPole, p1, p2];
        faces.push({ outer, holes: [], normal: computeLoopNormal3D(outer) });
      } else if (i === latBands - 1) {
        const p0 = ringPt(latBands - 1, j);
        const p1 = ringPt(latBands - 1, jNext);
        const outer = [p0, southPole, p1];
        faces.push({ outer, holes: [], normal: computeLoopNormal3D(outer) });
      } else {
        const p00 = ringPt(i, j);
        const p10 = ringPt(i + 1, j);
        const p11 = ringPt(i + 1, jNext);
        const p01 = ringPt(i, jNext);
        const outer = [p00, p10, p11, p01];
        faces.push({ outer, holes: [], normal: computeLoopNormal3D(outer) });
      }
    }
  }
  return { faces };
}

export function createTorus3D(
  majorR: number,
  minorR: number,
  majorSegs = 16,
  minorSegs = 10,
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
    for (let j = 0; j < minorSegs; j++) {
      const p00 = pt(i, j);
      const p10 = pt(i + 1, j);
      const p11 = pt(i + 1, j + 1);
      const p01 = pt(i, j + 1);
      const outer = [p00, p10, p11, p01];
      faces.push({ outer, holes: [], normal: computeLoopNormal3D(outer) });
    }
  }
  return { faces };
}

export function createStarGearPrism3D(
  teeth = 8,
  rOuter = 5.0,
  rInner = 2.6,
  zMin = -2.5,
  zMax = 2.5,
): Solid3D {
  const star2d: [number, number][] = [];
  for (let i = 0; i < teeth * 2; i++) {
    const a = (Math.PI * i) / teeth;
    const r = i % 2 === 0 ? rOuter : rInner;
    star2d.push([r * Math.cos(a), r * Math.sin(a)]);
  }
  return extrudePolygon3D(star2d, [], zMin, zMax);
}

export function createComb3D(): Solid3D {
  return extrudePolygon3D(
    [
      [-5, -4],
      [5, -4],
      [5, 4],
      [3, 4],
      [3, -1],
      [1, -1],
      [1, 4],
      [-1, 4],
      [-1, -1],
      [-3, -1],
      [-3, 4],
      [-5, 4],
    ],
    [],
    -2.5,
    2.5,
  );
}

export function createHollowPipe3D(
  rOuter: number,
  rInner: number,
  zMin: number,
  zMax: number,
  segments = 16,
): Solid3D {
  const outer: [number, number][] = [];
  const hole: [number, number][] = [];
  for (let i = 0; i < segments; i++) {
    const a = (2 * Math.PI * i) / segments;
    outer.push([rOuter * Math.cos(a), rOuter * Math.sin(a)]);
    hole.push([rInner * Math.cos(a), rInner * Math.sin(a)]);
  }
  return extrudePolygon3D(outer, [hole], zMin, zMax);
}

export function createBenchmarkWorkloads3D(): BenchWorkload3D[] {
  const workloads: BenchWorkload3D[] = [];

  // 1. Machined Multi-Hole Flange & Through-Pillar
  workloads.push({
    id: "machined-flange-pillar",
    name: "Multi-Hole Flange Plate & Tilted Pillar",
    category: "standard",
    description:
      "Extruded flange plate with 2 rectangular interior through-holes cut by a tilted hexahedral pillar.",
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
      -2.5,
      2.5,
    ),
    solidB: rotateAndTranslateSolid3D(
      createBox3D([-3.2, -3.2, -4.5], [3.2, 3.2, 4.5]),
      14,
      -11,
      18,
      0.4,
      -0.3,
      0,
    ),
    isClosedManifoldInput: true,
  });

  // 2. Coplanar Overlapping Hexahedra
  workloads.push({
    id: "coplanar-hexahedra",
    name: "Coplanar Overlapping Hexahedra (4 Shared Planes)",
    category: "standard",
    description:
      "Two boxes sharing 4 exact coplanar boundary planes (y = ±4, z = ±4); tests coplanar patch unification.",
    solidA: createBox3D([-6, -4, -4], [2, 4, 4]),
    solidB: createBox3D([-2, -4, -4], [6, 4, 4]),
    isClosedManifoldInput: true,
  });

  // 3. 8-Tooth Star Gear & Hexagonal Bore
  workloads.push({
    id: "star-gear-hex-bore",
    name: "8-Tooth Star Gear & Tilted Hexagonal Cutter",
    category: "standard",
    description:
      "Non-convex 16-vertex extruded star gear intersected/differenced with a 12°-tilted hexagonal prism.",
    solidA: createStarGearPrism3D(8, 5.0, 2.6, -2.5, 2.5),
    solidB: rotateAndTranslateSolid3D(
      createCylinder3D(3.6, -4.0, 4.0, 6),
      12,
      -9,
      15,
      0.3,
      -0.2,
      0,
    ),
    isClosedManifoldInput: true,
  });

  // 4. Orthogonal Steinmetz Bicylinder
  const cyl16 = createCylinder3D(4.0, -5.0, 5.0, 16);
  workloads.push({
    id: "steinmetz-bicylinder",
    name: "Orthogonal Equi-Radius Steinmetz Bicylinder",
    category: "quadrics",
    description:
      "Two equal-radius (R=4, 16-seg) cylinders along Z and X axes whose side walls meet tangentially at (0, ±4, 0).",
    solidA: cyl16,
    solidB: rotateAndTranslateSolid3D(cyl16, 0, 90, 0, 0, 0, 0),
    isClosedManifoldInput: true,
  });

  // 5. UV-Spheres Shallow Glancing Lens
  const sph12 = createUVSphere3D(4.0, 12, 12);
  workloads.push({
    id: "glancing-uv-spheres",
    name: "UV-Spheres Shallow Glancing Lens (3.5° Tilt)",
    category: "quadrics",
    description:
      "Two 144-face UV spheres overlapping by 0.5 units with a 3.5° glancing tilt across latitude/longitude bands.",
    solidA: sph12,
    solidB: rotateAndTranslateSolid3D(sph12, 3.5, -2.5, 4.0, 7.5, 0, 0),
    isClosedManifoldInput: true,
  });

  // 6. Torus Saddle Bore Cut by 2° Glancing Cylinder
  const torus16x10 = createTorus3D(4.5, 1.8, 16, 10);
  workloads.push({
    id: "torus-saddle-cylinder",
    name: "Genus-1 Torus Saddle Cut by 2° Glancing Cylinder",
    category: "quadrics",
    description:
      "160-face torus (R=4.5, r=1.8) threaded by a 2°-tilted cylinder (r=2.75) grazing the inner hyperbolic saddle ring.",
    solidA: torus16x10,
    solidB: rotateAndTranslateSolid3D(
      createCylinder3D(2.75, -4.0, 4.0, 16),
      2.0,
      -1.5,
      5.0,
      0.1,
      0,
      0,
    ),
    isClosedManifoldInput: true,
  });

  // 7. Nearly Parallel Cylinders (1.75° Glancing)
  workloads.push({
    id: "parallel-glancing-cylinders",
    name: "Nearly Parallel Cylinders (1.75° Glancing Walls)",
    category: "glancing",
    description:
      "Two R=4 cylinders tilted by 1.75° with 11.25° half-facet roll, creating long sliver intersections along side walls.",
    solidA: cyl16,
    solidB: rotateAndTranslateSolid3D(cyl16, 1.75, 0.8, 11.25, 0.35, 0.1, 0),
    isClosedManifoldInput: true,
  });

  // 8. Pointed Cone Grazing Box Face at 0.8°
  workloads.push({
    id: "cone-glancing-box",
    name: "Pointed Cone Grazing Box Top Face (0.8° Slant)",
    category: "glancing",
    description:
      "16-seg pointed cone tilted by 0.8° beyond its slant angle so a side generator nearly lies in the box top face.",
    solidA: createBox3D([-5, -5, -4], [5, 5, 1]),
    solidB: rotateAndTranslateSolid3D(
      createConeFrustum3D(4.0, 0.0, -4.0, 4.0, 16),
      64.2,
      0,
      11.25,
      0,
      -1.2,
      2.6,
    ),
    isClosedManifoldInput: true,
  });

  // 9. Intermeshing Comb Teeth (1.1° Glancing)
  const comb = createComb3D();
  workloads.push({
    id: "intermeshing-comb-teeth",
    name: "Intermeshing Extruded Comb Teeth (1.1° Glancing)",
    category: "glancing",
    description:
      "Two 3-tooth extruded comb blocks intermeshing with a 1.1° glancing side-wall tilt across multiple teeth.",
    solidA: comb,
    solidB: rotateAndTranslateSolid3D(comb, 180, 1.1, 0.8, 2.0, 1.5, 0.2),
    isClosedManifoldInput: true,
  });

  // 10. Closed Box with Dangling Non-Manifold Fin
  const boxWithFin = createBox3D([-4, -4, -4], [4, 4, 4]);
  const finVerts: Point3D[] = [
    { x: 4, y: 4, z: -4 },
    { x: 7, y: 4, z: -4 },
    { x: 7, y: 4, z: 4 },
    { x: 4, y: 4, z: 4 },
  ];
  const finNormal = computeLoopNormal3D(finVerts);
  boxWithFin.faces.push({ outer: finVerts, holes: [], normal: finNormal });

  workloads.push({
    id: "box-with-dangling-fin",
    name: "Closed Box with Dangling Non-Manifold Fin (Flap Pruning)",
    category: "robustness",
    description:
      "Closed cube with an unclosed non-manifold dangling fin protruding from edge x=4,y=4; tests ∂₂ 2-core flap pruning.",
    solidA: boxWithFin,
    solidB: rotateAndTranslateSolid3D(
      createBox3D([1, -3, -3], [6, 5, 3]),
      4,
      -3,
      5,
      0,
      0,
      0,
    ),
    isClosedManifoldInput: false,
  });

  return workloads;
}

/**
 * 24-scenario adversarial 3D stress suite covering kissing quadrics, shallow glancing angles,
 * multi-hole extrusions, and non-manifold/open inputs.
 */
export function createAdversarial3DStressScenarios(): BenchWorkload3D[] {
  const sph = createUVSphere3D(4.0, 12, 12);
  const cyl = createCylinder3D(4.0, -5.0, 5.0, 16);
  const cone = createConeFrustum3D(4.0, 0.0, -4.0, 4.0, 16);
  const frustum = createConeFrustum3D(4.5, 2.0, -4.0, 4.0, 16);
  const torus = createTorus3D(4.5, 1.8, 16, 10);
  const box = createBox3D([-4, -4, -4], [4, 4, 4]);
  const star = createStarGearPrism3D(8, 5.0, 2.5, -2.5, 2.5);
  const comb = createComb3D();

  const boxWithFin = createBox3D([-4, -4, -4], [4, 4, 4]);
  const finVerts: Point3D[] = [
    { x: 4, y: 4, z: -4 },
    { x: 7, y: 4, z: -4 },
    { x: 7, y: 4, z: 4 },
    { x: 4, y: 4, z: 4 },
  ];
  const finN = computeLoopNormal3D(finVerts);
  boxWithFin.faces.push({ outer: finVerts, holes: [], normal: finN });

  return [
    {
      id: "S1",
      name: "Two UV-Spheres Externally Kissing",
      category: "quadrics",
      description: "Spheres kissing at equatorial point.",
      solidA: sph,
      solidB: rotateAndTranslateSolid3D(sph, 0, 0, 0, 8.0, 0, 0),
      isClosedManifoldInput: true,
    },
    {
      id: "S2",
      name: "Two UV-Spheres Shallow Glancing Lens (3.5°)",
      category: "glancing",
      description: "Spheres overlapping by 0.5 units with 3.5° tilt.",
      solidA: sph,
      solidB: rotateAndTranslateSolid3D(sph, 3.5, -2.5, 4.0, 7.5, 0, 0),
      isClosedManifoldInput: true,
    },
    {
      id: "S3",
      name: "Sphere Skimming Planar Box Top Face (1.2°)",
      category: "glancing",
      description: "Sphere skimming z=2 box face.",
      solidA: createBox3D([-5, -5, -6], [5, 5, 2]),
      solidB: rotateAndTranslateSolid3D(sph, 1.2, -0.8, 15.0, 0, 0, 5.65),
      isClosedManifoldInput: true,
    },
    {
      id: "S4",
      name: "Sphere Wedged in Dihedral Corner of L-Bracket",
      category: "quadrics",
      description: "Sphere grazing both inner walls of an L-bracket.",
      solidA: extrudePolygon3D(
        [
          [-6, -6],
          [6, -6],
          [6, -2],
          [-2, -2],
          [-2, 6],
          [-6, 6],
        ],
        [],
        -4,
        4,
      ),
      solidB: rotateAndTranslateSolid3D(sph, 2.5, -1.8, 12.0, 1.85, 1.85, 0),
      isClosedManifoldInput: true,
    },
    {
      id: "C1",
      name: "Two Cylinders Kissing Along Vertical Generator",
      category: "quadrics",
      description: "Shared vertical generator x=4.",
      solidA: cyl,
      solidB: rotateAndTranslateSolid3D(cyl, 0, 0, 0, 8.0, 0, 0),
      isClosedManifoldInput: true,
    },
    {
      id: "C2",
      name: "Nearly Parallel Cylinders (1.75° Glancing)",
      category: "glancing",
      description: "1.75° glancing cylindrical walls.",
      solidA: cyl,
      solidB: rotateAndTranslateSolid3D(cyl, 1.75, 0.8, 11.25, 0.35, 0.1, 0),
      isClosedManifoldInput: true,
    },
    {
      id: "C3",
      name: "Orthogonal Equi-Radius Steinmetz Bicylinder",
      category: "quadrics",
      description: "Tangential 90° equal-radius cylinders.",
      solidA: cyl,
      solidB: rotateAndTranslateSolid3D(cyl, 0, 90, 0, 0, 0, 0),
      isClosedManifoldInput: true,
    },
    {
      id: "C4",
      name: "Orthogonal Cylinders Grazing Tangentially (1.5°)",
      category: "glancing",
      description: "Orthogonal cylinders barely grazing at 1.5°.",
      solidA: cyl,
      solidB: rotateAndTranslateSolid3D(cyl, 1.5, 90, 0, 0, 7.75, 0),
      isClosedManifoldInput: true,
    },
    {
      id: "C5",
      name: "Hollow Pipe Pierced by 2° Glancing Transverse Cylinder",
      category: "glancing",
      description: "Hollow pipe inner wall grazed at 2°.",
      solidA: createHollowPipe3D(5.0, 3.0, -5.0, 5.0, 16),
      solidB: rotateAndTranslateSolid3D(
        createCylinder3D(3.1, -6.0, 6.0, 16),
        2.0,
        88.5,
        0,
        0,
        0.1,
        0,
      ),
      isClosedManifoldInput: true,
    },
    {
      id: "K1",
      name: "Two Pointed Cones Kissing Tip-to-Tip at Origin (3°)",
      category: "quadrics",
      description: "Singular apex vertices meeting at (0,0,0).",
      solidA: createConeFrustum3D(4.0, 0.0, -5.0, 0.0, 16),
      solidB: rotateAndTranslateSolid3D(
        createConeFrustum3D(0.001, 4.0, 0.0, 5.0, 16),
        3.0,
        -2.0,
        11.25,
        0,
        0,
        -0.15,
      ),
      isClosedManifoldInput: true,
    },
    {
      id: "K2",
      name: "Pointed Cone Grazing Box Top Face (0.8°)",
      category: "glancing",
      description: "Slant generator nearly flush with z=1.",
      solidA: createBox3D([-5, -5, -4], [5, 5, 1]),
      solidB: rotateAndTranslateSolid3D(cone, 64.2, 0, 11.25, 0, -1.2, 2.6),
      isClosedManifoldInput: true,
    },
    {
      id: "K3",
      name: "Coaxial Frustum & Cylinder Glancing at 2.2°",
      category: "glancing",
      description: "Tapered frustum slicing cylinder wall at 2.2°.",
      solidA: createCylinder3D(4.0, -4.0, 4.0, 16),
      solidB: rotateAndTranslateSolid3D(
        createConeFrustum3D(5.0, 1.5, -4.0, 4.0, 16),
        2.2,
        0,
        11.25,
        0.15,
        0,
        0,
      ),
      isClosedManifoldInput: true,
    },
    {
      id: "K4",
      name: "Two Cones Glancing Along Slant Generators (1.5°)",
      category: "glancing",
      description: "Counter-oriented cones grazing along slant walls.",
      solidA: cone,
      solidB: rotateAndTranslateSolid3D(cone, 181.5, 1.0, 11.25, 3.75, 0, 0),
      isClosedManifoldInput: true,
    },
    {
      id: "T1",
      name: "Two Coplanar Toroids Kissing Externally",
      category: "quadrics",
      description: "Equatorial outer rim contact.",
      solidA: torus,
      solidB: rotateAndTranslateSolid3D(torus, 0, 0, 0, 12.6, 0, 0),
      isClosedManifoldInput: true,
    },
    {
      id: "T2",
      name: "Interlocking Hopf-Link Toroids Grazing at 2°",
      category: "glancing",
      description: "Perpendicular linked toroids grazing inner bore.",
      solidA: torus,
      solidB: rotateAndTranslateSolid3D(torus, 90, 2.0, 0, 4.5, 0, 0),
      isClosedManifoldInput: true,
    },
    {
      id: "T3",
      name: "Cylinder Threading & Grazing Torus Saddle Bore (2°)",
      category: "glancing",
      description: "Cylinder grazing torus inner saddle ring.",
      solidA: torus,
      solidB: rotateAndTranslateSolid3D(
        createCylinder3D(2.75, -4.0, 4.0, 16),
        2.0,
        -1.5,
        5.0,
        0.1,
        0,
        0,
      ),
      isClosedManifoldInput: true,
    },
    {
      id: "T4",
      name: "Sphere Resting in Torus Central Saddle Well (1.5°)",
      category: "glancing",
      description: "Sphere grazing inner hyperbolic funnel of torus.",
      solidA: torus,
      solidB: rotateAndTranslateSolid3D(
        createUVSphere3D(3.2, 12, 12),
        1.5,
        -1.0,
        10.0,
        0,
        0,
        0.8,
      ),
      isClosedManifoldInput: true,
    },
    {
      id: "T5",
      name: "Coaxial Stacked Toroids Kissing at 0.9° Glancing Tilt",
      category: "glancing",
      description: "Two stacked toroids kissing along z=1.8 ring.",
      solidA: torus,
      solidB: rotateAndTranslateSolid3D(torus, 0.9, -0.6, 11.25, 0, 0, 3.52),
      isClosedManifoldInput: true,
    },
    {
      id: "P1",
      name: "Multi-Hole Flange Plate Cut by 1.3° Glancing Frustum",
      category: "glancing",
      description: "Flange plate with 2 holes breached at 1.3°.",
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
        -2.0,
        2.0,
      ),
      solidB: rotateAndTranslateSolid3D(frustum, 1.3, -0.9, 7.5, 0.2, 0.2, 0),
      isClosedManifoldInput: true,
    },
    {
      id: "P2",
      name: "Intermeshing Comb Teeth Grazing at 1.1°",
      category: "glancing",
      description: "Multi-tooth comb contact at 1.1°.",
      solidA: comb,
      solidB: rotateAndTranslateSolid3D(comb, 180, 1.1, 0.8, 2.0, 1.5, 0.2),
      isClosedManifoldInput: true,
    },
    {
      id: "P3",
      name: "8-Tooth Star Gear Grazing Torus at 1.8°",
      category: "glancing",
      description: "Star gear teeth piercing torus ring at 1.8°.",
      solidA: star,
      solidB: rotateAndTranslateSolid3D(torus, 1.8, -1.2, 7.5, 0, 0, 0.3),
      isClosedManifoldInput: true,
    },
    {
      id: "P4",
      name: "Coplanar Overlapping Hexahedra (4 Shared Planes)",
      category: "standard",
      description: "Boxes sharing y=±4 and z=±4 planes.",
      solidA: createBox3D([-6, -4, -4], [2, 4, 4]),
      solidB: createBox3D([-2, -4, -4], [6, 4, 4]),
      isClosedManifoldInput: true,
    },
    {
      id: "R1",
      name: "Closed Box with Dangling External Fin",
      category: "robustness",
      description: "Zero-volume dangling fin protruding into B.",
      solidA: boxWithFin,
      solidB: rotateAndTranslateSolid3D(
        createBox3D([1, -3, -3], [6, 3, 3]),
        4,
        -3,
        5,
        0,
        0,
        0,
      ),
      isClosedManifoldInput: false,
    },
    {
      id: "R2",
      name: "Tilted Hexahedron & Cylinder Multi-Plane Cut",
      category: "standard",
      description: "General tilted box and cylinder intersection.",
      solidA: box,
      solidB: rotateAndTranslateSolid3D(cyl, 22, -15, 9, 1.2, -0.8, 0.4),
      isClosedManifoldInput: true,
    },
  ];
}
