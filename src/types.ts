/**
 * Standard 2-body Boolean operation selector.
 */
export type BooleanOp = "union" | "intersect" | "difference" | "xor";

/**
 * Winding rule used to classify inside/outside of a 3D solid from its signed 3D solid-angle winding number:
 * - `"positive"`: `w >= 1` (standard CAD/B-Rep convention where outward shell = +1, interior void = -1)
 * - `"nonZero"`: `w !== 0`
 * - `"evenOdd"`: `|w| % 2 === 1`
 */
export type WindingRule = "positive" | "nonZero" | "evenOdd";

/**
 * Configuration options for the 3D Stratified Tolerance Complex kernel.
 */
export interface KernelOptions {
  /**
   * Explicit base snap tolerance `r0` in world units.
   * When omitted or `<= 0`, the kernel automatically selects a scale-aware tolerance
   * (`1e-9 * boundingBoxDiagonal`) and caps vertex tolerance ball growth at `r_max = 4 * r0`.
   */
  snapTolerance?: number;
  /**
   * Merge collinear degree-2 vertices on output boundary loops (default: `true`).
   */
  simplifyCollinear?: boolean;
  /**
   * Winding rule used to determine inside/outside for each input solid (default: `"positive"`).
   */
  windingRule?: WindingRule;
}

/**
 * A 3D point in IEEE-754 double-precision (`f64`) world coordinates.
 */
export interface Point3D {
  x: number;
  y: number;
  z: number;
}

/**
 * A 3-element coordinate tuple `[x, y, z]`.
 */
export type TuplePoint3D = readonly [number, number, number];

/**
 * A planar 3D polygon face with an outward-oriented outer boundary loop and zero or more interior holes.
 */
export interface Face3D {
  /** Outer boundary loop in CCW order when viewed against `normal`. */
  outer: Point3D[];
  /** Optional interior hole loops in CW order when viewed against `normal`. */
  holes: Point3D[][];
  /** Outward unit normal vector `(nx, ny, nz)`. */
  normal: Point3D;
}

/**
 * A 3D planar Boundary-Representation (B-Rep) solid.
 */
export interface Solid3D {
  faces: Face3D[];
}

/**
 * A single 3D surface triangle emitted by ear-clipping a `Face3D`.
 */
export interface Triangle3D {
  a: Point3D;
  b: Point3D;
  c: Point3D;
  normal: Point3D;
  faceIndex: number;
}

/**
 * Output of a 3D Boolean or topological healing operation.
 */
export interface BRepResult3D {
  /** Closed watertight 3D planar B-Rep solid. */
  solid: Solid3D;
  /** Triangulated surface mesh of `solid` for immediate 3D WebGL/Canvas rendering. */
  triangles: Triangle3D[];
  /** Signed enclosed volume via the Divergence Theorem (`(1/6) ∑ a · (b × c)`). */
  totalVolume: number;
  /** Total boundary surface area. */
  totalSurfaceArea: number;
  /** Number of canonical 3D vertices (`0-strata`). */
  vertexCount: number;
  /** Number of canonical 3D edges (`1-strata`). */
  edgeCount: number;
  /** Number of unmatched directed half-edges (`0` for a closed watertight 2-manifold). */
  unmatchedHalfEdgeCount: number;
}

/**
 * Lightweight zero-copy TypedArray result returned by `prepareBooleanTask` / `prepareBooleanTask3D`.
 */
export interface WasmBRep3DResult {
  readonly faceCount: number;
  readonly loopCount: number;
  readonly triangleCount: number;
  readonly totalVolume: number;
  readonly totalSurfaceArea: number;
  readonly vertexCount: number;
  readonly edgeCount: number;
  readonly unmatchedHalfEdgeCount: number;
  readonly coords: Float64Array;
  readonly loopMeta: Uint32Array;
  readonly faceNormals: Float64Array;
  readonly triangles: Float64Array;
}
