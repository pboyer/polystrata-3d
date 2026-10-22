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
} from "./types.js";

export {
  initPolystrata,
  initPolystrata3D,
  initPolystrataWasm,
} from "./wasm-kernel.js";

export {
  clip,
  clip3D,
  computeLoopNormal,
  computeLoopNormal3D,
  createBox,
  createBox3D,
  difference,
  difference3D,
  extrudePolygon,
  extrudePolygon3D,
  heal,
  heal3D,
  intersect,
  intersect3D,
  prepareBooleanTask,
  prepareBooleanTask3D,
  transformSolid,
  transformSolid3D,
  union,
  union3D,
  xor,
  xor3D,
} from "./brep3d.js";
