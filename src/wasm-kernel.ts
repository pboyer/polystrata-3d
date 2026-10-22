import { WASM_BASE64 } from "./wasm-bytes.js";

export interface Polystrata3DWasmExports {
  memory: WebAssembly.Memory;
  wasm_brep3d_alloc_input: (f64Count: number) => number;
  wasm_brep3d_execute: (
    opCode: number,
    windingRule: number,
    snapFactor: number,
    simplifyCollinear: number,
  ) => number;
  wasm_brep3d_get_out_coords_ptr: () => number;
  wasm_brep3d_get_out_coords_len: () => number;
  wasm_brep3d_get_out_loop_meta_ptr: () => number;
  wasm_brep3d_get_out_loop_count: () => number;
  wasm_brep3d_get_out_face_normals_ptr: () => number;
  wasm_brep3d_get_out_face_count: () => number;
  wasm_brep3d_get_out_triangles_ptr: () => number;
  wasm_brep3d_get_out_triangles_len: () => number;
  wasm_brep3d_get_out_volume: () => number;
  wasm_brep3d_get_out_area: () => number;
  wasm_brep3d_get_out_vertex_count: () => number;
  wasm_brep3d_get_out_edge_count: () => number;
  wasm_brep3d_get_out_unmatched_he_count: () => number;
}

let wasmExports: Polystrata3DWasmExports | null = null;
let wasmInitPromise: Promise<void> | null = null;

function decodeBase64(b64: string): Uint8Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    bytes[i] = bin.charCodeAt(i);
  }
  return bytes;
}

let cachedWasmBytes: Uint8Array | null = null;
function getWasmBytes(): Uint8Array {
  if (cachedWasmBytes === null) {
    cachedWasmBytes = decodeBase64(WASM_BASE64);
  }
  return cachedWasmBytes;
}

/**
 * Asynchronously initialize the embedded 3D WebAssembly module.
 *
 * In Node.js, Bun, Deno, and Web Workers, `polystrata-3d` initializes synchronously on first call
 * so calling `initPolystrata3D()` is optional. On browser main threads that restrict synchronous
 * `new WebAssembly.Module()` compilation for modules > 4 KB, call `await initPolystrata3D()` once
 * at application startup before invoking synchronous 3D Boolean operations.
 */
export async function initPolystrata3D(): Promise<void> {
  if (wasmExports !== null) return;
  if (wasmInitPromise !== null) return wasmInitPromise;
  wasmInitPromise = (async () => {
    const bytes = getWasmBytes();
    const { instance } = await WebAssembly.instantiate(bytes as BufferSource, {});
    wasmExports = instance.exports as unknown as Polystrata3DWasmExports;
  })();
  return wasmInitPromise;
}

/**
 * Alias for {@link initPolystrata3D}.
 */
export const initPolystrata = initPolystrata3D;

/**
 * Alias for {@link initPolystrata3D}.
 */
export const initPolystrataWasm = initPolystrata3D;

export function getWasm(): Polystrata3DWasmExports {
  if (wasmExports !== null) return wasmExports;
  const bytes = getWasmBytes();
  try {
    const mod = new WebAssembly.Module(bytes as BufferSource);
    const instance = new WebAssembly.Instance(mod, {});
    wasmExports = instance.exports as unknown as Polystrata3DWasmExports;
    return wasmExports;
  } catch (err) {
    throw new Error(
      `polystrata-3d: Synchronous WebAssembly compilation failed on this thread (${err instanceof Error ? err.message : String(err)}). ` +
        `Call \`await initPolystrata3D()\` once at startup before invoking synchronous 3D Boolean operations on the browser main thread.`,
    );
  }
}
