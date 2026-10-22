import { LIBRARY_ADAPTERS_3D } from "./adapters3d.js";
import { initManifoldWasm } from "./manifold-wasm.js";
import { runFullBenchmarkSuite3D } from "./runner3d.js";

function pad(str: string, width: number, rightAlign = false): string {
  if (str.length >= width) return str.slice(0, width);
  const spaces = " ".repeat(width - str.length);
  return rightAlign ? spaces + str : str + spaces;
}

function logProgress(msg: string): void {
  if (process.stderr.isTTY) {
    process.stderr.write(`\r${msg.padEnd(76)}`);
  } else {
    console.error(msg);
  }
}

async function main(): Promise<void> {
  await initManifoldWasm();

  console.log(
    "================================================================================================================",
  );
  console.log("  3D PLANAR B-REP & MESH CSG KERNEL BENCHMARK SUITE");
  console.log(
    "  Comparing polystrata-3d vs. manifold-3d (C++/WASM), three-bvh-csg, @jscad/modeling",
  );
  console.log(
    "================================================================================================================\n",
  );

  console.log("3D Libraries under test:");
  for (const lib of LIBRARY_ADAPTERS_3D) {
    console.log(
      `  - ${pad(lib.name, 20)} v${pad(lib.version, 8)} [${pad(lib.representation, 29)}] ${lib.algorithm}`,
    );
  }
  console.log("");

  const report3D = runFullBenchmarkSuite3D(60, logProgress);
  if (process.stderr.isTTY) process.stderr.write("\r" + " ".repeat(76) + "\r");

  for (const summary of report3D.workloadSummaries) {
    const w = summary.workload;
    const facesA = w.solidA.faces.length;
    const facesB = w.solidB.faces.length;
    console.log(
      "----------------------------------------------------------------------------------------------------------------",
    );
    console.log(
      `3D Workload: ${w.name} [${w.category.toUpperCase()}] (Input Faces: A=${facesA}, B=${facesB})`,
    );
    console.log(`             ${w.description}`);
    console.log(
      "----------------------------------------------------------------------------------------------------------------",
    );
    console.log(
      `${pad("Library", 18)} | ${pad("Ops/sec", 10, true)} | ${pad("Mean (ms)", 10, true)} | ${pad("p50 (ms)", 9, true)} | ${pad("p95 (ms)", 9, true)} | ${pad("Out Faces/Tris", 15, true)} | ${pad("Vol Err", 9, true)} | ${pad("Unmatched", 9, true)} | Status`,
    );
    console.log(
      `${"-".repeat(18)}-+-${"-".repeat(10)}-+-${"-".repeat(10)}-+-${"-".repeat(9)}-+-${"-".repeat(9)}-+-${"-".repeat(15)}-+-${"-".repeat(9)}-+-${"-".repeat(9)}+--------`,
    );

    for (const r of summary.results) {
      const opsStr = r.opsPerSec > 0 ? Math.round(r.opsPerSec).toLocaleString() : "ERR";
      const meanStr = Number.isFinite(r.meanMs) ? r.meanMs.toFixed(3) : "ERR";
      const p50Str = Number.isFinite(r.p50Ms) ? r.p50Ms.toFixed(3) : "ERR";
      const p95Str = Number.isFinite(r.p95Ms) ? r.p95Ms.toFixed(3) : "ERR";
      const meshStr =
        r.unionFaceCount > 0 || r.unionTriangleCount > 0
          ? `${r.unionFaceCount}F / ${r.unionTriangleCount}T`
          : "—";
      const errStr = Number.isFinite(r.volumeIdentityError)
        ? r.volumeIdentityError.toExponential(1)
        : "NaN";
      const umStr = r.totalUnmatchedHalfEdges >= 0 ? String(r.totalUnmatchedHalfEdges) : "ERR";
      const status = r.succeeded
        ? "PASS"
        : r.errorMessage
          ? "CRASH"
          : r.totalUnmatchedHalfEdges > 0
            ? "NON-MANIFOLD"
            : "VOL-FAIL";
      console.log(
        `${pad(r.libraryName, 18)} | ${pad(opsStr, 10, true)} | ${pad(meanStr, 10, true)} | ${pad(p50Str, 9, true)} | ${pad(p95Str, 9, true)} | ${pad(meshStr, 15, true)} | ${pad(errStr, 9, true)} | ${pad(umStr, 9, true)} | ${status}`,
      );
    }
    console.log("");
  }

  console.log(
    "================================================================================================================",
  );
  console.log(
    "  3D ADVERSARIAL GLANCING & DEGENERACY STRESS SUITE (24 Scenarios x 3 Ops = 72 3D Boolean Ops)",
  );
  console.log(
    "================================================================================================================",
  );
  console.log(
    `${pad("Library", 20)} | ${pad("Ops/sec", 12, true)} | ${pad("Total (ms)", 11, true)} | ${pad("2-Manifold", 14, true)} | ${pad("Vol Identity", 14, true)} | ${pad("Fully Valid", 14, true)} | ${pad("Crashes", 8, true)}`,
  );
  console.log(
    `${"-".repeat(20)}-+-${"-".repeat(12)}-+-${"-".repeat(11)}-+-${"-".repeat(14)}-+-${"-".repeat(14)}-+-${"-".repeat(14)}-+-${"-".repeat(8)}`,
  );

  for (const st of report3D.stressSummaries) {
    const mStr = `${st.manifoldPass}/${st.scenarios}`;
    const vStr = `${st.volumeIdentityPass}/${st.scenarios}`;
    const fvPct = ((st.fullyValidScenarios / st.scenarios) * 100).toFixed(1);
    const fvStr = `${st.fullyValidScenarios}/${st.scenarios} (${fvPct}%)`;
    console.log(
      `${pad(st.libraryName, 20)} | ${pad(Math.round(st.opsPerSec).toLocaleString(), 12, true)} | ${pad(st.totalMs.toFixed(1), 11, true)} | ${pad(mStr, 14, true)} | ${pad(vStr, 14, true)} | ${pad(fvStr, 14, true)} | ${pad(String(st.crashes), 8, true)}`,
    );
  }
  console.log("");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
