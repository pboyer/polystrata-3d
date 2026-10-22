import { LIBRARY_ADAPTERS_3D } from "../bench/adapters3d.js";
import { initManifoldWasm } from "../bench/manifold-wasm.js";
import {
  type FullBenchmarkReport3D,
  runFullBenchmarkSuite3D,
} from "../bench/runner3d.js";
import { initPolystrata3D } from "../src/index.js";

const LIB_COLORS: Record<string, string> = {
  "polystrata-3d": "#38bdf8",
  "manifold-3d": "#f472b6",
  "three-bvh-csg": "#fbbf24",
  "jscad-modeling": "#34d399",
};

const statusEl = document.getElementById("bench-status")!;
const cards3dEl = document.getElementById("lib-cards-3d")!;
const stress3dTbody = document.getElementById("stress-3d-tbody")!;
const workload3dTbody = document.getElementById("workload-3d-tbody")!;

const btnRunAll = document.getElementById("btn-run-all") as HTMLButtonElement;
const btnRunQuick = document.getElementById("btn-run-quick") as HTMLButtonElement;

function renderReport3D(report: FullBenchmarkReport3D): void {
  cards3dEl.innerHTML = "";
  for (const lib of LIBRARY_ADAPTERS_3D) {
    const workloadPassCount = report.workloadSummaries.filter((ws) => {
      const r = ws.results.find((x) => x.libraryId === lib.id);
      return r?.succeeded;
    }).length;
    const st = report.stressSummaries.find((s) => s.libraryId === lib.id);
    const manifoldRate = st
      ? `${st.manifoldPass}/${st.scenarios} (${((st.manifoldPass / st.scenarios) * 100).toFixed(0)}%)`
      : "—";

    const card = document.createElement("div");
    card.className = `lib-card ${lib.id === "polystrata-3d" ? "highlight" : ""}`;
    card.innerHTML = `
      <div class="lib-card-header">
        <span class="lib-card-name" style="color: ${LIB_COLORS[lib.id] ?? "#f1f5f9"}">${lib.name}</span>
        <span class="lib-card-ver">v${lib.version}</span>
      </div>
      <div class="lib-card-rep">${lib.representation}</div>
      <div class="lib-card-algo">${lib.algorithm}</div>
      <div class="lib-card-stats">
        <span>Workloads: <strong>${workloadPassCount}/${report.workloadSummaries.length} PASS</strong></span>
        <span>2-Manifold: <strong>${manifoldRate}</strong></span>
      </div>
    `;
    cards3dEl.appendChild(card);
  }

  stress3dTbody.innerHTML = "";
  for (const st of report.stressSummaries) {
    const lib = LIBRARY_ADAPTERS_3D.find((l) => l.id === st.libraryId);
    const manifoldPct = (st.manifoldPass / st.scenarios) * 100;
    const validPct = (st.fullyValidScenarios / st.scenarios) * 100;
    let verdictBadge = `<span class="badge-pass">100% 2-MANIFOLD</span>`;
    if (st.crashes > 0 || validPct < 80) {
      verdictBadge = `<span class="badge-fail">${st.scenarios - st.manifoldPass} NON-MANIFOLD / ${st.crashes} CRASH</span>`;
    } else if (validPct < 100) {
      verdictBadge = `<span class="badge-warn">${validPct.toFixed(1)}% VALID</span>`;
    }

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong style="color: ${LIB_COLORS[st.libraryId] ?? "#fff"}">${st.libraryName}</strong></td>
      <td style="font-family: var(--font-mono); font-size: 11.5px; color: var(--text-secondary);">${lib?.representation ?? ""}</td>
      <td class="num">${Math.round(st.opsPerSec).toLocaleString()} op/s</td>
      <td class="num">${st.totalMs.toFixed(1)} ms</td>
      <td class="num">${st.manifoldPass} / ${st.scenarios} (${manifoldPct.toFixed(0)}%)</td>
      <td class="num">${st.volumeIdentityPass} / ${st.scenarios}</td>
      <td class="num">${st.fullyValidScenarios} / ${st.scenarios} (${validPct.toFixed(1)}%)</td>
      <td class="num">${st.crashes}</td>
      <td>${verdictBadge}</td>
    `;
    stress3dTbody.appendChild(tr);
  }

  workload3dTbody.innerHTML = "";
  for (const summary of report.workloadSummaries) {
    const w = summary.workload;
    const maxOps = Math.max(...summary.results.map((r) => r.opsPerSec), 1);

    const headerTr = document.createElement("tr");
    headerTr.className = "workload-header-row";
    headerTr.innerHTML = `
      <td colspan="9">
        ${w.name} <span style="color: var(--text-muted); font-weight: 400; margin-left: 8px;">[${w.category.toUpperCase()} &bull; Input Faces: ${w.solidA.faces.length} + ${w.solidB.faces.length}] &mdash; ${w.description}</span>
      </td>
    `;
    workload3dTbody.appendChild(headerTr);

    for (const r of summary.results) {
      const pct = Math.max(2, Math.min(100, (r.opsPerSec / maxOps) * 100));
      const color = LIB_COLORS[r.libraryId] ?? "#38bdf8";
      const meshStr =
        r.unionFaceCount > 0 || r.unionTriangleCount > 0
          ? `${r.unionFaceCount}F / ${r.unionTriangleCount}T`
          : "—";
      const statusBadge = r.succeeded
        ? `<span class="badge-pass">PASS</span>`
        : r.errorMessage
          ? `<span class="badge-fail" title="${r.errorMessage}">CRASH</span>`
          : r.totalUnmatchedHalfEdges > 0
            ? `<span class="badge-fail">NON-MANIFOLD (${r.totalUnmatchedHalfEdges})</span>`
            : `<span class="badge-warn">VOL-DRIFT</span>`;

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong style="color: ${color}">${r.libraryName}</strong></td>
        <td class="num">${r.opsPerSec > 0 ? Math.round(r.opsPerSec).toLocaleString() : "ERR"}</td>
        <td class="bar-cell">
          <div class="bar-track">
            <div class="bar-fill" style="width: ${pct.toFixed(1)}%; background: ${color};"></div>
          </div>
        </td>
        <td class="num">${Number.isFinite(r.meanMs) ? r.meanMs.toFixed(3) : "—"}</td>
        <td class="num">${Number.isFinite(r.p50Ms) ? r.p50Ms.toFixed(3) : "—"}</td>
        <td class="num">${meshStr}</td>
        <td class="num">${Number.isFinite(r.volumeIdentityError) ? r.volumeIdentityError.toExponential(1) : "NaN"}</td>
        <td class="num">${r.totalUnmatchedHalfEdges >= 0 ? r.totalUnmatchedHalfEdges : "ERR"}</td>
        <td>${statusBadge}</td>
      `;
      workload3dTbody.appendChild(tr);
    }
  }
}

async function executeSuite(durationMs: number): Promise<void> {
  btnRunAll.disabled = true;
  btnRunQuick.disabled = true;
  statusEl.textContent = "Running 3D benchmark suite...";

  await new Promise((r) => setTimeout(r, 30));

  const t0 = performance.now();
  const report3D = runFullBenchmarkSuite3D(durationMs, (msg) => {
    statusEl.textContent = msg;
  });
  renderReport3D(report3D);

  const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
  statusEl.textContent = `Completed 3D benchmark suite in ${elapsed}s`;
  btnRunAll.disabled = false;
  btnRunQuick.disabled = false;
}

btnRunAll.addEventListener("click", () => executeSuite(55));
btnRunQuick.addEventListener("click", () => executeSuite(20));

async function initPage(): Promise<void> {
  try {
    statusEl.textContent = "Loading polystrata-3d & manifold-3d WASM...";
    await Promise.all([initPolystrata3D(), initManifoldWasm()]);
    await executeSuite(20);
  } catch (err) {
    statusEl.textContent = `Error initializing benchmarks: ${err instanceof Error ? err.message : String(err)}`;
  }
}

initPage();
