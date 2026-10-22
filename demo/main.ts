import {
  type BooleanOp,
  type BRepResult3D,
  clip3D,
  heal3D,
  initPolystrata3D,
  type Solid3D,
  type WindingRule,
} from "../src/index.js";
import {
  audit3DBooleanInvariants,
  type Camera3D,
  DEMO_SCENARIOS_3D,
  type RenderOptions3D,
  renderBRep3DScene,
  type Scenario3D,
  type Scenario3DCategory,
} from "./brep3d-lab.js";

type DemoOp3D = BooleanOp | "heal";

let currentScenario3D: Scenario3D = DEMO_SCENARIOS_3D[0]!;
let current3DCategoryFilter: Scenario3DCategory | "all" = "all";
let currentOp: DemoOp3D = currentScenario3D.defaultOp;
let currentWindingRule: WindingRule = "positive";
let customSnapTolerance: number | null = null;
let simplifyCollinear = true;

let transform3D = {
  dx: 0,
  dy: 0,
  dz: 0,
  rxDeg: 0,
  ryDeg: 0,
  rotDeg: 0,
  detailScale: 1,
};

const cam3D: Camera3D = {
  yaw: -0.68,
  pitch: 0.52,
  distance: 34,
  panX: 0,
  panY: 0,
};

const renderOpts3D: RenderOptions3D = {
  showInputWireframe: true,
  showShadedFaces: true,
  showBRepEdges: true,
  showNormals: false,
  colorByFace: false,
};

let latest3DSolidA: Solid3D | null = null;
let latest3DSolidB: Solid3D | null = null;
let latest3DResult: BRepResult3D | null = null;
let drag3D: {
  mode: "orbit" | "pan";
  lastX: number;
  lastY: number;
} | null = null;
let spacePressed = false;

const canvas = document.getElementById("kernel-canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;

function resizeCanvas(): void {
  const rect = canvas.parentElement!.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  renderCanvas();
}

function renderCanvas(): void {
  const rect = canvas.parentElement!.getBoundingClientRect();
  if (latest3DSolidA && latest3DSolidB && latest3DResult) {
    renderBRep3DScene(
      ctx,
      rect.width,
      rect.height,
      latest3DSolidA,
      latest3DSolidB,
      latest3DResult,
      cam3D,
      renderOpts3D,
    );
  }
}

function syncControlsFromState(): void {
  document.getElementById("active-scenario-title")!.textContent = currentScenario3D.title;
  document.getElementById("scenario-description")!.textContent = currentScenario3D.description;

  document.querySelectorAll("#op-selector .op-btn").forEach((btn) => {
    const el = btn as HTMLButtonElement;
    el.classList.toggle("active", el.dataset.op === currentOp);
  });

  document.querySelectorAll("#winding-selector .seg-btn").forEach((btn) => {
    const el = btn as HTMLButtonElement;
    el.classList.toggle("active", el.dataset.rule === currentWindingRule);
  });

  const tolValEl = document.getElementById("val-snap-tol")!;
  const tolSlider = document.getElementById("slider-snap-tol") as HTMLInputElement;
  if (customSnapTolerance === null) {
    tolValEl.textContent = "Auto";
    tolSlider.value = "-9";
  } else {
    tolValEl.textContent = customSnapTolerance.toExponential(1);
    tolSlider.value = String(Math.log10(customSnapTolerance));
  }

  document
    .getElementById("btn-tol-auto")!
    .classList.toggle("active", customSnapTolerance === null);
  document
    .getElementById("btn-tol-micro")!
    .classList.toggle("active", customSnapTolerance === 1e-4);
  document
    .getElementById("btn-tol-coarse")!
    .classList.toggle("active", customSnapTolerance === 0.25);

  document.querySelectorAll("#example-3d-list .example-card").forEach((card) => {
    const el = card as HTMLElement;
    el.classList.toggle("active", el.dataset.id === currentScenario3D.id);
  });
}

function recomputeAndRender(): void {
  const { solidA, solidB } = currentScenario3D.buildSolids(transform3D);
  latest3DSolidA = solidA;
  latest3DSolidB = solidB;
  const op3D: BooleanOp =
    currentOp === "union" ||
    currentOp === "intersect" ||
    currentOp === "difference" ||
    currentOp === "xor"
      ? currentOp
      : "difference";
  const kernelOpts = {
    windingRule: currentWindingRule,
    snapTolerance: customSnapTolerance ?? undefined,
    simplifyCollinear,
  };
  const t0 = performance.now();
  const res =
    currentOp === "heal"
      ? heal3D({ faces: [...solidA.faces, ...solidB.faces] }, kernelOpts)
      : clip3D(op3D, solidA, solidB, kernelOpts);
  const elapsedMs = performance.now() - t0;
  latest3DResult = res;

  const audit = audit3DBooleanInvariants(solidA, solidB, res, op3D, kernelOpts);

  document.getElementById("metric-time")!.textContent = `${elapsedMs.toFixed(2)} ms`;
  document.getElementById("metric-ve")!.textContent = `${res.vertexCount} V / ${res.edgeCount} E`;
  document.getElementById("metric-loops")!.textContent =
    `${res.solid.faces.length} F (${audit.holeLoopCount} holes)`;
  document.getElementById("metric-area")!.textContent = `Vol ${res.totalVolume.toFixed(1)}`;

  const diagEl = document.getElementById("pipeline-diagnostics")!;
  diagEl.innerHTML = `
    <div class="diag-row"><span>3D Enclosed Volume:</span><span class="diag-val">${res.totalVolume.toFixed(3)}</span></div>
    <div class="diag-row"><span>3D Surface Area:</span><span class="diag-val">${res.totalSurfaceArea.toFixed(3)}</span></div>
    <div class="diag-row"><span>Canonical 0/1-Strata:</span><span class="diag-val">${res.vertexCount} V / ${res.edgeCount} E</span></div>
    <div class="diag-row"><span>Unified Planar 2-Strata:</span><span class="diag-val">${res.solid.faces.length} faces (${audit.holeLoopCount} holes)</span></div>
    <div class="diag-row"><span>Unmatched Half-Edges:</span><span class="diag-val" style="color:${res.unmatchedHalfEdgeCount === 0 ? "#34d399" : "#fb7185"}">${res.unmatchedHalfEdgeCount} (${res.unmatchedHalfEdgeCount === 0 ? "2-Manifold ✓" : "Open"})</span></div>
    <div class="diag-row"><span>Euler &chi; = V&minus;E+F&minus;H:</span><span class="diag-val">${audit.eulerCharacteristic}</span></div>
    <div class="diag-row"><span>Vol(A&cup;B)+Vol(A&cap;B):</span><span class="diag-val">${(audit.volUnion + audit.volIntersect).toFixed(2)}</span></div>
    <div class="diag-row"><span>Inclusion-Exclusion Err:</span><span class="diag-val" style="color:${audit.inclusionExclusionPass ? "#34d399" : "#fb7185"}">${audit.inclusionExclusionError.toExponential(2)} ${audit.inclusionExclusionPass ? "✓" : ""}</span></div>
  `;

  renderCanvas();
}

function setupUI(): void {
  const example3DListEl = document.getElementById("example-3d-list")!;
  function render3DPresetList(): void {
    example3DListEl.innerHTML = "";
    const filtered =
      current3DCategoryFilter === "all"
        ? DEMO_SCENARIOS_3D
        : DEMO_SCENARIOS_3D.filter((s) => s.category === current3DCategoryFilter);
    const badgeCountEl = document.getElementById("badge-3d-count");
    if (badgeCountEl) {
      badgeCountEl.textContent = `${filtered.length} presets`;
    }
    for (const s3d of filtered) {
      const card = document.createElement("button");
      card.className = `example-card${s3d.id === currentScenario3D.id ? " active" : ""}`;
      card.dataset.id = s3d.id;
      card.innerHTML = `
        <div class="example-card-top">
          <span class="example-title">${s3d.title}</span>
          <span class="example-tag">${s3d.badge}</span>
        </div>
        <div class="example-summary">${s3d.description}</div>
      `;
      card.addEventListener("click", () => {
        currentScenario3D = s3d;
        currentOp = s3d.defaultOp;
        transform3D = {
          dx: 0,
          dy: 0,
          dz: 0,
          rxDeg: 0,
          ryDeg: 0,
          rotDeg: 0,
          detailScale: transform3D.detailScale,
        };
        sync3DTransformSliders();
        syncControlsFromState();
        recomputeAndRender();
      });
      example3DListEl.appendChild(card);
    }
  }
  render3DPresetList();

  document.querySelectorAll("#filter-3d-categories .btn-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      document
        .querySelectorAll("#filter-3d-categories .btn-chip")
        .forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      current3DCategoryFilter = ((btn as HTMLElement).dataset.cat ?? "all") as
        | Scenario3DCategory
        | "all";
      render3DPresetList();
    });
  });

  function sync3DTransformSliders(): void {
    (document.getElementById("slider-3d-dx") as HTMLInputElement).value = String(transform3D.dx);
    (document.getElementById("slider-3d-dy") as HTMLInputElement).value = String(transform3D.dy);
    (document.getElementById("slider-3d-dz") as HTMLInputElement).value = String(transform3D.dz);
    (document.getElementById("slider-3d-rx") as HTMLInputElement).value = String(transform3D.rxDeg);
    (document.getElementById("slider-3d-ry") as HTMLInputElement).value = String(transform3D.ryDeg);
    (document.getElementById("slider-3d-rot") as HTMLInputElement).value = String(
      transform3D.rotDeg,
    );
    document.getElementById("val-3d-dx")!.textContent = transform3D.dx.toFixed(2);
    document.getElementById("val-3d-dy")!.textContent = transform3D.dy.toFixed(2);
    document.getElementById("val-3d-dz")!.textContent = transform3D.dz.toFixed(2);
    document.getElementById("val-3d-rx")!.textContent = `${transform3D.rxDeg.toFixed(2)}°`;
    document.getElementById("val-3d-ry")!.textContent = `${transform3D.ryDeg.toFixed(2)}°`;
    document.getElementById("val-3d-rot")!.textContent = `${transform3D.rotDeg}°`;
  }

  (["dx", "dy", "dz", "rx", "ry", "rot"] as const).forEach((axis) => {
    const slider = document.getElementById(`slider-3d-${axis}`) as HTMLInputElement;
    slider.addEventListener("input", () => {
      const v = parseFloat(slider.value);
      if (axis === "dx") transform3D.dx = v;
      else if (axis === "dy") transform3D.dy = v;
      else if (axis === "dz") transform3D.dz = v;
      else if (axis === "rx") transform3D.rxDeg = v;
      else if (axis === "ry") transform3D.ryDeg = v;
      else transform3D.rotDeg = v;
      sync3DTransformSliders();
      recomputeAndRender();
    });
  });

  document.querySelectorAll("#detail-3d-selector .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document
        .querySelectorAll("#detail-3d-selector .seg-btn")
        .forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      transform3D.detailScale = parseFloat((btn as HTMLElement).dataset.scale ?? "1");
      recomputeAndRender();
    });
  });

  const resetTransform = () => {
    transform3D = {
      dx: 0,
      dy: 0,
      dz: 0,
      rxDeg: 0,
      ryDeg: 0,
      rotDeg: 0,
      detailScale: transform3D.detailScale,
    };
    sync3DTransformSliders();
    recomputeAndRender();
  };

  document.getElementById("btn-3d-reset-transform")!.addEventListener("click", resetTransform);
  document.getElementById("btn-reset-geometry")!.addEventListener("click", resetTransform);

  document.getElementById("btn-fit-view")!.addEventListener("click", () => {
    cam3D.yaw = -0.68;
    cam3D.pitch = 0.52;
    cam3D.distance = 34;
    cam3D.panX = 0;
    cam3D.panY = 0;
    renderCanvas();
  });

  document.getElementById("chk-3d-inputs")!.addEventListener("change", (e) => {
    renderOpts3D.showInputWireframe = (e.target as HTMLInputElement).checked;
    renderCanvas();
  });
  document.getElementById("chk-3d-shaded")!.addEventListener("change", (e) => {
    renderOpts3D.showShadedFaces = (e.target as HTMLInputElement).checked;
    renderCanvas();
  });
  document.getElementById("chk-3d-edges")!.addEventListener("change", (e) => {
    renderOpts3D.showBRepEdges = (e.target as HTMLInputElement).checked;
    renderCanvas();
  });
  document.getElementById("chk-3d-normals")!.addEventListener("change", (e) => {
    renderOpts3D.showNormals = (e.target as HTMLInputElement).checked;
    renderCanvas();
  });
  document.getElementById("chk-3d-color-face")!.addEventListener("change", (e) => {
    renderOpts3D.colorByFace = (e.target as HTMLInputElement).checked;
    renderCanvas();
  });

  document.querySelectorAll("#op-selector .op-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      currentOp = (btn as HTMLButtonElement).dataset.op as DemoOp3D;
      syncControlsFromState();
      recomputeAndRender();
    });
  });

  document.querySelectorAll("#winding-selector .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      currentWindingRule = (btn as HTMLButtonElement).dataset.rule as WindingRule;
      syncControlsFromState();
      recomputeAndRender();
    });
  });

  const tolSlider = document.getElementById("slider-snap-tol") as HTMLInputElement;
  tolSlider.addEventListener("input", () => {
    const exp = parseFloat(tolSlider.value);
    if (exp <= -8.95) {
      customSnapTolerance = null;
    } else {
      customSnapTolerance = Math.pow(10, exp);
    }
    syncControlsFromState();
    recomputeAndRender();
  });

  document.getElementById("btn-tol-auto")!.addEventListener("click", () => {
    customSnapTolerance = null;
    syncControlsFromState();
    recomputeAndRender();
  });
  document.getElementById("btn-tol-micro")!.addEventListener("click", () => {
    customSnapTolerance = 1e-4;
    syncControlsFromState();
    recomputeAndRender();
  });
  document.getElementById("btn-tol-coarse")!.addEventListener("click", () => {
    customSnapTolerance = 0.25;
    syncControlsFromState();
    recomputeAndRender();
  });

  document.getElementById("chk-simplify-collinear")!.addEventListener("change", (e) => {
    simplifyCollinear = (e.target as HTMLInputElement).checked;
    recomputeAndRender();
  });

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") spacePressed = true;
  });
  window.addEventListener("keyup", (e) => {
    if (e.code === "Space") spacePressed = false;
  });

  canvas.addEventListener("mousedown", (e) => {
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    drag3D = {
      mode: e.button === 1 || e.shiftKey || spacePressed ? "pan" : "orbit",
      lastX: sx,
      lastY: sy,
    };
    canvas.style.cursor = "grabbing";
  });

  window.addEventListener("mousemove", (e) => {
    if (!drag3D) return;
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const dx = sx - drag3D.lastX;
    const dy = sy - drag3D.lastY;
    drag3D.lastX = sx;
    drag3D.lastY = sy;

    if (drag3D.mode === "orbit") {
      cam3D.yaw += dx * 0.009;
      cam3D.pitch = Math.max(-1.45, Math.min(1.45, cam3D.pitch + dy * 0.009));
    } else {
      cam3D.panX += dx;
      cam3D.panY += dy;
    }
    renderCanvas();
  });

  window.addEventListener("mouseup", () => {
    drag3D = null;
    canvas.style.cursor = "grab";
  });

  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const factor = Math.exp(e.deltaY * 0.0012);
      cam3D.distance = Math.max(8, Math.min(120, cam3D.distance * factor));
      renderCanvas();
    },
    { passive: false },
  );

  window.addEventListener("resize", resizeCanvas);
}

async function boot(): Promise<void> {
  await initPolystrata3D();
  setupUI();
  syncControlsFromState();
  resizeCanvas();
  recomputeAndRender();
}

boot().catch((err) => {
  console.error("Failed to initialize polystrata-3d lab:", err);
});
