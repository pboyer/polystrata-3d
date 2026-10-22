import { heal3D } from "../src/index.js";
import { LIBRARY_ADAPTERS_3D, type LibraryAdapter3D } from "./adapters3d.js";
import {
  type BenchOp3D,
  type BenchWorkload3D,
  createAdversarial3DStressScenarios,
  createBenchmarkWorkloads3D,
} from "./workloads3d.js";

export interface WorkloadLibraryResult3D {
  libraryId: string;
  libraryName: string;
  opsPerSec: number;
  meanMs: number;
  p50Ms: number;
  p95Ms: number;
  unionVolume: number;
  intersectVolume: number;
  differenceVolume: number;
  volumeIdentityError: number;
  unionFaceCount: number;
  unionTriangleCount: number;
  totalUnmatchedHalfEdges: number;
  succeeded: boolean;
  errorMessage?: string;
}

export interface WorkloadBenchmarkSummary3D {
  workload: BenchWorkload3D;
  referenceVolumeA: number;
  referenceVolumeB: number;
  results: WorkloadLibraryResult3D[];
}

export interface StressSummary3D {
  libraryId: string;
  libraryName: string;
  scenarios: number;
  totalOps: number;
  crashes: number;
  manifoldPass: number;
  volumeIdentityPass: number;
  fullyValidScenarios: number;
  totalMs: number;
  opsPerSec: number;
}

export interface FullBenchmarkReport3D {
  timestamp: string;
  workloadSummaries: WorkloadBenchmarkSummary3D[];
  stressSummaries: StressSummary3D[];
}

const ALL_OPS_3D: BenchOp3D[] = ["union", "intersect", "difference"];

function measureWorkloadForLibrary3D(
  workload: BenchWorkload3D,
  adapter: LibraryAdapter3D,
  refVolA: number,
  refVolB: number,
  targetDurationMs = 90,
): WorkloadLibraryResult3D {
  let task: ReturnType<LibraryAdapter3D["prepare"]> | null = null;
  try {
    task = adapter.prepare(workload.solidA, workload.solidB);

    // Warmup & topological/volume verification across union, intersect, difference
    const resU = task.run("union");
    const resI = task.run("intersect");
    const resD = task.run("difference");

    const inspU = task.inspect(resU);
    const inspI = task.inspect(resI);
    const inspD = task.inspect(resD);

    const err1 = Math.abs(inspU.volume + inspI.volume - (refVolA + refVolB));
    const err2 = Math.abs(inspD.volume + inspI.volume - refVolA);
    const volumeIdentityError = Math.max(err1, err2);
    const totalUnmatchedHalfEdges =
      inspU.unmatchedHalfEdges + inspI.unmatchedHalfEdges + inspD.unmatchedHalfEdges;

    // Short warmup loop (8ms)
    const warmupStart = performance.now();
    while (performance.now() - warmupStart < 8) {
      for (const op of ALL_OPS_3D) {
        task.run(op);
      }
    }

    // Timed iterations (each sample runs union, intersect, difference)
    const sampleTimesMs: number[] = [];
    const startWall = performance.now();

    while (
      (performance.now() - startWall < targetDurationMs && sampleTimesMs.length < 120) ||
      sampleTimesMs.length < 2
    ) {
      const t0 = performance.now();
      for (const op of ALL_OPS_3D) {
        task.run(op);
      }
      const dt = (performance.now() - t0) / ALL_OPS_3D.length;
      sampleTimesMs.push(dt);
    }

    task.dispose?.();
    task = null;

    sampleTimesMs.sort((a, b) => a - b);
    const trimCount =
      sampleTimesMs.length >= 10
        ? Math.max(1, Math.floor(sampleTimesMs.length * 0.9))
        : sampleTimesMs.length;
    let sum = 0;
    for (let i = 0; i < trimCount; i++) {
      sum += sampleTimesMs[i]!;
    }
    const meanMs = sum / trimCount;
    const p50Ms = sampleTimesMs[Math.floor(sampleTimesMs.length * 0.5)]!;
    const p95Ms = sampleTimesMs[Math.floor(sampleTimesMs.length * 0.95)]!;
    const opsPerSec = meanMs > 0 ? 1000 / meanMs : 0;

    return {
      libraryId: adapter.id,
      libraryName: adapter.name,
      opsPerSec,
      meanMs,
      p50Ms,
      p95Ms,
      unionVolume: inspU.volume,
      intersectVolume: inspI.volume,
      differenceVolume: inspD.volume,
      volumeIdentityError,
      unionFaceCount: inspU.faceCount,
      unionTriangleCount: inspU.triangleCount,
      totalUnmatchedHalfEdges,
      succeeded:
        Number.isFinite(volumeIdentityError) &&
        volumeIdentityError < 1.5 &&
        totalUnmatchedHalfEdges === 0,
    };
  } catch (err) {
    task?.dispose?.();
    return {
      libraryId: adapter.id,
      libraryName: adapter.name,
      opsPerSec: 0,
      meanMs: Infinity,
      p50Ms: Infinity,
      p95Ms: Infinity,
      unionVolume: NaN,
      intersectVolume: NaN,
      differenceVolume: NaN,
      volumeIdentityError: Infinity,
      unionFaceCount: 0,
      unionTriangleCount: 0,
      totalUnmatchedHalfEdges: -1,
      succeeded: false,
      errorMessage: err instanceof Error ? err.message : String(err),
    };
  }
}

export function runAdversarial3DStressComparison(
  onProgress?: (msg: string) => void,
): StressSummary3D[] {
  const scenarios = createAdversarial3DStressScenarios();
  const refVolumes = scenarios.map((sc) => ({
    volA: heal3D(sc.solidA).totalVolume,
    volB: heal3D(sc.solidB).totalVolume,
  }));

  return LIBRARY_ADAPTERS_3D.map((adapter) => {
    onProgress?.(`Running 3D Adversarial Stress Suite on ${adapter.name}...`);
    let crashes = 0;
    let manifoldPass = 0;
    let volumeIdentityPass = 0;
    let fullyValidScenarios = 0;
    let totalExecMs = 0;

    for (let i = 0; i < scenarios.length; i++) {
      const sc = scenarios[i]!;
      const { volA, volB } = refVolumes[i]!;
      let task: ReturnType<LibraryAdapter3D["prepare"]> | null = null;
      try {
        const t0 = performance.now();
        task = adapter.prepare(sc.solidA, sc.solidB);
        const resU = task.run("union");
        const resI = task.run("intersect");
        const resD = task.run("difference");
        totalExecMs += performance.now() - t0;

        const inspU = task.inspect(resU);
        const inspI = task.inspect(resI);
        const inspD = task.inspect(resD);
        task.dispose?.();
        task = null;

        const um =
          inspU.unmatchedHalfEdges + inspI.unmatchedHalfEdges + inspD.unmatchedHalfEdges;
        const err1 = Math.abs(inspU.volume + inspI.volume - (volA + volB));
        const err2 = Math.abs(inspD.volume + inspI.volume - volA);
        const maxErr = Math.max(err1, err2);

        const isManifold = um === 0;
        const isVolOk = Number.isFinite(maxErr) && maxErr < 1.5;

        if (isManifold) manifoldPass++;
        if (isVolOk) volumeIdentityPass++;
        if (isManifold && isVolOk) fullyValidScenarios++;
      } catch {
        task?.dispose?.();
        crashes++;
      }
    }

    const totalOps = scenarios.length * ALL_OPS_3D.length;
    return {
      libraryId: adapter.id,
      libraryName: adapter.name,
      scenarios: scenarios.length,
      totalOps,
      crashes,
      manifoldPass,
      volumeIdentityPass,
      fullyValidScenarios,
      totalMs: totalExecMs,
      opsPerSec: totalExecMs > 0 ? (totalOps / totalExecMs) * 1000 : 0,
    };
  });
}

export function runFullBenchmarkSuite3D(
  targetDurationPerWorkloadMs = 90,
  onProgress?: (msg: string) => void,
): FullBenchmarkReport3D {
  const workloads = createBenchmarkWorkloads3D();
  const workloadSummaries: WorkloadBenchmarkSummary3D[] = [];

  for (const workload of workloads) {
    const refVolA = heal3D(workload.solidA).totalVolume;
    const refVolB = heal3D(workload.solidB).totalVolume;

    const results = LIBRARY_ADAPTERS_3D.map((adapter) => {
      onProgress?.(`Running 3D ${workload.id} on ${adapter.name}...`);
      return measureWorkloadForLibrary3D(
        workload,
        adapter,
        refVolA,
        refVolB,
        targetDurationPerWorkloadMs,
      );
    });

    workloadSummaries.push({
      workload,
      referenceVolumeA: refVolA,
      referenceVolumeB: refVolB,
      results,
    });
  }

  const stressSummaries = runAdversarial3DStressComparison(onProgress);

  return {
    timestamp: new Date().toISOString(),
    workloadSummaries,
    stressSummaries,
  };
}
