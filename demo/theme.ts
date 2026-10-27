import {
  ChevronLeft,
  ChevronRight,
  createIcons,
  Gauge,
  type IconNode,
  Maximize,
  Moon,
  Pause,
  Play,
  RotateCcw,
  StepForward,
  Sun,
} from "lucide";

// Lucide 1.x dropped brand icons; this is the `github` glyph from lucide-static 0.469 (ISC).
const Github: IconNode = [
  [
    "path",
    {
      d: "M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4",
    },
  ],
  ["path", { d: "M9 18c-4.51 2-5-2-7-2" }],
];

const ICONS = {
  ChevronLeft,
  ChevronRight,
  Gauge,
  Github,
  Maximize,
  Moon,
  Pause,
  Play,
  RotateCcw,
  StepForward,
  Sun,
};

export function renderIcons(): void {
  createIcons({ icons: ICONS });
}

type RGBA = [number, number, number, number];

export interface ScenePalette {
  dark: boolean;
  bg: string;
  grid: string;
  bodyA: string;
  bodyB: string;
  /** Base surface color for shaded faces (0–1 RGB). */
  material: [number, number, number];
  /** Per-face colors when "color by face" is on. */
  faces: string[];
  edge: RGBA;
  hole: RGBA;
  normal: string;
  weighted: string;
  bezier: string;
  ok: string;
  bad: string;
  warn: string;
  muted: string;
}

// sage, sand, sky, clay, lavender, sea, straw, umber, mist, rose
const FACE_COLORS = [
  "#8fae89",
  "#d4b07a",
  "#8eabc8",
  "#c48f7c",
  "#a69bc0",
  "#7fa9a2",
  "#cdbd85",
  "#a98473",
  "#9eb2b8",
  "#c29a9e",
];

const LIGHT: ScenePalette = {
  dark: false,
  bg: "#ffffff",
  grid: "rgba(0, 0, 0, 0.07)",
  bodyA: "#3f6f9f",
  bodyB: "#c0643a",
  material: [0.86, 0.82, 0.76],
  faces: FACE_COLORS,
  edge: [0.1, 0.1, 0.1, 0.85],
  hole: [0.55, 0.35, 0.56, 1],
  normal: "#c39130",
  weighted: "#c39130",
  bezier: "#6f8a3c",
  ok: "#2f7d4f",
  bad: "#b8433a",
  warn: "#a8761f",
  muted: "#8a8a8a",
};

const DARK: ScenePalette = {
  dark: true,
  bg: "#000000",
  grid: "rgba(255, 255, 255, 0.08)",
  bodyA: "#86aed6",
  bodyB: "#e08b62",
  material: [0.78, 0.74, 0.68],
  faces: FACE_COLORS,
  edge: [0.96, 0.96, 0.96, 0.8],
  hole: [0.77, 0.59, 0.77, 1],
  normal: "#e0b357",
  weighted: "#e0b357",
  bezier: "#a7bf6c",
  ok: "#6fbf8b",
  bad: "#e07a6f",
  warn: "#d9a75a",
  muted: "#7a7a7a",
};

const STORAGE_KEY = "polystrata-theme";
const media = window.matchMedia("(prefers-color-scheme: dark)");
const listeners: Array<() => void> = [];

// Explicit user choice; null follows the OS setting.
let override: "light" | "dark" | null = (() => {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
})();

export function isDark(): boolean {
  return override ? override === "dark" : media.matches;
}

export function palette(): ScenePalette {
  return isDark() ? DARK : LIGHT;
}

/** Translucent version of a #rrggbb palette color. */
export function alpha(hex: string, a: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

function apply(): void {
  if (override) document.documentElement.dataset.theme = override;
  else delete document.documentElement.dataset.theme;
  const btn = document.getElementById("btn-theme");
  if (btn) {
    const dark = isDark();
    btn.innerHTML = `<i data-lucide="${dark ? "sun" : "moon"}"></i>`;
    btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
    btn.title = btn.getAttribute("aria-label")!;
    renderIcons();
  }
  for (const fn of listeners) fn();
}

export function onThemeChange(fn: () => void): void {
  listeners.push(fn);
}

/** Wire up the header theme toggle and Lucide icons. Call once after the DOM is ready. */
export function initTheme(): void {
  renderIcons();
  apply();
  media.addEventListener("change", apply);
  document.getElementById("btn-theme")?.addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    // Picking the system's own preference clears the override so it keeps following the OS.
    override = (next === "dark") === media.matches ? null : next;
    try {
      if (override) localStorage.setItem(STORAGE_KEY, override);
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Storage unavailable; the choice lasts for this page view.
    }
    apply();
  });
}
