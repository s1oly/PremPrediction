"""
Build the visualisations from outputs/predictions.json:

  * title_race.gif  — a looping, Kalshi-style animation of the top-5 teams'
                      title probability growing gameweek by gameweek (for the README).
  * dashboard.html  — a self-contained interactive chart of every team's title
                      probability across the season (crosshair tooltip + table view).

Usage:
    python -m prem_prediction.viz
"""

import json

import numpy as np

from .config import season_label
from .paths import DASHBOARD_FILE, GIF_FILE, PREDICTIONS_FILE, ensure_dirs

# dataviz reference palette — categorical slots, light then dark.
LIGHT_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DARK_SLOTS = ["#3987e5", "#d95926", "#199e70", "#c98500",
              "#d55181", "#008300", "#9085e9", "#e66767"]
OTHERS_COLOR = "#898781"
N_COLORED = 8


# ---------------------------------------------------------------------------
# Data loading / shaping
# ---------------------------------------------------------------------------

def load_predictions() -> dict:
    with open(PREDICTIONS_FILE) as f:
        return json.load(f)


def shape(pred: dict):
    """
    Return (gameweeks, teams_ranked, series) where:
      gameweeks    — sorted list of int gameweeks
      teams_ranked — all teams sorted by their latest probability (desc)
      series       — {team: [prob_per_gameweek in %]}, 0.0 where a team is absent
    """
    gw_map = pred["gameweeks"]
    gameweeks = sorted(int(g) for g in gw_map)
    teams = set()
    for probs in gw_map.values():
        teams.update(probs.keys())

    series = {t: [] for t in teams}
    for gw in gameweeks:
        probs = gw_map[str(gw)]
        for t in teams:
            series[t].append(round(probs.get(t, 0.0) * 100, 3))

    latest = str(gameweeks[-1])
    teams_ranked = sorted(teams, key=lambda t: gw_map[latest].get(t, 0.0), reverse=True)
    return gameweeks, teams_ranked, series


def color_map(teams_ranked: list[str], slots: list[str]) -> dict[str, str]:
    """Assign the top-N teams fixed categorical slots; everyone else is grey."""
    cmap = {}
    for i, team in enumerate(teams_ranked):
        cmap[team] = slots[i] if i < N_COLORED else OTHERS_COLOR
    return cmap


# ---------------------------------------------------------------------------
# Animated GIF (README)
# ---------------------------------------------------------------------------

def build_gif(pred: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    pred = pred or load_predictions()
    gameweeks, teams_ranked, series = shape(pred)
    top = teams_ranked[:5]
    cmap = color_map(teams_ranked, LIGHT_SLOTS)

    # Smooth the reveal by interpolating between gameweeks.
    gxs = np.array(gameweeks, dtype=float)
    steps = 10 if len(gameweeks) <= 8 else 5
    if len(gameweeks) > 1:
        fine_x = np.linspace(gxs[0], gxs[-1], (len(gameweeks) - 1) * steps + 1)
    else:
        fine_x = gxs
    fine_y = {t: np.interp(fine_x, gxs, series[t]) for t in top}

    surface, page, ink, muted, grid = "#fcfcfb", "#f9f9f7", "#0b0b0b", "#898781", "#e1e0d9"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})

    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
    fig.patch.set_facecolor(page)
    ax.set_facecolor(surface)

    y_max = max(max(fine_y[t]) for t in top) * 1.18 + 2
    label = pred.get("season_label", "")

    def draw(frame):
        ax.clear()
        ax.set_facecolor(surface)
        n = frame + 1
        x = fine_x[:n]
        for t in top:
            ax.plot(x, fine_y[t][:n], color=cmap[t], linewidth=2.4, solid_capstyle="round")
        if len(x):
            xtip = x[-1]
            # De-overlap the tip labels: place from the top down with a min gap.
            tips = sorted(((fine_y[t][n - 1], t) for t in top), reverse=True)
            min_gap = y_max * 0.058
            placed = []
            for yv, t in tips:
                ly = yv if not placed else min(yv, placed[-1] - min_gap)
                placed.append(ly)
                ax.plot(xtip, yv, "o", color=cmap[t], markersize=6,
                        markeredgecolor=surface, markeredgewidth=1.5)
                ax.text(xtip + 0.2, ly, f" {t}  {yv:.0f}%",
                        color=cmap[t], fontsize=10, va="center", fontweight="bold")

        cur_gw = int(round(fine_x[min(frame, len(fine_x) - 1)]))
        ax.set_xlim(gxs[0] - 0.2, gxs[-1] + max(6, gxs[-1] * 0.28))
        ax.set_ylim(0, y_max)
        ax.set_xticks(gameweeks)
        ax.set_xlabel("Gameweek", color=muted, fontsize=10)
        ax.set_ylabel("Title probability", color=muted, fontsize=10)
        ax.set_title(f"Premier League {label} — Title Race",
                     color=ink, fontsize=15, fontweight="bold", loc="left", pad=14)
        ax.text(0.0, 1.015, f"Through Gameweek {cur_gw}", transform=ax.transAxes,
                color=muted, fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=muted)
        fig.tight_layout()

    total = len(fine_x)
    hold = 10  # pause on the final frame before looping
    frames = list(range(total)) + [total - 1] * hold

    anim = FuncAnimation(fig, draw, frames=frames, interval=110)
    ensure_dirs()
    anim.save(GIF_FILE, writer=PillowWriter(fps=9, metadata={"loop": 0}))
    plt.close(fig)
    print(f"GIF written to {GIF_FILE}")


# ---------------------------------------------------------------------------
# Interactive dashboard (self-contained HTML)
# ---------------------------------------------------------------------------

def build_dashboard(pred: dict | None = None) -> None:
    pred = pred or load_predictions()
    gameweeks, teams_ranked, series = shape(pred)
    light = color_map(teams_ranked, LIGHT_SLOTS)
    dark = color_map(teams_ranked, DARK_SLOTS)

    data = {
        "seasonLabel": pred.get("season_label", ""),
        "updated": pred.get("updated_utc", ""),
        "gameweeks": gameweeks,
        "teamsRanked": teams_ranked,
        "series": {t: series[t] for t in teams_ranked},
        "colorLight": light,
        "colorDark": dark,
        "nColored": N_COLORED,
        "othersColor": OTHERS_COLOR,
    }
    html = _DASHBOARD_TEMPLATE.replace("__DATA__", json.dumps(data))
    ensure_dirs()
    with open(DASHBOARD_FILE, "w") as f:
        f.write(html)
    print(f"Dashboard written to {DASHBOARD_FILE}")


def build_all() -> None:
    pred = load_predictions()
    build_gif(pred)
    build_dashboard(pred)


_DASHBOARD_TEMPLATE = r"""<title>Premier League Title Race</title>
<style>
  :root {
    color-scheme: light;
    --plane: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --sec: #52514e;
    --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --others: #898781;
    --ring: rgba(11,11,11,0.10); --good: #006300;
  }
  :root:not([data-theme="light"]) { }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --plane: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --sec: #c3c2b7;
      --muted: #898781; --grid: #2c2c2a; --axis: #383835; --others: #898781;
      --ring: rgba(255,255,255,0.10); --good: #0ca30c;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --plane: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --sec: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --others: #898781;
    --ring: rgba(255,255,255,0.10); --good: #0ca30c;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--plane); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 960px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 40px; }
  .eyebrow { font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
  h1 { font-size: clamp(24px, 5vw, 34px); margin: 6px 0 4px; text-wrap: balance; }
  .meta { color: var(--sec); font-size: 13px; }
  .tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 22px 0 26px; }
  @media (max-width: 560px) { .tiles { grid-template-columns: 1fr; } }
  .tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 14px 16px; }
  .tile .name { display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 14px; }
  .swatch { width: 10px; height: 10px; border-radius: 3px; flex: none; }
  .tile .val { font-size: 30px; font-weight: 700; margin-top: 6px; font-variant-numeric: tabular-nums; }
  .tile .delta { font-size: 13px; color: var(--sec); margin-top: 2px; font-variant-numeric: tabular-nums; }
  .card { background: var(--surface); border: 1px solid var(--ring); border-radius: 14px; padding: 16px 14px 10px; }
  .cardhead { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 6px; }
  .cardhead h2 { font-size: 15px; margin: 0; }
  .toggle { display: inline-flex; border: 1px solid var(--ring); border-radius: 999px; overflow: hidden; }
  .toggle button { appearance: none; border: 0; background: transparent; color: var(--sec); font: inherit;
    font-size: 12px; padding: 6px 12px; cursor: pointer; }
  .toggle button[aria-pressed="true"] { background: var(--ink); color: var(--surface); }
  .chartbox { position: relative; width: 100%; }
  svg { width: 100%; height: auto; display: block; touch-action: none; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 10px 2px 2px; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--sec);
    font-variant-numeric: tabular-nums; }
  .tt { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--ring);
    border-radius: 10px; padding: 8px 10px; font-size: 12px; box-shadow: 0 6px 20px rgba(0,0,0,.14);
    min-width: 150px; opacity: 0; transition: opacity .08s; z-index: 5; }
  .tt h4 { margin: 0 0 6px; font-size: 12px; color: var(--muted); font-weight: 600; }
  .tt .row { display: flex; align-items: center; gap: 7px; justify-content: space-between; margin: 2px 0; }
  .tt .row .l { display: flex; align-items: center; gap: 6px; }
  .tt .row .v { font-variant-numeric: tabular-nums; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: right; padding: 7px 8px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
  th:first-child, td:first-child { text-align: left; }
  thead th { color: var(--muted); font-weight: 600; font-size: 11px; letter-spacing: .04em; text-transform: uppercase; }
  .foot { color: var(--muted); font-size: 12px; margin-top: 22px; line-height: 1.55; }
  .hidden { display: none !important; }
</style>

<div class="wrap">
  <div class="eyebrow">Title probability market</div>
  <h1 id="title"></h1>
  <div class="meta" id="meta"></div>

  <div class="tiles" id="tiles"></div>

  <div class="card">
    <div class="cardhead">
      <h2>Win probability by gameweek</h2>
      <div class="toggle" role="group" aria-label="view">
        <button id="btnChart" aria-pressed="true">Chart</button>
        <button id="btnTable" aria-pressed="false">Table</button>
      </div>
    </div>
    <div class="chartbox" id="chartbox">
      <svg id="chart" viewBox="0 0 900 460" preserveAspectRatio="xMidYMid meet" role="img"></svg>
      <div class="tt" id="tt"></div>
    </div>
    <div class="legend" id="legend"></div>
    <div id="tableWrap" class="hidden" style="overflow-x:auto"></div>
  </div>

  <p class="foot" id="foot"></p>
</div>

<script>
const DATA = __DATA__;

const isDark = () => {
  const t = document.documentElement.getAttribute("data-theme");
  if (t === "dark") return true;
  if (t === "light") return false;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
};
const colors = () => isDark() ? DATA.colorDark : DATA.colorLight;
const cssvar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const GW = DATA.gameweeks, TEAMS = DATA.teamsRanked, S = DATA.series;
const lastIdx = GW.length - 1;
const pct = (v) => (v).toFixed(1) + "%";
let showAll = false;

function visibleTeams() {
  return showAll ? TEAMS : TEAMS.slice(0, DATA.nColored);
}

// ---- header + tiles ----
document.getElementById("title").textContent = "Premier League " + DATA.seasonLabel;
const upd = DATA.updated ? new Date(DATA.updated).toLocaleDateString(undefined,
  {year:"numeric", month:"short", day:"numeric"}) : "";
document.getElementById("meta").textContent =
  `After Gameweek ${GW[lastIdx]}` + (upd ? ` · Updated ${upd}` : "");

function renderTiles() {
  const c = colors();
  const box = document.getElementById("tiles");
  box.innerHTML = "";
  TEAMS.slice(0, 3).forEach(t => {
    const cur = S[t][lastIdx];
    const prev = lastIdx > 0 ? S[t][lastIdx - 1] : cur;
    const d = cur - prev;
    const arrow = d > 0.05 ? "▲" : d < -0.05 ? "▼" : "→";
    const dcol = d > 0.05 ? "var(--good)" : "var(--sec)";
    const el = document.createElement("div");
    el.className = "tile";
    el.innerHTML =
      `<div class="name"><span class="swatch" style="background:${c[t]}"></span>${t}</div>
       <div class="val">${pct(cur)}</div>
       <div class="delta" style="color:${dcol}">${arrow} ${Math.abs(d).toFixed(1)} pts since GW${GW[lastIdx-1] ?? GW[lastIdx]}</div>`;
    box.appendChild(el);
  });
}

// ---- chart ----
const VB = {w: 900, h: 460, l: 46, r: 128, t: 20, b: 40};
function sx(gw) {
  if (GW.length === 1) return VB.l;
  return VB.l + (gw - GW[0]) / (GW[lastIdx] - GW[0]) * (VB.w - VB.l - VB.r);
}
function yMax() {
  let m = 0;
  visibleTeams().forEach(t => S[t].forEach(v => { if (v > m) m = v; }));
  return Math.max(10, Math.ceil(m / 10) * 10 + 5);
}
function sy(v, ymax) { return VB.t + (1 - v / ymax) * (VB.h - VB.t - VB.b); }

function renderChart() {
  const c = colors(), ymax = yMax();
  const svg = document.getElementById("chart");
  const ink = cssvar("--ink"), muted = cssvar("--muted"), grid = cssvar("--grid"), axis = cssvar("--axis");
  let g = "";

  // y gridlines + labels
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = ymax * i / ticks, y = sy(v, ymax);
    g += `<line x1="${VB.l}" y1="${y}" x2="${VB.w - VB.r}" y2="${y}" stroke="${grid}" stroke-width="1"/>`;
    g += `<text x="${VB.l - 8}" y="${y + 4}" fill="${muted}" font-size="12" text-anchor="end">${v.toFixed(0)}%</text>`;
  }
  // x labels (thin out if many)
  const stepGW = GW.length > 14 ? Math.ceil(GW.length / 12) : 1;
  GW.forEach((gw, i) => {
    if (i % stepGW && i !== lastIdx) return;
    const x = sx(gw);
    g += `<text x="${x}" y="${VB.h - VB.b + 20}" fill="${muted}" font-size="12" text-anchor="middle">${gw}</text>`;
  });
  g += `<text x="${(VB.l + VB.w - VB.r) / 2}" y="${VB.h - 4}" fill="${muted}" font-size="12" text-anchor="middle">Gameweek</text>`;

  // lines
  const teams = visibleTeams();
  const others = TEAMS.slice(DATA.nColored);
  const drawTeam = (t, col, width) => {
    let d = "";
    S[t].forEach((v, i) => { d += (i ? "L" : "M") + sx(GW[i]).toFixed(1) + " " + sy(v, ymax).toFixed(1) + " "; });
    g += `<path d="${d}" fill="none" stroke="${col}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round"/>`;
  };
  if (!showAll) others.forEach(t => drawTeam(t, cssvar("--others"), 1)); // faint field
  teams.forEach(t => drawTeam(t, c[t], 2.2));

  // end labels for top 5 visible
  teams.slice(0, 5).forEach(t => {
    const v = S[t][lastIdx], x = sx(GW[lastIdx]), y = sy(v, ymax);
    g += `<circle cx="${x}" cy="${y}" r="3.5" fill="${c[t]}" stroke="${cssvar('--surface')}" stroke-width="1.5"/>`;
    g += `<text x="${x + 8}" y="${y + 4}" fill="${c[t]}" font-size="12" font-weight="700">${t} ${v.toFixed(0)}%</text>`;
  });

  // crosshair placeholder
  g += `<line id="cross" x1="0" y1="${VB.t}" x2="0" y2="${VB.h - VB.b}" stroke="${axis}" stroke-width="1" opacity="0"/>`;
  svg.innerHTML = g;
}

function renderLegend() {
  const c = colors();
  const box = document.getElementById("legend");
  box.innerHTML = "";
  const teams = visibleTeams();
  teams.forEach(t => {
    const s = document.createElement("span");
    s.innerHTML = `<span class="swatch" style="background:${c[t]}"></span>${t} ${pct(S[t][lastIdx])}`;
    box.appendChild(s);
  });
  if (!showAll && TEAMS.length > DATA.nColored) {
    const s = document.createElement("span");
    s.innerHTML = `<span class="swatch" style="background:${cssvar('--others')}"></span>Rest of field`;
    box.appendChild(s);
  }
}

// ---- tooltip / crosshair ----
const svg = document.getElementById("chart"), tt = document.getElementById("tt"), box = document.getElementById("chartbox");
function nearestGW(clientX) {
  const rect = svg.getBoundingClientRect();
  const vx = (clientX - rect.left) / rect.width * VB.w;
  let best = 0, bd = Infinity;
  GW.forEach((gw, i) => { const d = Math.abs(sx(gw) - vx); if (d < bd) { bd = d; best = i; } });
  return best;
}
function moveTip(clientX) {
  const i = nearestGW(clientX), c = colors();
  const cross = document.getElementById("cross");
  if (cross) { cross.setAttribute("x1", sx(GW[i])); cross.setAttribute("x2", sx(GW[i])); cross.setAttribute("opacity", "1"); }
  const rows = visibleTeams()
    .map(t => ({t, v: S[t][i]}))
    .filter(r => r.v >= 0.1)
    .sort((a, b) => b.v - a.v).slice(0, 8);
  tt.innerHTML = `<h4>Gameweek ${GW[i]}</h4>` + rows.map(r =>
    `<div class="row"><span class="l"><span class="swatch" style="background:${c[r.t]}"></span>${r.t}</span><span class="v">${pct(r.v)}</span></div>`).join("");
  const rect = svg.getBoundingClientRect();
  const px = (sx(GW[i]) / VB.w) * rect.width;
  let left = px + 16; if (left + 170 > rect.width) left = px - 170;
  tt.style.left = Math.max(4, left) + "px";
  tt.style.top = "12px";
  tt.style.opacity = "1";
}
svg.addEventListener("pointermove", e => moveTip(e.clientX));
svg.addEventListener("pointerleave", () => {
  tt.style.opacity = "0";
  const cross = document.getElementById("cross"); if (cross) cross.setAttribute("opacity", "0");
});

// ---- table ----
function renderTable() {
  const c = colors();
  const peak = t => Math.max(...S[t]);
  let h = `<table><thead><tr><th>Team</th><th>Now</th><th>Δ GW</th><th>Peak</th></tr></thead><tbody>`;
  TEAMS.forEach(t => {
    const cur = S[t][lastIdx], prev = lastIdx > 0 ? S[t][lastIdx - 1] : cur, d = cur - prev;
    h += `<tr><td><span class="swatch" style="display:inline-block;background:${c[t]};margin-right:7px"></span>${t}</td>
      <td>${pct(cur)}</td><td>${d >= 0 ? "+" : ""}${d.toFixed(1)}</td><td>${pct(peak(t))}</td></tr>`;
  });
  document.getElementById("tableWrap").innerHTML = h + "</tbody></table>";
}

// ---- view toggle ----
const btnC = document.getElementById("btnChart"), btnT = document.getElementById("btnTable");
function setView(table) {
  btnC.setAttribute("aria-pressed", String(!table));
  btnT.setAttribute("aria-pressed", String(table));
  document.getElementById("chartbox").classList.toggle("hidden", table);
  document.getElementById("legend").classList.toggle("hidden", table);
  document.getElementById("tableWrap").classList.toggle("hidden", !table);
}
btnC.onclick = () => setView(false);
btnT.onclick = () => setView(true);

document.getElementById("foot").innerHTML =
  "Probabilities from an LSTM trained on every Premier League season since 1993-94, " +
  "with a results-based Elo strength feature; the 20 teams' logits are passed through a " +
  "softmax each gameweek, so they always sum to 100%. Data: football-data.co.uk.";

function renderAll() { renderTiles(); renderChart(); renderLegend(); renderTable(); }
renderAll();
if (window.matchMedia) {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
}
new MutationObserver(renderAll).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
</script>
"""


if __name__ == "__main__":
    build_all()
