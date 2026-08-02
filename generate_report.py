#!/usr/bin/env python3
"""
Genere un rapport HTML unique (report.html) a partir de stats.csv.
Le fichier est autonome (heatmaps en base64) mais charge Chart.js via CDN
pour les graphiques interactifs -> necessite une connexion internet pour
afficher les courbes (fonctionne aussi en local, juste avec le CDN).

Necessite : pandas, matplotlib
    py -m pip install pandas matplotlib
"""

import os
import sys
import io
import json
import base64
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats.csv")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.html")

JOURS_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
COLORS = ["#2dd4bf", "#818cf8", "#f472b6", "#fbbf24", "#38bdf8"]


def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"Aucun fichier trouve : {CSV_PATH}")
        print("Lance d'abord monitor.py pendant quelques jours pour collecter des donnees.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["players"] = pd.to_numeric(df["players"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    return df


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_heatmap_image(df, server_label):
    sub = df[df["server_label"] == server_label].dropna(subset=["players"])
    if sub.empty:
        return None, None

    pivot = sub.pivot_table(index="weekday", columns="hour", values="players", aggfunc="mean")
    pivot = pivot.reindex(index=range(7), columns=range(24))

    plt.rcParams["font.family"] = "sans-serif"
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor("#11162a")
    ax.set_facecolor("#11162a")

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "aeron", ["#151b32", "#3730a3", "#2dd4bf"]
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}h" for h in range(24)], fontsize=8, color="#94a3b8")
    ax.set_yticks(range(7))
    ax.set_yticklabels(JOURS_FR, color="#94a3b8")
    ax.set_title(f"Population moyenne — {server_label}", color="#e5e7eb", fontsize=13, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Joueurs en moyenne", color="#94a3b8")
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")
    cbar.outline.set_visible(False)

    for i in range(7):
        for j in range(24):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6, color="#e5e7eb")

    plt.tight_layout()
    img_b64 = fig_to_base64(fig)
    return img_b64, pivot


def get_summary_stats(df, server_label):
    """Retourne un dict avec les stats cles (actuel, moyenne 24h, pic, creux, tendance 7j)."""
    sub = df[df["server_label"] == server_label].dropna(subset=["players"]).sort_values("timestamp")
    if sub.empty:
        return None

    now = sub["timestamp"].max()
    last_24h = sub[sub["timestamp"] >= now - timedelta(hours=24)]
    last_7d = sub[sub["timestamp"] >= now - timedelta(days=7)]

    current_row = sub.iloc[-1]
    current = int(current_row["players"]) if not pd.isna(current_row["players"]) else None
    current_max = int(current_row["max_players"]) if not pd.isna(current_row["max_players"]) else None

    avg_24h = last_24h["players"].mean() if not last_24h.empty else float("nan")
    peak_24h = last_24h["players"].max() if not last_24h.empty else float("nan")
    avg_7d = last_7d["players"].mean() if not last_7d.empty else float("nan")

    # tendance : moyenne 1ere moitie vs 2eme moitie des 7 derniers jours
    if len(last_7d) >= 4:
        mid = last_7d["timestamp"].min() + (last_7d["timestamp"].max() - last_7d["timestamp"].min()) / 2
        first_half = last_7d[last_7d["timestamp"] < mid]["players"].mean()
        second_half = last_7d[last_7d["timestamp"] >= mid]["players"].mean()
        if pd.isna(first_half) or first_half == 0:
            trend = None
        else:
            trend = (second_half - first_half) / first_half * 100
    else:
        trend = None

    return {
        "current": current,
        "current_max": current_max,
        "avg_24h": avg_24h,
        "peak_24h": peak_24h,
        "avg_7d": avg_7d,
        "trend": trend,
        "last_seen": now.strftime("%d/%m %H:%M"),
    }


def get_timeseries(df, server_label, max_points=200):
    """Retourne (labels, values) pour le graphique d'evolution, sous-echantillonne si besoin."""
    sub = df[df["server_label"] == server_label].dropna(subset=["players"]).sort_values("timestamp")
    if sub.empty:
        return [], []
    if len(sub) > max_points:
        step = len(sub) // max_points
        sub = sub.iloc[::step]
    labels = sub["timestamp"].dt.strftime("%d/%m %Hh%M").tolist()
    values = [None if pd.isna(v) else round(float(v), 1) for v in sub["players"]]
    return labels, values


def get_low_activity_rows(pivot, n=5):
    stacked = pivot.stack().sort_values().head(n)
    rows = []
    for (weekday, hour), val in stacked.items():
        rows.append((JOURS_FR[int(weekday)], f"{int(hour):02d}h", f"{val:.1f}"))
    return rows


def fmt_num(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.0f}"


def trend_badge(trend):
    if trend is None or np.isnan(trend):
        return '<span class="badge badge-neutral">stable</span>'
    if trend > 8:
        return f'<span class="badge badge-up">▲ +{trend:.0f}%</span>'
    if trend < -8:
        return f'<span class="badge badge-down">▼ {trend:.0f}%</span>'
    return '<span class="badge badge-neutral">≈ stable</span>'


def build_html(sections, generated_at, data_range):
    html_parts = [f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aeron — Observatoire de population</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg-0: #05070f;
    --bg-1: #0b0f1e;
    --glass: rgba(255,255,255,0.04);
    --glass-border: rgba(255,255,255,0.08);
    --teal: #2dd4bf;
    --indigo: #818cf8;
    --pink: #f472b6;
    --text: #e8eaf0;
    --muted: #8b93a7;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    background:
      radial-gradient(circle at 12% 8%, rgba(129,140,248,0.16), transparent 42%),
      radial-gradient(circle at 88% 18%, rgba(45,212,191,0.13), transparent 40%),
      var(--bg-0);
    background-attachment: scroll;
    color: var(--text);
    font-family: 'Inter', -apple-system, sans-serif;
    padding: 48px clamp(20px, 6vw, 96px) 80px;
    min-height: 100vh;
  }}
  header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 44px;
  }}
  .eyebrow {{
    font-size: 12px;
    letter-spacing: 3px;
    color: var(--teal);
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 6px;
  }}
  h1 {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: clamp(26px, 3.2vw, 38px);
    margin: 0;
    background: linear-gradient(90deg, #f5f7ff 0%, var(--teal) 120%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }}
  .meta {{ color: var(--muted); font-size: 13px; text-align: right; }}

  .server-block {{
    background: var(--glass);
    border: 1px solid var(--glass-border);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border-radius: 28px;
    padding: 32px clamp(20px, 3vw, 40px);
    margin-bottom: 32px;
    box-shadow: 0 20px 60px -30px rgba(0,0,0,0.6);
  }}
  .server-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
  }}
  .server-dot {{
    width: 10px; height: 10px;
    border-radius: 50%;
    box-shadow: 0 0 12px currentColor;
  }}
  .server-block h2 {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    margin: 0;
    font-size: 19px;
    letter-spacing: 0.3px;
  }}

  .stat-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px;
    margin-bottom: 32px;
  }}
  .stat-card {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 16px 18px;
  }}
  .stat-card .label {{
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }}
  .stat-card .value {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 26px;
    margin-top: 6px;
  }}
  .badge {{
    display: inline-block;
    font-size: 12px;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 100px;
    margin-top: 8px;
  }}
  .badge-up {{ background: rgba(45,212,191,0.14); color: var(--teal); }}
  .badge-down {{ background: rgba(244,114,182,0.14); color: var(--pink); }}
  .badge-neutral {{ background: rgba(139,147,167,0.14); color: var(--muted); }}

  .section-title {{
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 28px 0 12px;
    font-weight: 600;
  }}
  .chart-wrap {{
    background: rgba(255,255,255,0.02);
    border-radius: 20px;
    padding: 18px 20px 8px;
    border: 1px solid rgba(255,255,255,0.05);
  }}
  canvas {{ max-width: 100%; }}
  img.heatmap {{
    max-width: 100%;
    border-radius: 20px;
    display: block;
    border: 1px solid rgba(255,255,255,0.05);
  }}
  table {{
    border-collapse: separate;
    border-spacing: 0;
    margin-top: 14px;
    width: 100%;
    font-size: 14px;
    overflow: hidden;
    border-radius: 16px;
  }}
  th, td {{
    text-align: left;
    padding: 11px 16px;
  }}
  thead th {{
    color: var(--muted);
    font-weight: 500;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    background: rgba(255,255,255,0.03);
  }}
  tbody tr:nth-child(odd) {{ background: rgba(255,255,255,0.015); }}
  tbody td {{ color: var(--text); }}
  .no-data {{ color: var(--muted); font-style: italic; }}
</style>
</head>
<body>
<header>
  <div>
    <div class="eyebrow">Flotte Aeron · Telemetrie</div>
    <h1>Observatoire de population</h1>
  </div>
  <div class="meta">Genere le {generated_at}<br>Periode couverte : {data_range}</div>
</header>
"""]

    for label, img_b64, low_rows, stats, chart_labels, chart_values, color in sections:
        html_parts.append(
            f'<div class="server-block"><div class="server-header">'
            f'<span class="server-dot" style="color:{color}; background:{color};"></span>'
            f'<h2>{label}</h2></div>'
        )

        if stats is None:
            html_parts.append('<p class="no-data">Pas encore assez de donnees pour ce serveur.</p></div>')
            continue

        html_parts.append(f"""
        <div class="stat-cards">
          <div class="stat-card">
            <div class="label">Joueurs actuellement</div>
            <div class="value">{fmt_num(stats['current'])}/{fmt_num(stats['current_max'])}</div>
          </div>
          <div class="stat-card">
            <div class="label">Moyenne 24h</div>
            <div class="value">{fmt_num(stats['avg_24h'])}</div>
          </div>
          <div class="stat-card">
            <div class="label">Pic 24h</div>
            <div class="value">{fmt_num(stats['peak_24h'])}</div>
          </div>
          <div class="stat-card">
            <div class="label">Moyenne 7 jours</div>
            <div class="value">{fmt_num(stats['avg_7d'])}</div>
            {trend_badge(stats['trend'])}
          </div>
        </div>
        """)

        if chart_labels:
            chart_id = f"chart_{label}".replace(" ", "_").replace("-", "_")
            html_parts.append('<div class="section-title">Evolution de la population</div>')
            html_parts.append(f'<div class="chart-wrap"><canvas id="{chart_id}" height="70"></canvas></div>')
            html_parts.append(f"""
            <script>
            new Chart(document.getElementById("{chart_id}"), {{
              type: "line",
              data: {{
                labels: {json.dumps(chart_labels)},
                datasets: [{{
                  label: "Joueurs",
                  data: {json.dumps(chart_values)},
                  borderColor: "{color}",
                  backgroundColor: "{color}22",
                  fill: true,
                  tension: 0.3,
                  pointRadius: 0,
                  borderWidth: 2.5
                }}]
              }},
              options: {{
                responsive: true,
                animation: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                  x: {{ ticks: {{ color: "#8b93a7", maxTicksLimit: 8 }}, grid: {{ display: false }} }},
                  y: {{ ticks: {{ color: "#8b93a7" }}, grid: {{ color: "rgba(255,255,255,0.05)" }}, beginAtZero: true }}
                }}
              }}
            }});
            </script>
            """)

        if img_b64:
            html_parts.append('<div class="section-title">Population moyenne par jour / heure</div>')
            html_parts.append(f'<img class="heatmap" src="data:image/png;base64,{img_b64}" alt="Heatmap {label}">')
            html_parts.append('<div class="section-title">Creneaux les moins actifs</div>')
            html_parts.append('<table><thead><tr><th>Jour</th><th>Heure</th><th>Joueurs (moy.)</th></tr></thead><tbody>')
            for jour, heure, val in low_rows:
                html_parts.append(f"<tr><td>{jour}</td><td>{heure}</td><td>{val}</td></tr>")
            html_parts.append("</tbody></table>")

        html_parts.append("</div>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def main():
    df = load_data()
    labels = sorted(df["server_label"].unique())

    sections = []
    for i, label in enumerate(labels):
        img_b64, pivot = make_heatmap_image(df, label)
        low_rows = get_low_activity_rows(pivot) if pivot is not None else []
        stats = get_summary_stats(df, label)
        chart_labels, chart_values = get_timeseries(df, label)
        color = COLORS[i % len(COLORS)]
        sections.append((label, img_b64, low_rows, stats, chart_labels, chart_values, color))

    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")
    if not df.empty:
        data_range = f"{df['timestamp'].min().strftime('%d/%m/%Y')} au {df['timestamp'].max().strftime('%d/%m/%Y')}"
    else:
        data_range = "aucune donnee"

    html = build_html(sections, generated_at, data_range)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Rapport genere : {OUT_PATH}")
    print("Ouvre-le dans un navigateur (double-clic), ou publie-le via GitHub Pages.")


if __name__ == "__main__":
    main()
