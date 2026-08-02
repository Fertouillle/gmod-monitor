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
COLORS = ["#e29443", "#5fb3e0", "#7fd47f", "#e05f7f", "#c99be0"]


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
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_heatmap_image(df, server_label):
    sub = df[df["server_label"] == server_label].dropna(subset=["players"])
    if sub.empty:
        return None, None

    pivot = sub.pivot_table(index="weekday", columns="hour", values="players", aggfunc="mean")
    pivot = pivot.reindex(index=range(7), columns=range(24))

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}h" for h in range(24)], fontsize=8)
    ax.set_yticks(range(7))
    ax.set_yticklabels(JOURS_FR)
    ax.set_title(f"Population moyenne — {server_label}")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Joueurs en moyenne")

    for i in range(7):
        for j in range(24):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6, color="black")

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


def get_timeseries(df, server_label, max_points=500):
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
<title>Aeron — Rapport population serveurs</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Exo+2:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --orange: #e29443;
    --bg: #0f0f10;
    --card: #1a1a1c;
    --card-border: #2a2a2d;
    --text: #eaeaea;
    --muted: #8a8a8f;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Exo 2', 'Segoe UI', Arial, sans-serif;
    margin: 0;
    padding: 32px 5vw 64px;
  }}
  header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--card-border);
    padding-bottom: 20px;
    margin-bottom: 36px;
  }}
  h1 {{
    font-family: 'Orbitron', sans-serif;
    color: var(--orange);
    letter-spacing: 1px;
    margin: 0;
    font-size: 26px;
  }}
  .meta {{ color: var(--muted); font-size: 13px; }}
  .server-block {{
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 28px;
    margin-bottom: 28px;
  }}
  .server-block h2 {{
    font-family: 'Orbitron', sans-serif;
    color: var(--orange);
    margin-top: 0;
    font-size: 18px;
    letter-spacing: 0.5px;
  }}
  .stat-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 14px;
    margin: 20px 0 28px;
  }}
  .stat-card {{
    background: #141415;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 14px 16px;
  }}
  .stat-card .label {{
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .stat-card .value {{
    font-family: 'Orbitron', sans-serif;
    font-size: 24px;
    color: var(--text);
    margin-top: 4px;
  }}
  .badge {{
    display: inline-block;
    font-size: 12px;
    padding: 2px 8px;
    border-radius: 4px;
    margin-top: 6px;
  }}
  .badge-up {{ background: rgba(127,212,127,0.15); color: #7fd47f; }}
  .badge-down {{ background: rgba(224,95,127,0.15); color: #e05f7f; }}
  .badge-neutral {{ background: rgba(138,138,143,0.15); color: var(--muted); }}
  canvas {{ max-width: 100%; }}
  .chart-wrap {{ margin-bottom: 28px; }}
  img.heatmap {{
    max-width: 100%;
    border-radius: 6px;
    background: #fff;
  }}
  table {{
    border-collapse: collapse;
    margin-top: 14px;
    width: 100%;
    font-size: 14px;
  }}
  th, td {{
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid var(--card-border);
  }}
  th {{ color: var(--orange); font-weight: 600; }}
  .no-data {{ color: var(--muted); font-style: italic; }}
  .section-title {{
    color: var(--muted);
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 24px 0 8px;
  }}
</style>
</head>
<body>
<header>
  <h1>AERON — Population des serveurs</h1>
  <div class="meta">Genere le {generated_at} · periode couverte : {data_range}</div>
</header>
"""]

    for label, img_b64, low_rows, stats, chart_labels, chart_values, color in sections:
        html_parts.append(f'<div class="server-block"><h2>{label}</h2>')

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
            html_parts.append(f'<div class="section-title">Evolution de la population</div>')
            html_parts.append(f'<div class="chart-wrap"><canvas id="{chart_id}" height="80"></canvas></div>')
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
                  backgroundColor: "{color}33",
                  fill: true,
                  tension: 0.25,
                  pointRadius: 0,
                  borderWidth: 2
                }}]
              }},
              options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                  x: {{ ticks: {{ color: "#8a8a8f", maxTicksLimit: 10 }}, grid: {{ color: "#2a2a2d" }} }},
                  y: {{ ticks: {{ color: "#8a8a8f" }}, grid: {{ color: "#2a2a2d" }}, beginAtZero: true }}
                }}
              }}
            }});
            </script>
            """)

        if img_b64:
            html_parts.append('<div class="section-title">Population moyenne par jour / heure</div>')
            html_parts.append(f'<img class="heatmap" src="data:image/png;base64,{img_b64}" alt="Heatmap {label}">')
            html_parts.append('<div class="section-title">Creneaux les moins actifs</div>')
            html_parts.append('<table><tr><th>Jour</th><th>Heure</th><th>Joueurs (moyenne)</th></tr>')
            for jour, heure, val in low_rows:
                html_parts.append(f"<tr><td>{jour}</td><td>{heure}</td><td>{val}</td></tr>")
            html_parts.append("</table>")

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