"""
live_dashboard.py
=================
Generate a lightweight web dashboard from benchmark CSV outputs.

Usage:
    python live_dashboard.py --results results --output dashboard.html
"""

import argparse
import csv
import json
import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler


def load_condition_csv(path):
    """Load per-step benchmark rows from one condition CSV."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_dashboard(results_dir: str, output_path: str) -> str:
    """Write a standalone HTML dashboard and return its path."""
    conditions = {}
    for name in ("fixed_timer", "ppo_only", "maestro_fl"):
        rows = load_condition_csv(os.path.join(results_dir, f"{name}.csv"))
        if rows:
            conditions[name] = rows

    payload = json.dumps(conditions)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MAESTRO-FL Dashboard</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #18202a; }}
    header {{ padding: 20px 28px; background: #18202a; color: white; }}
    main {{ padding: 24px 28px; display: grid; gap: 18px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card {{ background: white; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; }}
    .label {{ color: #5c6775; font-size: 13px; }}
    .value {{ font-size: 28px; font-weight: 700; margin-top: 6px; }}
    canvas {{ width: 100%; height: 320px; background: white; border: 1px solid #d9dee7; border-radius: 8px; }}
  </style>
</head>
<body>
  <header><h1>MAESTRO-FL Dashboard</h1></header>
  <main>
    <section class="metrics" id="metrics"></section>
    <canvas id="chart" width="1100" height="320"></canvas>
  </main>
  <script>
    const data = {payload};
    const labels = {{ fixed_timer: "Fixed Timer", ppo_only: "PPO Only", maestro_fl: "MAESTRO-FL" }};
    const colors = {{ fixed_timer: "#c43d4b", ppo_only: "#2f6fbb", maestro_fl: "#23865a" }};

    function numeric(row, key) {{ return Number(row[key] || 0); }}
    function avg(rows, key) {{
      if (!rows.length) return 0;
      return rows.reduce((sum, row) => sum + numeric(row, key), 0) / rows.length;
    }}
    function latest(rows, key) {{ return rows.length ? numeric(rows[rows.length - 1], key) : 0; }}

    const metrics = document.getElementById("metrics");
    Object.entries(data).forEach(([name, rows]) => {{
      const card = document.createElement("div");
      card.className = "card";
      card.innerHTML = `<div class="label">${{labels[name] || name}}</div>
        <div class="value">${{avg(rows, "waiting_time").toFixed(1)}}s</div>
        <div class="label">Avg waiting time, queue now ${{latest(rows, "queue_length").toFixed(0)}}</div>`;
      metrics.appendChild(card);
    }});

    const canvas = document.getElementById("chart");
    const ctx = canvas.getContext("2d");
    const series = Object.entries(data).map(([name, rows]) => [name, rows.map(row => numeric(row, "queue_length"))]);
    const maxLen = Math.max(1, ...series.map(([, values]) => values.length));
    const maxVal = Math.max(1, ...series.flatMap(([, values]) => values));
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = "14px Arial";
    ctx.fillText("Queue length over simulation steps", 18, 24);
    series.forEach(([name, values], idx) => {{
      ctx.strokeStyle = colors[name] || "#333";
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((value, i) => {{
        const x = 18 + (i / Math.max(1, maxLen - 1)) * (canvas.width - 42);
        const y = canvas.height - 24 - (value / maxVal) * (canvas.height - 62);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();
      ctx.fillStyle = colors[name] || "#333";
      ctx.fillText(labels[name] || name, 22 + idx * 150, canvas.height - 8);
    }});
  </script>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html)
    return output_path


def serve(directory: str, port: int) -> None:
    os.chdir(directory)
    server = ThreadingHTTPServer(("127.0.0.1", port), SimpleHTTPRequestHandler)
    print(f"[DASHBOARD] Serving at http://127.0.0.1:{port}/dashboard.html")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build MAESTRO-FL dashboard")
    parser.add_argument("--results", default="results", help="Results directory")
    parser.add_argument("--output", default="dashboard.html", help="HTML output path")
    parser.add_argument("--serve", action="store_true", help="Serve dashboard locally")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    args = parser.parse_args()

    output = build_dashboard(args.results, args.output)
    print(f"[DASHBOARD] Wrote {output}")
    if args.serve:
        serve(os.path.dirname(os.path.abspath(output)) or ".", args.port)
