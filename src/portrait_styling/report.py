"""Static HTML A/B comparison dashboard generator with Blind Mode & Expert Rating export.

Produces a single self-contained HTML file that can be opened locally in any browser
or sent directly to styling experts — no live Colab web server or tunnel required.
"""

from __future__ import annotations

import base64
import html
import json
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


def _image_to_inline_data_uri(image_path: str | Path, max_side: int = 720) -> str:
    path = Path(image_path)
    if not path.exists():
        return ""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _compute_model_summary(records: list[dict[str, Any]], model_aliases: list[str]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for alias in model_aliases:
        id_scores: list[float] = []
        vlm_scores: list[float] = []
        latencies: list[float] = []
        pass_count = 0
        total_id = 0
        ok_count = 0

        for rec in records:
            run = (rec.get("model_runs") or {}).get(alias) or {}
            best = run.get("best_candidate")
            if not best or best.get("error"):
                continue
            ok_count += 1
            if best.get("latency_sec") is not None:
                latencies.append(float(best["latency_sec"]))
            ev = best.get("eval_result") or {}
            if ev.get("id_similarity") is not None:
                id_scores.append(float(ev["id_similarity"]))
                total_id += 1
                if ev.get("id_passed") is True:
                    pass_count += 1
            if ev.get("vlm_score") is not None:
                vlm_scores.append(float(ev["vlm_score"]))

        summary.append(
            {
                "alias": alias,
                "completed": ok_count,
                "total": len(records),
                "avg_id": round(sum(id_scores) / len(id_scores), 4) if id_scores else None,
                "id_pass_rate": round(100.0 * pass_count / total_id, 1) if total_id else None,
                "avg_vlm": round(sum(vlm_scores) / len(vlm_scores), 2) if vlm_scores else None,
                "avg_latency": round(sum(latencies) / len(latencies), 2) if latencies else None,
            }
        )
    return summary


def generate_html_report(
    results_json_path: str | Path,
    output_html_path: str | Path,
    embed_images: bool = True,
    title: str = "Portrait Styling Lab — Model A/B Comparison",
) -> Path:
    """Build a self-contained static HTML dashboard from benchmark_results.json."""
    results_path = Path(results_json_path)
    out_path = Path(output_html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = payload.get("records", [])
    model_aliases: list[str] = payload.get("models", [])
    meta: dict[str, Any] = payload.get("meta", {})

    summary = _compute_model_summary(records, model_aliases)

    # Prepare frontend JSON payload (with inline JPEG thumbnails if embed_images=True)
    ui_records: list[dict[str, Any]] = []
    for rec in records:
        orig_path = rec.get("preprocessed_image") or rec.get("source_image") or ""
        orig_src = _image_to_inline_data_uri(orig_path) if embed_images else orig_path

        runs_ui: dict[str, Any] = {}
        for alias in model_aliases:
            run = (rec.get("model_runs") or {}).get(alias) or {}
            best = run.get("best_candidate")
            if best and not best.get("error"):
                img_src = (
                    _image_to_inline_data_uri(best["image_path"])
                    if embed_images
                    else best["image_path"]
                )
                runs_ui[alias] = {
                    "img": img_src,
                    "seed": best.get("seed"),
                    "latency_sec": best.get("latency_sec"),
                    "eval": best.get("eval_result") or {},
                    "error": None,
                }
            else:
                first_err = None
                for c in run.get("candidates") or []:
                    if c.get("error"):
                        first_err = c["error"]
                        break
                runs_ui[alias] = {
                    "img": "",
                    "seed": None,
                    "latency_sec": None,
                    "eval": {},
                    "error": first_err or "No output",
                }

        ui_records.append(
            {
                "sample_id": rec.get("sample_id"),
                "orig_img": orig_src,
                "diagnosis": rec.get("diagnosis") or {},
                "prompt": (rec.get("translated_prompt") or {}).get("prompt_zh", ""),
                "runs": runs_ui,
            }
        )

    data_blob = json.dumps(
        {
            "title": title,
            "meta": meta,
            "models": model_aliases,
            "summary": summary,
            "records": ui_records,
        },
        ensure_ascii=False,
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg: #0f1117;
    --card: #181b23;
    --border: #2a2f3d;
    --text: #e6e9ef;
    --muted: #9aa3b2;
    --accent: #6c8cff;
    --good: #22c55e;
    --warn: #f59e0b;
    --bad: #ef4444;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", sans-serif;
    line-height: 1.5;
  }}
  header {{
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    gap: 16px; padding-bottom: 18px; border-bottom: 1px solid var(--border); margin-bottom: 22px;
  }}
  h1 {{ margin: 0; font-size: 22px; }}
  .subtitle {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
  .controls {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  button {{
    background: #232837; color: var(--text); border: 1px solid var(--border);
    padding: 8px 14px; border-radius: 8px; cursor: pointer; font-size: 13px;
  }}
  button:hover {{ border-color: var(--accent); }}
  button.primary {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  button.active {{ background: #3b4b8a; border-color: var(--accent); }}
  .summary-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px; margin-bottom: 26px;
  }}
  .stat-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px;
  }}
  .stat-title {{ font-weight: 600; font-size: 15px; margin-bottom: 8px; }}
  .stat-row {{ display: flex; justify-content: space-between; font-size: 13px; color: var(--muted); margin: 3px 0; }}
  .stat-row strong {{ color: var(--text); }}
  .sample-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    padding: 18px; margin-bottom: 24px;
  }}
  .sample-head {{
    display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap;
    gap: 10px; margin-bottom: 10px;
  }}
  .sample-id {{ font-size: 16px; font-weight: 700; }}
  .diag-tags {{ display: flex; flex-wrap: wrap; gap: 6px; font-size: 12px; }}
  .tag {{ background: #242938; color: #c9d1e0; padding: 2px 8px; border-radius: 999px; }}
  .prompt-box {{
    background: #12151d; border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 12px; font-size: 12.5px; color: #cbd2df; margin-bottom: 14px;
  }}
  .compare-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px;
  }}
  .col {{
    background: #131620; border: 1px solid var(--border); border-radius: 10px;
    padding: 10px; display: flex; flex-direction: column; gap: 8px;
  }}
  .col.winner {{ border-color: var(--good); box-shadow: 0 0 0 1px var(--good); }}
  .col-head {{ display: flex; justify-content: space-between; align-items: center; font-size: 13px; font-weight: 600; }}
  .img-wrap {{
    width: 100%; aspect-ratio: 3 / 4; background: #0b0d12; border-radius: 8px;
    overflow: hidden; display: flex; align-items: center; justify-content: center;
  }}
  .img-wrap img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  .badges {{ display: flex; flex-wrap: wrap; gap: 6px; font-size: 11.5px; }}
  .badge {{ padding: 2px 7px; border-radius: 6px; background: #222736; color: #d5dce8; }}
  .badge.pass {{ background: rgba(34, 197, 94, 0.16); color: #86efac; }}
  .badge.fail {{ background: rgba(239, 68, 68, 0.18); color: #fca5a5; }}
  .vote-row {{ display: flex; gap: 6px; margin-top: auto; }}
  .vote-btn {{ flex: 1; padding: 6px; font-size: 12px; }}
  .error-box {{ color: var(--bad); font-size: 12px; padding: 12px; text-align: center; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>{html.escape(title)}</h1>
    <div class="subtitle" id="metaLine"></div>
  </div>
  <div class="controls">
    <button id="blindBtn" onclick="toggleBlind()">🕵️ 盲测模式 (Blind A/B): 关</button>
    <button class="primary" onclick="exportVotes()">⬇️ 导出专家评审 JSON</button>
  </div>
</header>

<section class="summary-grid" id="summaryGrid"></section>
<main id="samplesRoot"></main>

<script>
const DATA = {data_blob};
let blindMode = false;
const votes = {{}};
const blindOrder = {{}};

function deterministicShuffle(list, seedStr) {{
  const arr = [...list];
  let h = 2166136261;
  for (let i = 0; i < seedStr.length; i++) {{
    h ^= seedStr.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }}
  for (let i = arr.length - 1; i > 0; i--) {{
    h ^= h << 13; h ^= h >>> 17; h ^= h << 5;
    const j = Math.abs(h) % (i + 1);
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }}
  return arr;
}}

DATA.records.forEach(r => {{
  blindOrder[r.sample_id] = deterministicShuffle(DATA.models, r.sample_id);
}});

function toggleBlind() {{
  blindMode = !blindMode;
  const btn = document.getElementById("blindBtn");
  btn.textContent = blindMode ? "🕵️ 盲测模式 (Blind A/B): 开" : "🕵️ 盲测模式 (Blind A/B): 关";
  btn.classList.toggle("active", blindMode);
  render();
}}

function pickWinner(sampleId, modelAlias) {{
  votes[sampleId] = votes[sampleId] === modelAlias ? null : modelAlias;
  render();
}}

function exportVotes() {{
  const tally = {{}};
  DATA.models.forEach(m => tally[m] = 0);
  Object.values(votes).forEach(m => {{ if (m && tally[m] !== undefined) tally[m]++; }});
  const out = {{
    exported_at: new Date().toISOString(),
    blind_mode_active: blindMode,
    tally: tally,
    per_sample_winner: votes,
  }};
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: "application/json" }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "expert_blind_eval_votes.json";
  a.click();
}}

function renderSummary() {{
  const grid = document.getElementById("summaryGrid");
  if (blindMode) {{
    grid.innerHTML = `<div class="stat-card"><div class="stat-title">盲测模式已开启</div><div class="stat-row"><span>模型汇总指标已隐藏，避免先入为主。关闭盲测或导出 JSON 后可查看各模型得票。</span></div></div>`;
    return;
  }}
  grid.innerHTML = DATA.summary.map(s => `
    <div class="stat-card">
      <div class="stat-title">${{s.alias}}</div>
      <div class="stat-row"><span>完成样本</span><strong>${{s.completed}} / ${{s.total}}</strong></div>
      <div class="stat-row"><span>平均 ID 相似度</span><strong>${{s.avg_id ?? "N/A"}}</strong></div>
      <div class="stat-row"><span>ID 门禁通过率</span><strong>${{s.id_pass_rate !== null ? s.id_pass_rate + "%" : "N/A"}}</strong></div>
      <div class="stat-row"><span>平均 VLM 分</span><strong>${{s.avg_vlm ?? "N/A"}}</strong></div>
      <div class="stat-row"><span>平均耗时</span><strong>${{s.avg_latency !== null ? s.avg_latency + "s" : "N/A"}}</strong></div>
    </div>
  `).join("");
}}

function render() {{
  document.getElementById("metaLine").textContent =
    `样本数: ${{DATA.records.length}} · 模型数: ${{DATA.models.length}} · ID 阈值: ${{DATA.meta.id_threshold ?? 0.65}} · 生成时间: ${{DATA.meta.generated_at ?? ""}}`;
  renderSummary();

  const root = document.getElementById("samplesRoot");
  root.innerHTML = DATA.records.map(rec => {{
    const diag = rec.diagnosis || {{}};
    const tags = [diag.face_shape, diag.skin_tone, diag.style_goal].filter(Boolean);
    const orderedModels = blindMode ? blindOrder[rec.sample_id] : DATA.models;

    const modelCols = orderedModels.map((alias, idx) => {{
      const run = rec.runs[alias] || {{}};
      const ev = run.eval || {{}};
      const label = blindMode ? `方案 ${{String.fromCharCode(65 + idx)}}` : alias;
      const isWinner = votes[rec.sample_id] === alias;
      const idBadge = ev.id_similarity !== null && ev.id_similarity !== undefined
        ? `<span class="badge ${{ev.id_passed ? "pass" : "fail"}}">ID: ${{ev.id_similarity}}</span>`
        : `<span class="badge">ID: N/A</span>`;
      const vlmBadge = ev.vlm_score !== null && ev.vlm_score !== undefined
        ? `<span class="badge">VLM: ${{ev.vlm_score}}/10</span>`
        : "";
      const latBadge = run.latency_sec !== null && run.latency_sec !== undefined
        ? `<span class="badge">${{run.latency_sec}}s</span>`
        : "";

      return `
        <div class="col ${{isWinner ? "winner" : ""}}">
          <div class="col-head"><span>${{label}}</span>${{isWinner ? "<span>✅ 已选最佳</span>" : ""}}</div>
          <div class="img-wrap">
            ${{run.img ? `<img src="${{run.img}}" alt="${{label}}" loading="lazy" />` : `<div class="error-box">${{run.error || "生成失败"}}</div>`}}
          </div>
          <div class="badges">${{blindMode ? "" : idBadge + vlmBadge + latBadge}}</div>
          <div class="vote-row">
            <button class="vote-btn ${{isWinner ? "primary" : ""}}" onclick="pickWinner('${{rec.sample_id}}', '${{alias}}')">
              ${{isWinner ? "已选为本组最佳" : "选为最佳效果"}}
            </button>
          </div>
        </div>
      `;
    }}).join("");

    return `
      <article class="sample-card">
        <div class="sample-head">
          <div class="sample-id">Sample: ${{rec.sample_id}}</div>
          <div class="diag-tags">${{tags.map(t => `<span class="tag">${{t}}</span>`).join("")}}</div>
        </div>
        <div class="prompt-box"><strong>视觉翻译 Prompt：</strong>${{rec.prompt}}</div>
        <div class="compare-grid">
          <div class="col">
            <div class="col-head"><span>原图 (Aligned)</span></div>
            <div class="img-wrap">${{rec.orig_img ? `<img src="${{rec.orig_img}}" alt="original" loading="lazy" />` : `<div class="error-box">原图缺失</div>`}}</div>
            <div class="badges"><span class="badge">Reference Input</span></div>
          </div>
          ${{modelCols}}
        </div>
      </article>
    `;
  }}).join("");
}}

render();
</script>
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
