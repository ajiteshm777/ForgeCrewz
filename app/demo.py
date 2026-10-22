"""Leaf instance segmentation demo (Mask R-CNN on PhenoBench).

Run from the repo root:  python app/demo.py
The model runs once per image. The sliders re-filter the saved predictions,
so they update instantly without re-running the model.
"""
import colorsys
import glob
import os
import sys
import time

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    from src.model import load_checkpoint_for_inference
except ImportError:
    from model import load_checkpoint_for_inference

# ---------------------------------------------------------------- model
CKPT = os.environ.get("CHECKPOINT", "models/epoch14.pth")
if not os.path.exists(CKPT):
    alt = os.path.join(ROOT, CKPT)
    if os.path.exists(alt):
        CKPT = alt
    else:  # e.g. on Hugging Face Spaces
        from huggingface_hub import hf_hub_download
        CKPT = hf_hub_download("ForgeCrewz/leaf-maskrcnn", "maskrcnn_leaf.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_m = load_checkpoint_for_inference(CKPT, DEVICE)
MODEL = (_m[0] if isinstance(_m, tuple) else _m).to(DEVICE).eval()

SCORE_FLOOR = 0.20   # lowest confidence kept; the slider filters above this
MAX_KEEP = 150       # cap on stored leaves (keeps memory small)
MAX_SIDE = 1024      # larger uploads are shrunk to this

MODEL.roi_heads.score_thresh = SCORE_FLOOR


def _palette(n=64):
    cols, h = [], 0.07
    for _ in range(n):
        h = (h + 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.62, 0.97)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return cols


PALETTE = _palette()


# ------------------------------------------------------------ inference
def run_model(img):
    """Run Mask R-CNN once and keep everything the sliders need."""
    if img is None:
        return None
    pil = Image.fromarray(img).convert("RGB")
    scale = MAX_SIDE / max(pil.size)
    if scale < 1:
        pil = pil.resize((int(pil.width * scale), int(pil.height * scale)))
    t0 = time.time()
    with torch.inference_mode():
        out = MODEL([to_tensor(pil).to(DEVICE)])[0]
    dt = time.time() - t0
    order = out["scores"].argsort(descending=True)[:MAX_KEEP]
    return {
        "rgb": np.array(pil),
        "scores": out["scores"][order].cpu().numpy(),
        "boxes": out["boxes"][order].cpu().numpy(),
        "probs": (out["masks"][order, 0].cpu().numpy() * 255).astype(np.uint8),
        "seconds": dt,
    }


# -------------------------------------------------------------- render
def stats_html(count=None, conf=None, cover=None, secs=None):
    def card(value, label):
        return f'<div class="stat"><div class="num">{value}</div><div class="lbl">{label}</div></div>'

    v = lambda x, f: "–" if x is None else f.format(x)
    return (
        '<div class="stats">'
        + card(v(count, "{}"), "leaves found")
        + card(v(conf, "{:.0%}"), "average confidence")
        + card(v(cover, "{:.0%}"), "of image covered")
        + card(v(secs, "{:.1f}s"), "model time")
        + "</div>"
    )


EMPTY_TABLE = pd.DataFrame(columns=["Leaf", "Confidence", "Area (px)", "Area (%)"])


def render(state, score_thr, mask_thr, opacity, show_outline, show_boxes, show_scores, min_area):
    if state is None:
        return None, stats_html(), EMPTY_TABLE
    rgb = state["rgb"]
    h, w = rgb.shape[:2]
    canvas = rgb.astype(np.float32)
    kept = []
    for k in np.where(state["scores"] >= score_thr)[0]:
        m = state["probs"][k] > int(mask_thr * 255)
        area = int(m.sum())
        if area < min_area:
            continue
        color = np.array(PALETTE[len(kept) % len(PALETTE)], dtype=np.float32)
        canvas[m] = canvas[m] * (1 - opacity) + color * opacity
        kept.append((k, m, area, tuple(int(c) for c in color)))

    out = canvas.clip(0, 255).astype(np.uint8)
    cover = np.zeros((h, w), bool)
    rows = []
    for i, (k, m, area, color) in enumerate(kept, 1):
        cover |= m
        if show_outline:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, color, 2)
        x1, y1, x2, y2 = [int(v) for v in state["boxes"][k]]
        if show_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if show_scores:
            cv2.putText(out, f"{state['scores'][k]:.2f}", (x1 + 3, max(y1 + 14, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append([i, round(float(state["scores"][k]), 3), area, round(100 * area / (h * w), 2)])

    table = pd.DataFrame(rows, columns=EMPTY_TABLE.columns) if rows else EMPTY_TABLE
    conf = float(np.mean([r[1] for r in rows])) if rows else None
    html = stats_html(len(rows), conf, float(cover.mean()), state["seconds"])
    return out, html, table


# ------------------------------------------------------------------ UI
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Bricolage Grotesque"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
)

CSS = """
.gradio-container { max-width: 1200px !important; margin: auto; background: #f1f5f0 !important; }
#hero { padding: 28px 4px 8px; }
#hero h1 { font-size: 2.6rem; line-height: 1.05; letter-spacing: -0.02em; color: #14301f; margin: 0 0 8px; font-weight: 800; }
#hero p { color: #41584a; font-size: 1.05rem; margin: 0 0 14px; max-width: 60ch; }
.swatches { display: flex; gap: 6px; }
.swatches span { width: 34px; height: 8px; border-radius: 4px; display: block; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 8px; }
.stat { background: #fff; border: 1px solid #d5e0d6; border-radius: 14px; padding: 14px 16px; }
.stat .num { font-size: 2.1rem; font-weight: 800; color: #14301f; line-height: 1; }
.stat .lbl { font-size: .85rem; color: #5b7263; margin-top: 6px; }
.stat:first-child { background: #14301f; border-color: #14301f; }
.stat:first-child .num { color: #b9f3c9; }
.stat:first-child .lbl { color: #9cc4a8; }
@media (max-width: 700px) { .stats { grid-template-columns: repeat(2, 1fr); } #hero h1 { font-size: 2rem; } }
footer { display: none !important; }
"""

GR6 = int(gr.__version__.split(".")[0]) >= 6
blocks_kw = {} if GR6 else {"theme": THEME, "css": CSS}

swatches = "".join(f'<span style="background:rgb{c}"></span>' for c in PALETTE[:8])
HERO = f"""
<div id="hero">
  <h1>Count every leaf in your field photo</h1>
  <p>Upload an image from a crop field. The model outlines each visible leaf as its own instance.
  Move the sliders to see how the thresholds change the result.</p>
  <div class="swatches">{swatches}</div>
</div>
"""

with gr.Blocks(title="Leaf segmentation", **blocks_kw) as demo:
    state = gr.State(None)
    gr.HTML(HERO)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            inp = gr.Image(label="Field photo", type="numpy", sources=["upload", "clipboard"], height=420)
            with gr.Group():
                score_thr = gr.Slider(0.2, 0.95, value=0.5, step=0.05, label="Confidence threshold",
                                      info="Higher keeps only the leaves the model is sure about.")
                mask_thr = gr.Slider(0.2, 0.9, value=0.5, step=0.05, label="Mask threshold",
                                     info="Higher gives tighter outlines.")
                min_area = gr.Slider(0, 2000, value=50, step=10, label="Minimum leaf size (pixels)",
                                     info="Hides tiny fragments.")
                opacity = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Fill opacity")
            with gr.Row():
                show_outline = gr.Checkbox(True, label="Outlines")
                show_boxes = gr.Checkbox(False, label="Boxes")
                show_scores = gr.Checkbox(False, label="Scores")

        with gr.Column(scale=7):
            stats = gr.HTML(stats_html())
            with gr.Tabs():
                with gr.Tab("Segmentation"):
                    result = gr.Image(label="Result", interactive=False, height=560)
                with gr.Tab("Leaf details"):
                    table = gr.Dataframe(value=EMPTY_TABLE, interactive=False, wrap=True)

    examples = sorted(glob.glob(os.path.join(ROOT, "examples", "*.[pj][np]g")))
    if examples:
        gr.Examples(examples=[[e] for e in examples], inputs=inp, label="Try an example")

    ctrl = [state, score_thr, mask_thr, opacity, show_outline, show_boxes, show_scores, min_area]
    outs = [result, stats, table]

    inp.change(run_model, inp, state, show_progress="minimal").then(render, ctrl, outs)
    for c in ctrl[1:]:
        c.change(render, ctrl, outs, show_progress="hidden")

if __name__ == "__main__":
    launch_kw = {"server_name": "0.0.0.0"}
    if GR6:
        launch_kw.update(theme=THEME, css=CSS)
    try:
        demo.launch(**launch_kw)
    except TypeError:
        demo.launch(server_name="0.0.0.0")