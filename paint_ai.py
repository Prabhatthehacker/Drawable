"""Drawable - an AI that imagines a picture and then draws it, step by step, on its own canvas.

Type what you want and press "Create & draw". It makes the picture on your PC, then draws it
outlines first, colors second. When it finishes you get the finished drawing and the full drawing
video, each with a Download button. Nothing is saved anywhere until you press a Download button.
"""
import atexit
import base64
import io
import math
import multiprocessing
import os
import queue
import random
import re
import shutil
import sys
import tempfile
import threading
import time

# A windowed app has no console, but some libraries print progress to it.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
from PIL import Image

import tkinter as tk
from tkinter import filedialog, ttk

MODEL = "stabilityai/sd-turbo"
STYLES = {                                                      # the choices in the Style box (words added in front of your prompt)
    "Storybook cartoon": "colorful storybook illustration, clean smooth shapes, flat colors, simple clean outlines, ",
    "Coloring page": "black and white coloring page, clean thick line art, no shading, white background, ",
    "Watercolor": "soft watercolor painting, gentle washes of color, light paper texture, ",
    "Pencil sketch": "detailed pencil sketch, graphite shading, hand-drawn lines, ",
    "Anime": "anime style illustration, clean line art, vibrant colors, cel shading, ",
    "Pixel art": "pixel art, retro game style, limited palette, sharp pixels, ",
    "Oil painting": "oil painting, rich brush strokes, warm colors, ",
    "3D cartoon": "cute 3D cartoon render, soft lighting, smooth shapes, bright colors, ",
    "Realistic photo": "realistic photo, natural lighting, sharp focus, detailed, ",
}
DEFAULT_STYLE = "Storybook cartoon"
SURPRISE = "Surprise me"                                         # picks one of the styles above at random
STYLE_NAMES = list(STYLES) + [SURPRISE]
DETAIL = {"Fast": 360, "Normal": 520, "Detailed": 720}          # longest side of the canvas, in pixels
SPEED = {"Slow": 120, "Normal": 45, "Fast": 15, "Instant": 0}   # seconds the drawing takes
FPS = 24
HOLD_SECONDS = 1.5                                              # the finished picture stays on screen this long in the video
WHITE, NONE = 254, 255                                          # label values: paper left white / dark outline pixel

stop_flag = threading.Event()


class Stopped(Exception):
    pass


# ------------------------------------------------------------------ temporary files
# The drawing video is kept in the system temp folder (never in the app folder) until you
# download it, and it is deleted when the app closes.
_tmp = None


def temp_dir():
    global _tmp
    if _tmp is None or not os.path.isdir(_tmp):
        _tmp = tempfile.mkdtemp(prefix="drawable_")
    return _tmp


def cleanup_temp():
    global _tmp
    if _tmp:
        shutil.rmtree(_tmp, ignore_errors=True)
        _tmp = None


atexit.register(cleanup_temp)


# ------------------------------------------------------------------ the image model (runs on your PC)
class Heartbeat:
    """Keeps the status line moving during a long wait so the app does not look frozen."""

    def __init__(self, status, text):
        self.status, self.text = status, text
        self._stop = threading.Event()

    def __enter__(self):
        t0 = time.time()

        def beat():
            while not self._stop.wait(3):
                self.status(f"{self.text} - {int((time.time() - t0) // 60)} min so far. Keep this window open.")

        self.status(self.text + " ...")
        threading.Thread(target=beat, daemon=True).start()
        return self

    def __exit__(self, *a):
        self._stop.set()


pipe = None


def get_pipe(status):
    global pipe
    if pipe is None:
        import torch
        from diffusers import AutoPipelineForText2Image
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        kw = {"torch_dtype": torch.float16} if dev == "cuda" else {}
        status("Loading the image model (about a minute the first time)...")
        try:
            p = AutoPipelineForText2Image.from_pretrained(MODEL, local_files_only=True, **kw)
        except Exception:
            with Heartbeat(status, "Downloading the image model (about 5 GB, first time only)"):
                p = AutoPipelineForText2Image.from_pretrained(MODEL, **kw)
        try:
            p.set_progress_bar_config(disable=True)
        except Exception:
            pass
        pipe = p.to(dev)
    return pipe


_META = re.compile(r"palette|colou?r|instruction|quality|composition|requirement|final|sequence|order|rules?", re.I)
_VERBS = re.compile(r"^(?:\d+[.)]\s*)?(?:draw|create|add|include|make|show|place|put|position|give|use|keep|ensure|surround|"
                    r"maintain|finish|fill|follow|let|have)\s+(?:a|an|the|some|several|beautiful|breathtaking)?\s*", re.I)


def clean_prompt(text):
    """Turns a long, written-out brief into the short list of things to picture (the model only reads the start)."""
    parts, skip, title_done = [], False, False
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            head = line.lstrip("#").strip()
            skip = bool(_META.search(head))
            if not title_done and not parts and re.search(r"prompt|drawing", head, re.I):
                title_done = True
                skip = False
                continue
            continue
        if skip:
            continue
        line = re.sub(r"^[-*\u2022]+\s*|^\d+[.)]\s*", "", line)
        line = line.replace("**", "").strip()
        if re.search(r"\b(exactly|do not|don't|avoid|must|should)\b", line, re.I):
            continue
        line = _VERBS.sub("", line).strip(" .")
        if line:
            parts.append(line)
    out = ", ".join(parts) if parts else text
    out = re.sub(r"\b(breathtaking|beautiful|charming|adorable|cosy|magical|especially|naturally|important|really|very)\b,?\s*", "", out, flags=re.I)
    out = re.sub(r"[^,]*\bpart of the drawing\b[^,]*,?\s*", "", out, flags=re.I)
    return re.sub(r"\s+", " ", out).replace(" ,", ",").replace(":,", ":")


def wanted_colors(text):
    """If the prompt says 'exactly 25 colours', return 25."""
    m = re.search(r"exactly\s+(\d+)\s+colou?rs?", text, re.I)
    return max(6, min(100, int(m.group(1)))) if m else None


def encode_long(p, text, max_chunks=4):
    """Lets the model read more than 77 words-pieces by reading the prompt in several pieces."""
    import torch
    tok, enc = p.tokenizer, p.text_encoder
    ids = tok(text, add_special_tokens=False).input_ids
    chunks = [ids[i:i + 75] for i in range(0, len(ids), 75)][:max_chunks] or [[]]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    embs = []
    for ch in chunks:
        full = [tok.bos_token_id] + ch + [tok.eos_token_id] + [pad] * (75 - len(ch))
        with torch.no_grad():
            embs.append(enc(torch.tensor([full], device=enc.device))[0])
    return torch.cat(embs, dim=1).to(p.unet.dtype)


def generate_image(prompt, status, style=DEFAULT_STYLE):
    p = get_pipe(status)
    text = STYLES.get(style, STYLES[DEFAULT_STYLE]) + clean_prompt(prompt)
    args = dict(num_inference_steps=4, guidance_scale=0.0, height=512, width=512)

    def cb(_pipe, step, _t, kwargs):
        if stop_flag.is_set():
            raise Stopped()
        status(f"Imagining your picture... step {step + 1}/4 (a CPU can take a minute or two)")
        return kwargs

    status("Imagining your picture... (a CPU can take a minute or two)")
    try:
        emb = encode_long(p, text)
    except Exception:
        emb = None                                              # fall back to the normal short prompt
    try:
        if emb is not None:
            return p(prompt_embeds=emb, callback_on_step_end=cb, **args).images[0]
        return p(text, callback_on_step_end=cb, **args).images[0]
    except Stopped:
        raise
    except Exception:
        if emb is not None:                                     # long-prompt route failed: use the short one
            return p(text, **args).images[0]
        return p(text, **args).images[0]


def load_image(path):
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


# ------------------------------------------------------------------ combining and editing pictures
LAYOUTS = ["Side by side", "Top to bottom", "Grid", "Blend"]
FILTERS = ["None", "Black & white", "Sepia", "Warm", "Cool", "Vivid", "Soft"]
CHANGE = {"Low": 0.4, "Medium": 0.55, "High": 0.8}                # how much the AI may change the pictures
DEFAULT_EDITS = {"brightness": 0, "contrast": 100, "saturation": 100, "blur": 0, "sharpen": 0,
                 "filter": "None", "rotate": 0, "flip": False}


def edits_changed(e):
    return any(e.get(k) != v for k, v in DEFAULT_EDITS.items())


def _cover(im, w, h):
    """Resize to fill w x h, cutting off the extra edges."""
    s = max(w / im.width, h / im.height)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    x, y = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))


def _fit_max(im, max_side):
    s = max_side / max(im.size)
    return im if s >= 1 else im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)


def combine_images(imgs, layout="Side by side", max_side=1600):
    """Puts several pictures into one."""
    imgs = [i.convert("RGB") for i in imgs]
    n = len(imgs)
    if n == 1:
        return _fit_max(imgs[0], max_side)
    if layout == "Side by side":
        hh = int(min(900, max(256, sorted(i.height for i in imgs)[n // 2])))
        row = [i.resize((max(1, round(i.width * hh / i.height)), hh), Image.LANCZOS) for i in imgs]
        out = Image.new("RGB", (sum(i.width for i in row), hh), "white")
        x = 0
        for i in row:
            out.paste(i, (x, 0))
            x += i.width
    elif layout == "Top to bottom":
        ww = int(min(900, max(256, sorted(i.width for i in imgs)[n // 2])))
        col = [i.resize((ww, max(1, round(i.height * ww / i.width))), Image.LANCZOS) for i in imgs]
        out = Image.new("RGB", (ww, sum(i.height for i in col)), "white")
        y = 0
        for i in col:
            out.paste(i, (0, y))
            y += i.height
    elif layout == "Grid":
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        cw = 512
        ch = int(min(768, max(288, cw * imgs[0].height / imgs[0].width)))
        out = Image.new("RGB", (cols * cw, rows * ch), "white")
        for k, i in enumerate(imgs):
            out.paste(_cover(i, cw, ch), ((k % cols) * cw, (k // cols) * ch))
    else:                                                       # Blend: all pictures laid on top of each other
        first = _fit_max(imgs[0], max_side)
        acc = np.zeros((first.height, first.width, 3), np.float32)
        for i in imgs:
            acc += np.array(_cover(i, first.width, first.height), np.float32)
        out = Image.fromarray((acc / n).astype(np.uint8))
    return _fit_max(out, max_side)


def apply_edits(img, e):
    """Brightness, contrast, color, filters, blur, sharpen, flip and rotate."""
    e = {**DEFAULT_EDITS, **e}
    a = np.array(img.convert("RGB"), np.float32)
    if e["flip"]:
        a = a[:, ::-1]
    if e["rotate"]:
        a = np.rot90(a, k=-(int(e["rotate"]) // 90) % 4)
    a = np.ascontiguousarray(a)
    sat, con = e["saturation"] / 100.0, e["contrast"] / 100.0
    f = e["filter"]
    if f == "Vivid":
        sat, con = sat * 1.35, con * 1.1
    elif f == "Black & white":
        g = a @ np.array([0.299, 0.587, 0.114], np.float32)
        a = np.repeat(g[..., None], 3, axis=2)
    elif f == "Sepia":
        a = a @ np.array([[.393, .349, .272], [.769, .686, .534], [.189, .168, .131]], np.float32)
    elif f == "Warm":
        a[..., 0] *= 1.08
        a[..., 2] *= 0.92
    elif f == "Cool":
        a[..., 0] *= 0.92
        a[..., 2] *= 1.08
    elif f == "Soft":
        a = 0.5 * a + 0.5 * cv2.GaussianBlur(a, (0, 0), 3)
    if abs(sat - 1.0) > 0.01:
        hsv = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
        a = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    a = (a - 128.0) * con + 128.0 + e["brightness"] * 2.55
    if e["blur"] > 0:
        a = cv2.GaussianBlur(a, (0, 0), float(e["blur"]))
    if e["sharpen"] > 0:
        a = a + (a - cv2.GaussianBlur(a, (0, 0), 2.0)) * (e["sharpen"] * 0.5)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def restyle_image(img, prompt, change, status, style=DEFAULT_STYLE):
    """Lets the AI blend and change the picture(s) using your words (the same AI model, in image-to-image mode)."""
    from diffusers import AutoPipelineForImage2Image
    p = get_pipe(status)
    i2i = AutoPipelineForImage2Image.from_pipe(p)
    s = 512 / max(img.size)
    init = img.convert("RGB").resize((max(256, round(img.width * s / 8) * 8), max(256, round(img.height * s / 8) * 8)),
                                     Image.LANCZOS)
    text = STYLES.get(style, STYLES[DEFAULT_STYLE]) + clean_prompt(prompt)
    args = dict(image=init, strength=CHANGE.get(change, 0.55), num_inference_steps=4, guidance_scale=0.0)

    def cb(_pipe, step, _t, kwargs):
        if stop_flag.is_set():
            raise Stopped()
        status(f"Blending your pictures with the AI... step {step + 1} (a CPU can take a minute or two)")
        return kwargs

    status("Blending your pictures with the AI... (a CPU can take a minute or two)")
    try:
        emb = encode_long(p, text)
    except Exception:
        emb = None
    try:
        if emb is not None:
            return i2i(prompt_embeds=emb, callback_on_step_end=cb, **args).images[0]
        return i2i(text, callback_on_step_end=cb, **args).images[0]
    except Stopped:
        raise
    except Exception:
        return i2i(text, **args).images[0]


# ------------------------------------------------------------------ picture -> drawing plan
def runs_from_mask(mask):
    """Horizontal runs [y, x0, x1] covering every True pixel."""
    m = np.zeros((mask.shape[0], mask.shape[1] + 2), np.int8)
    m[:, 1:-1] = mask
    d = np.diff(m, axis=1)
    ys, xs = np.nonzero(d == 1)
    _, xe = np.nonzero(d == -1)
    if not len(ys):
        return np.zeros((0, 3), np.int32)
    return np.stack([ys, xs, xe - 1], 1).astype(np.int32)


def find_lines(rgb):
    """The picture's own dark outlines (cartoons, coloring pages) or, for painted pictures, only its strongest edges."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray < 100) & (rgb.max(axis=2) < 110)       # dark in every channel, so red or brown never count
    # Real outlines are thin. Big dark areas (shadows, dark leaves) are shading, not outlines: the color fill handles them.
    k = max(3, int(round(max(h, w) / 90)) | 1)
    blobs = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    dark = dark & ~(cv2.dilate(blobs, np.ones((3, 3), np.uint8)) > 0)
    frac = float(dark.mean())
    mode = "lines"
    if frac < 0.004 or frac > 0.2:
        mode = "edges"
        sm = cv2.pyrMeanShiftFiltering(cv2.bilateralFilter(rgb, 9, 40, 7), 8, 20)   # smooth away painted texture
        g = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY), (5, 5), 0)
        for hi in range(40, 400, 10):                   # raise the threshold until only the strongest ~5% remain
            dark = cv2.Canny(g, hi * 0.4, hi) > 0
            if dark.mean() <= 0.05:
                break
        dark[:4, :] = False
        dark[-4:, :] = False
        dark[:, :4] = False
        dark[:, -4:] = False
    num, cc, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
    keep = np.zeros(num, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= (max(5, (h * w) // 50000) if mode == "lines" else max(25, (h * w) // 11000))
    return keep[cc], mode


def thin(mask):
    """Zhang-Suen thinning: reduces thick lines to a one-pixel centre line."""
    img = np.pad(mask.astype(np.uint8), 1)
    for _ in range(80):
        changed = False
        for step in (0, 1):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            c = img[1:-1, 1:-1]
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            a = ((p2 == 0) & (p3 == 1)).astype(np.uint8) + ((p3 == 0) & (p4 == 1)) + ((p4 == 0) & (p5 == 1)) \
                + ((p5 == 0) & (p6 == 1)) + ((p6 == 0) & (p7 == 1)) + ((p7 == 0) & (p8 == 1)) \
                + ((p8 == 0) & (p9 == 1)) + ((p9 == 0) & (p2 == 1))
            if step == 0:
                cond = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            rm = (c == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1] > 0


def trace_strokes(line):
    """Turns the outline pixels into pen strokes: [(points, width, length), ...] in drawing order."""
    h, w = line.shape
    sk = thin(line)
    if not sk.any():
        return []
    dt = cv2.distanceTransform(line.astype(np.uint8), cv2.DIST_L2, 3)
    deg = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - 1
    skp = np.pad(sk, 1)
    vis = np.zeros_like(skp)
    ys, xs = np.nonzero(sk)
    ends = [(y, x) for y, x in zip(ys.tolist(), xs.tolist()) if deg[y, x] == 1]
    rest = [(y, x) for y, x in zip(ys.tolist(), xs.tolist()) if deg[y, x] != 1]

    def walk(y, x):
        pts = [(x, y)]
        vis[y + 1, x + 1] = True
        dx0 = dy0 = 0
        wsum, cnt = float(dt[y, x]), 1
        while True:
            best, bs = None, -9.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not dx and not dy:
                        continue
                    yy, xx = y + dy, x + dx
                    if skp[yy + 1, xx + 1] and not vis[yy + 1, xx + 1]:
                        sc = (dx * dx0 + dy * dy0) / math.hypot(dx, dy) - (0.05 if dx and dy else 0.0)
                        if sc > bs:
                            bs, best = sc, (yy, xx, dy, dx)
            if best is None:
                break
            y, x, dy0, dx0 = best
            vis[y + 1, x + 1] = True
            pts.append((x, y))
            wsum += float(dt[y, x])
            cnt += 1
        return pts, wsum / cnt

    raw = []
    for y, x in ends + rest:
        if not vis[y + 1, x + 1]:
            pts, mean_dt = walk(y, x)
            a = np.array(pts, np.float32).reshape(-1, 1, 2)
            if len(pts) > 2:
                a = cv2.approxPolyDP(a, 0.7, False)
            a = a.reshape(-1, 2).astype(np.float32)
            width = float(max(1.0, min(14.0, 2 * mean_dt - 1)))
            if len(a) >= 2 or width > 2:
                raw.append([a, width])
    # order the strokes so the pen travels as little as possible
    out, cur = [], np.array([0.0, 0.0], np.float32)
    starts = np.array([s[0][0] for s in raw])
    ends_ = np.array([s[0][-1] for s in raw])
    alive = np.ones(len(raw), bool)
    for _ in range(len(raw)):
        d1 = np.where(alive, np.linalg.norm(starts - cur, axis=1), np.inf)
        d2 = np.where(alive, np.linalg.norm(ends_ - cur, axis=1), np.inf)
        i1, i2 = int(d1.argmin()), int(d2.argmin())
        if d1[i1] <= d2[i2]:
            i, pts = i1, raw[i1][0]
        else:
            i, pts = i2, raw[i2][0][::-1].copy()
        alive[i] = False
        length = float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) if len(pts) > 1 else 0.0
        out.append((pts, raw[i][1], length))
        cur = pts[-1]
    return out


def to_lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def build_plan(img, max_side, colors, outlines, fill, keep_bg):
    """Works out the whole drawing: the outline strokes, the palette, and every region to color."""
    w0, h0 = img.size
    s = max_side / max(w0, h0)
    w, h = max(8, int(w0 * s) // 2 * 2), max(8, int(h0 * s) // 2 * 2)       # even sizes: the video needs them
    rgb = np.array(img.convert("RGB").resize((w, h), Image.LANCZOS))

    line, mode = find_lines(rgb)
    lc = rgb[line].mean(axis=0) if (line.any() and mode == "lines") else (28, 28, 28)
    if sum(lc) > 270:
        lc = (28, 28, 28)
    line_color = tuple(int(v) for v in lc)
    plan = {"w": w, "h": h, "mode": mode, "line": line, "line_color": line_color,
            "strokes": [], "stroke_len": 0.0, "comps": [], "palette": [], "fill_px": 0}

    if outlines:
        plan["strokes"] = trace_strokes(line)
        plan["stroke_len"] = float(sum(s[2] for s in plan["strokes"]))

    if fill:
        wide = cv2.dilate(line.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if (~wide).sum() < 200:
            wide[:] = False
        lab = to_lab(rgb)
        pts = lab[~wide]
        pts = np.ascontiguousarray(pts[::max(1, len(pts) // (12000 if colors <= 32 else 24000))], np.float32)
        k = int(max(1, min(colors, len(pts))))
        cv2.setRNGSeed(12345)
        _, _, cents = cv2.kmeans(pts, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5),
                                 3, cv2.KMEANS_PP_CENTERS)
        cents = np.array(cents, np.float32)
        thr = 7.0 if colors <= 32 else (4.0 if colors <= 64 else 2.5)   # with many colors, keep finer differences
        while len(cents) > 1:                                   # merge colors that look the same
            dist = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(dist, np.inf)
            i, j = np.unravel_index(int(dist.argmin()), dist.shape)
            if dist[i, j] >= thr:
                break
            cents[i] = (cents[i] + cents[j]) / 2
            cents = np.delete(cents, j, axis=0)
        k = len(cents)
        labels = np.empty((h, w), np.uint8)
        for y0 in range(0, h, 64):
            blk = lab[y0:y0 + 64]
            labels[y0:y0 + 64] = ((blk[:, :, None, :] - cents[None, None, :, :]) ** 2).sum(-1).argmin(-1)
        if wide.any():                                          # pixels on / beside the lines take the nearest region color
            _, lbl = cv2.distanceTransformWithLabels(wide.astype(np.uint8), cv2.DIST_L2, 3,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
            free = ~wide
            table = np.zeros(int(lbl.max()) + 1, np.uint8)
            table[lbl[free]] = labels[free]
            labels[wide] = table[lbl[wide]]
        for _ in range(2):                                      # remove lonely specks (one color at a time, so 100 colors stay light on memory)
            own = np.zeros((h, w), np.float32)
            best = np.full((h, w), -1.0, np.float32)
            arg = np.zeros((h, w), np.uint8)
            for c in range(k):
                sc = cv2.blur((labels == c).astype(np.float32), (3, 3))
                own = np.where(labels == c, sc, own)
                better = sc > best
                best = np.where(better, sc, best)
                arg = np.where(better, np.uint8(c), arg)
            labels = np.where(own * 9 <= 2.01, arg, labels)
        pal = []
        for c in range(k):
            m = labels == c
            pal.append(tuple(int(v) for v in rgb[m].mean(axis=0)) if m.any() else (255, 255, 255))
        skip = np.zeros(256, bool)                              # near-white colors stay as white paper
        for c in range(k):
            skip[c] = cents[c][0] > 93 and math.hypot(cents[c][1], cents[c][2]) < 8
        labels = np.where(skip[labels], WHITE, labels).astype(np.uint8)
        if not keep_bg:                                         # leave the background white
            border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
            border = border[border < k]
            if len(border):
                bl = int(np.bincount(border, minlength=k).argmax())
                _, cc = cv2.connectedComponents((labels == bl).astype(np.uint8), connectivity=4)
                edge = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
                labels[np.isin(cc, edge[edge > 0])] = WHITE
        if outlines:
            labels[line] = NONE                                 # colors never cover the outlines
        area = np.bincount(labels[labels < k], minlength=k)
        order = [c for c in np.argsort(-area) if area[c] > 0]   # biggest colors first
        comps = []
        for rank, c in enumerate(order):
            num, cc, stats, _ = cv2.connectedComponentsWithStats((labels == c).astype(np.uint8), connectivity=4)
            for i in range(1, num):
                x, y, bw, bh, px = stats[i]
                r = runs_from_mask(cc[y:y + bh, x:x + bw] == i)
                r[:, 0] += y
                r[:, 1] += x
                r[:, 2] += x
                comps.append({"rank": rank, "color": pal[c], "px": int(px), "runs": r})
        plan["comps"] = comps
        plan["palette"] = [pal[c] for c in order]
        plan["fill_px"] = int(sum(c["px"] for c in comps))
    return plan


# ------------------------------------------------------------------ the canvas and the pen
_PEN = [([(0, 0), (3, -9), (9, -3)], (240, 207, 155)),
        ([(3, -9), (9, -3), (25, -19), (19, -25)], None),
        ([(19, -25), (25, -19), (28, -22), (22, -28)], (138, 138, 138)),
        ([(0, 0), (1.6, -4.8), (4.8, -1.6)], (34, 34, 34))]


def draw_pen(frame, x, y, color):
    k = max(1.0, frame.shape[1] / 420.0)
    for pts, fill in _PEN:
        poly = np.array([[round(x + px * k), round(y + py * k)] for px, py in pts], np.int32)
        cv2.fillPoly(frame, [poly], fill or color, cv2.LINE_AA)
        cv2.polylines(frame, [poly], True, (40, 40, 40), 1, cv2.LINE_AA)


class OutlinePhase:
    name = "outline"

    def __init__(self, plan, paper):
        self.S, self.paper, self.line = plan["strokes"], paper, plan["line"]
        self.color = tuple(int(c) for c in plan["line_color"])
        self.pen_color = self.color
        self.total, self.done = max(1.0, plan["stroke_len"]), 0.0
        self.si = self.pi = 0
        self.x, self.y = (float(self.S[0][0][0][0]), float(self.S[0][0][0][1])) if self.S else (0.0, 0.0)
        self.finished = not self.S

    def advance(self, b):
        S = self.S
        while b > 0 and self.si < len(S):
            pts, width, _ = S[self.si]
            if self.pi == 0:
                self.x, self.y = float(pts[0][0]), float(pts[0][1])
                self.pi = 1
            if self.pi >= len(pts):
                self.si += 1
                self.pi = 0
                continue
            tx, ty = float(pts[self.pi][0]), float(pts[self.pi][1])
            dx, dy = tx - self.x, ty - self.y
            d = math.hypot(dx, dy)
            if d <= b:
                nx, ny = tx, ty
                b -= d
                self.done += d
                self.pi += 1
            else:
                f = b / d
                nx, ny = self.x + dx * f, self.y + dy * f
                self.done += b
                b = 0
            cv2.line(self.paper, (round(self.x), round(self.y)), (round(nx), round(ny)),
                     self.color, max(1, round(width)), cv2.LINE_AA)
            self.x, self.y = nx, ny
        self.finished = self.si >= len(S)

    def leave(self):                                            # make the outline match the picture exactly
        self.paper[self.line] = self.color

    def label(self):
        return f"Drawing outlines... {min(100, int(100 * self.done / self.total))}%"

    def swatch(self):
        return 0


class FillPhase:
    name = "fill"

    def __init__(self, plan, paper, offset):
        self.C, self.paper, self.offset = plan["comps"], paper, offset
        self.n_colors = len(plan["palette"])
        self.total, self.done = max(1.0, float(plan["fill_px"])), 0.0
        self.ci = self.ri = self.off = 0
        self.x = self.y = 0.0
        self.pen_color = self.C[0]["color"] if self.C else (0, 0, 0)
        self.finished = not self.C

    def advance(self, b):
        C = self.C
        while b > 0 and self.ci < len(C):
            comp = C[self.ci]
            R = comp["runs"]
            if self.ri >= len(R):
                self.ci += 1
                self.ri = self.off = 0
                continue
            col = comp["color"]
            self.pen_color = col
            y, xa, xb = int(R[self.ri, 0]), int(R[self.ri, 1]) + self.off, int(R[self.ri, 2])
            ln = xb - xa + 1
            if ln <= b:
                self.paper[y, xa:xb + 1] = col
                b -= ln
                self.done += ln
                self.ri += 1
                self.off = 0
                self.x, self.y = xb + 1, y + 0.5
            else:
                n = int(b)
                if n <= 0:
                    break
                self.paper[y, xa:xa + n] = col
                self.off += n
                self.done += n
                b -= n
                self.x, self.y = xa + n, y + 0.5
        self.finished = self.ci >= len(C)

    def leave(self):
        pass

    def _rank(self):
        return self.C[self.ci]["rank"] if self.ci < len(self.C) else self.n_colors - 1

    def label(self):
        return f"Coloring... color {self._rank() + 1} of {self.n_colors}"

    def swatch(self):
        return self._rank() + self.offset


def _fourcc(code):
    f = getattr(cv2.VideoWriter, "fourcc", None) or cv2.VideoWriter_fourcc
    return f(*code)


class Drawing:
    """Draws the plan step by step on a canvas and records the video while it does."""

    def __init__(self, plan, seconds, record):
        self.w, self.h = plan["w"], plan["h"]
        self.paper = np.full((self.h, self.w, 3), 255, np.uint8)
        self.has_outline = bool(plan["strokes"])
        self.phases = []
        if self.has_outline:
            self.phases.append(OutlinePhase(plan, self.paper))
        if plan["comps"]:
            self.phases.append(FillPhase(plan, self.paper, 1 if self.has_outline else 0))
        self.instant = seconds <= 0
        self.total_dur = max(float(seconds), 0.001)
        for i, ph in enumerate(self.phases):
            share = 1.0 if len(self.phases) == 1 else (0.4 if i == 0 else 0.6)
            ph.dur = max(0.001, seconds * share)
        self.t = self.phase_t0 = 0.0
        self.pi = 0
        self.speed = 1.0                                        # "Skip to end" raises this
        self.finished = False
        self.progress, self.label, self.swatch = 0.0, "", 0
        self.writer = self.video_path = None
        self.video_failed = False
        if record and not self.instant and self.phases:
            self._open_writer()

    def _open_writer(self):
        for ext, code in (("mp4", "mp4v"), ("avi", "MJPG")):
            path = os.path.join(temp_dir(), "drawing." + ext)
            vw = cv2.VideoWriter(path, _fourcc(code), FPS, (self.w, self.h))
            if vw.isOpened():
                self.writer, self.video_path = vw, path
                return
            vw.release()
        self.video_failed = True

    def _write(self, frame):
        if self.writer is not None:
            self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def _close_video(self):
        if self.writer is not None:
            final = self.paper.copy()
            for _ in range(int(HOLD_SECONDS * FPS)):
                self._write(final)
            self.writer.release()
            self.writer = None

    def render(self, ph=None):
        frame = self.paper.copy()
        if ph is not None:
            draw_pen(frame, ph.x, ph.y, ph.pen_color)
        return frame

    def step(self):
        """Advance one video frame and return it."""
        if self.finished:
            return self.render()
        if self.instant or not self.phases:
            for ph in self.phases:
                ph.advance(math.inf)
                ph.leave()
            self.finished, self.progress, self.label = True, 1.0, "Done"
            return self.render()
        ph = self.phases[self.pi]
        f = min(1.0, (self.t - self.phase_t0) / ph.dur)
        ph.advance(math.inf if f >= 1.0 else max(0.0, f * ph.total - ph.done))
        before = sum(p.dur for p in self.phases[:self.pi])
        self.progress = min(1.0, (before + f * ph.dur) / self.total_dur)
        self.label, self.swatch = ph.label(), ph.swatch()
        pen = ph
        if f >= 1.0:
            ph.leave()
            self.phase_t0 += ph.dur
            self.pi += 1
            if self.pi >= len(self.phases):
                self.finished, self.progress, pen = True, 1.0, None
        frame = self.render(pen)
        self._write(frame)
        self.t += self.speed / FPS
        if self.finished:
            self._close_video()
        return frame

    def finalize(self):
        """Stop here (also used when you press Stop): keep what has been drawn."""
        self.finished = True
        self._close_video()

    def image(self):
        return Image.fromarray(self.paper.copy())


# ------------------------------------------------------------------ the window
def mixhex(color, amount, base):
    """Fade an (r, g, b) color toward a base hex color (amount 0 = same, 1 = base)."""
    b = tuple(int(base[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(c * (1 - amount) + bc * amount) for c, bc in zip(color, b))


# the two looks: light (sun) and dark (moon)
THEMES = {
    "light": dict(BG="#f5f3ee", HEAD="#1f2937", CARD="#ffffff", INPUT="#faf8f3", BORDER="#e3ded3", TEXT="#1f2937",
                  MUTED="#78716c", ACCENT="#e8590c", ACCENT_DARK="#c94a08", ACCENT_OFF="#f1b79a", HOVER="#f1ede4",
                  PAPER="#fbfaf7", TROUGH="#e8e4da", HINT="#a8a29e", DISABLED="#b5afa3", HEADMUTED="#cbd5e1"),
    "dark": dict(BG="#14171c", HEAD="#0b0e12", CARD="#1d2229", INPUT="#161a20", BORDER="#2c333c", TEXT="#f1efe9",
                 MUTED="#a8a196", ACCENT="#ff7a33", ACCENT_DARK="#ff9257", ACCENT_OFF="#6b3a1f", HOVER="#262d36",
                 PAPER="#0f1216", TROUGH="#2c333c", HINT="#6b645a", DISABLED="#5b6470", HEADMUTED="#9aa4b2"),
}
FONT = "Segoe UI"

# which colors each kind of widget uses (T is the current look)
ROLES = {
    "bg": lambda T: dict(bg=T["BG"]),
    "head": lambda T: dict(bg=T["HEAD"]),
    "headname": lambda T: dict(bg=T["HEAD"], fg="#ffffff"),
    "headtag": lambda T: dict(bg=T["HEAD"], fg=T["HEADMUTED"]),
    "card": lambda T: dict(bg=T["CARD"], highlightbackground=T["BORDER"]),
    "cardin": lambda T: dict(bg=T["CARD"]),
    "label": lambda T: dict(bg=T["CARD"], fg=T["TEXT"]),
    "muted": lambda T: dict(bg=T["CARD"], fg=T["MUTED"]),
    "bglabel": lambda T: dict(bg=T["BG"], fg=T["TEXT"]),
    "text": lambda T: dict(bg=T["INPUT"], fg=T["TEXT"], insertbackground=T["TEXT"],
                           highlightbackground=T["BORDER"], highlightcolor=T["ACCENT"]),
    "check": lambda T: dict(bg=T["CARD"], fg=T["TEXT"], activebackground=T["CARD"], activeforeground=T["TEXT"],
                            selectcolor=T["INPUT"]),
    "scale": lambda T: dict(bg=T["CARD"], fg=T["TEXT"], troughcolor=T["TROUGH"], activebackground=T["ACCENT"]),
    "btn": lambda T: dict(bg=T["CARD"], fg=T["TEXT"], activebackground=T["HOVER"], activeforeground=T["TEXT"],
                          disabledforeground=T["DISABLED"], highlightbackground=T["BORDER"]),
    "canvas": lambda T: dict(bg=T["PAPER"]),
}


class App:
    CW, CH = 600, 380

    def __init__(self, root):
        from PIL import ImageTk
        self.ImageTk = ImageTk
        self.root = root
        root.title("Drawable")
        try:                                                     # the Drawable logo in the window corner
            root.iconbitmap(os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))), "Drawable.ico"))
        except Exception:
            pass
        w = min(1100, max(800, root.winfo_screenwidth() - 40))
        h = min(720, max(560, root.winfo_screenheight() - 90))
        root.geometry(f"{w}x{h}")
        root.minsize(min(w, 960), min(h, 600))
        self.q = queue.Queue()
        self.drawing = None
        self.final_img = None
        self.source_img = None
        self.source_kind = "imagined"
        self.surprise_style = None
        self.video_path = None
        self.photo = None
        self.img_item = None
        self.pics = []
        self.theme = "light"
        self.T = THEMES["light"]
        self.reg = []                                            # every widget with its look, so the theme can change
        self.combos = []
        self.palette_colors = []
        self.shown_swatch = -1
        self._build()
        self.apply_theme()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.report_callback_exception = self.on_ui_error        # a mistake inside a button never leaves the app stuck
        root.after(80, self.poll)

    # ---- look
    def _t(self, widget, role):
        self.reg.append((widget, role))
        return widget

    def apply_theme(self):
        T = self.T
        for w, role in self.reg:
            try:
                if role == "primary":
                    ok = str(w["state"]) == "normal"
                    w.config(bg=T["ACCENT"] if ok else T["ACCENT_OFF"], fg="#ffffff", activebackground=T["ACCENT_DARK"],
                             activeforeground="#ffffff", disabledforeground="#ffffff")
                else:
                    w.config(**ROLES[role](T))
            except Exception:
                pass
        try:
            self.root.configure(bg=T["BG"])
            st = ttk.Style()
            st.configure("TCombobox", fieldbackground=T["INPUT"], background=T["CARD"], foreground=T["TEXT"],
                         arrowcolor=T["TEXT"], bordercolor=T["BORDER"], lightcolor=T["INPUT"], darkcolor=T["INPUT"],
                         padding=4)
            st.map("TCombobox", fieldbackground=[("readonly", T["INPUT"])], foreground=[("readonly", T["TEXT"])],
                   background=[("readonly", T["CARD"])], selectbackground=[("readonly", T["INPUT"])],
                   selectforeground=[("readonly", T["TEXT"])])
            st.configure("TNotebook", background=T["CARD"], borderwidth=0)
            st.configure("TNotebook.Tab", background=T["BG"], foreground=T["MUTED"], padding=(16, 6),
                         font=(FONT, 10, "bold"), borderwidth=0)
            st.map("TNotebook.Tab", background=[("selected", T["CARD"])], foreground=[("selected", T["ACCENT"])])
            st.configure("Drawable.Horizontal.TProgressbar", troughcolor=T["TROUGH"], background=T["ACCENT"],
                         lightcolor=T["ACCENT"], darkcolor=T["ACCENT"], bordercolor=T["TROUGH"], thickness=10)
            for cb in self.combos:                               # the drop-down lists
                try:
                    pop = self.root.tk.call("ttk::combobox::PopdownWindow", str(cb))
                    self.root.tk.call(f"{pop}.f.l", "configure", "-background", T["INPUT"], "-foreground", T["TEXT"],
                                      "-selectbackground", T["ACCENT"], "-selectforeground", "#ffffff")
                except Exception:
                    pass
        except Exception:
            pass
        self.draw_theme_icon()
        self.draw_hint()
        if self.palette_colors:
            self.draw_palette(self.palette_colors, self.shown_swatch)
        self.set_status(self.status.get())

    def toggle_theme(self, _event=None):
        self.theme = "dark" if self.theme == "light" else "light"
        self.T = THEMES[self.theme]
        self.apply_theme()

    def draw_theme_icon(self):
        c, T = self.theme_btn, self.T
        c.delete("all")
        c.config(bg=T["HEAD"])
        x, y = 22, 22
        if self.theme == "light":                                # the sun
            for k in range(8):
                a = k * math.pi / 4
                c.create_line(x + 11 * math.cos(a), y + 11 * math.sin(a), x + 15 * math.cos(a), y + 15 * math.sin(a),
                              fill="#fbbf24", width=2, capstyle="round")
            c.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#fbbf24", outline="#f59e0b")
        else:                                                    # the moon
            c.create_oval(x - 10, y - 10, x + 10, y + 10, fill="#e5e7eb", outline="")
            c.create_oval(x - 5, y - 14, x + 13, y + 4, fill=T["HEAD"], outline="")
            c.create_oval(x + 9, y + 6, x + 11, y + 8, fill="#ffffff", outline="")
            c.create_oval(x + 3, y + 10, x + 5, y + 12, fill="#ffffff", outline="")

    def draw_hint(self):
        try:
            self.canvas.itemconfig("hint", fill=self.T["HINT"])
        except Exception:
            pass

    # ---- layout
    def _button(self, parent, text, command, kind="plain", **kw):
        """A flat, clean button. kind: 'primary' (orange) or 'plain'."""
        if kind == "primary":
            look = dict(font=(FONT, 11, "bold"), pady=7)
        else:
            look = dict(font=(FONT, 10), pady=4, highlightthickness=1)
        look.update(kw)
        b = tk.Button(parent, text=text, command=command, relief="flat", bd=0, cursor="hand2", padx=12, **look)
        self._t(b, "primary" if kind == "primary" else "btn")
        if kind == "primary":
            b.bind("<Enter>", lambda e: b["state"] == "normal" and b.config(bg=self.T["ACCENT_DARK"]))
            b.bind("<Leave>", lambda e: b["state"] == "normal" and b.config(bg=self.T["ACCENT"]))
        else:
            b.bind("<Enter>", lambda e: b["state"] == "normal" and b.config(bg=self.T["HOVER"]))
            b.bind("<Leave>", lambda e: b["state"] == "normal" and b.config(bg=self.T["CARD"]))
        return b

    def _label(self, parent, text, role="label", **kw):
        return self._t(tk.Label(parent, text=text, **kw), role)

    def _frame(self, parent, role="cardin", **kw):
        return self._t(tk.Frame(parent, **kw), role)

    def _card(self, parent, **pack):
        card = self._t(tk.Frame(parent, highlightthickness=1), "card")
        card.pack(**pack)
        return card

    def _combo(self, parent, var, values, width):
        cb = ttk.Combobox(parent, textvariable=var, values=values, width=width, state="readonly")
        self.combos.append(cb)
        return cb

    def _scale(self, parent, text, var, lo, hi, length=140):
        return self._t(tk.Scale(parent, from_=lo, to=hi, orient="horizontal", variable=var, length=length, label=text,
                                highlightthickness=0, bd=0, font=(FONT, 9)), "scale")

    def _build(self):
        root = self.root
        try:
            ttk.Style().theme_use("clam")                        # the one built-in look that accepts our colors
        except Exception:
            pass
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        # ---- header, with the sun / moon switch in the corner
        head = self._frame(root, "head")
        head.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._label(head, "Drawable", "headname", font=(FONT, 18, "bold")).pack(side="left", padx=(18, 10), pady=10)
        self._label(head, "Describe a picture. Edit pictures. Watch it get drawn.", "headtag",
                    font=(FONT, 10)).pack(side="left", pady=10)
        self.theme_btn = self._t(tk.Canvas(head, width=44, height=44, highlightthickness=0, cursor="hand2"), "head")
        self.theme_btn.pack(side="right", padx=(0, 14), pady=9)
        self.theme_btn.bind("<Button-1>", self.toggle_theme)

        left = self._frame(root, "bg")
        left.grid(row=1, column=0, sticky="ns", padx=(14, 8), pady=14)
        right = self._frame(root, "bg")
        right.grid(row=1, column=1, sticky="nsew", padx=(0, 14), pady=14)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        # ---- left, top: what to draw / what to change
        ask = self._card(left, fill="x")
        inner = self._frame(ask)
        inner.pack(fill="x", padx=12, pady=10)
        self._label(inner, "What should it draw or change?", font=(FONT, 10, "bold")).pack(anchor="w")
        self.prompt = self._t(tk.Text(inner, width=38, height=4, wrap="word", font=(FONT, 10), undo=True, relief="flat",
                                      bd=0, padx=8, pady=6, highlightthickness=1), "text")
        self.prompt.pack(fill="x", pady=(4, 8))
        self.prompt.bind("<Control-Return>", lambda e: (self.on_create(), "break")[1])
        srow = self._frame(inner)
        srow.pack(fill="x", pady=(0, 8))
        self._label(srow, "Style", font=(FONT, 10, "bold")).pack(side="left")
        self.style = tk.StringVar(value=DEFAULT_STYLE)
        self.style_btn = self._button(srow, DEFAULT_STYLE, self.open_styles)
        self.style_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.set_style(DEFAULT_STYLE)
        self.create_btn = self._button(inner, "Create & draw", self.on_create, "primary")
        self.create_btn.pack(fill="x")
        up = self._frame(inner)
        up.pack(fill="x", pady=(8, 0))
        self._button(up, "Add pictures", self.pick_images).pack(side="left")
        self._button(up, "Clear", self.clear_images).pack(side="left", padx=6)
        self.path_label = self._label(inner, "No pictures chosen", "muted", font=(FONT, 9))
        self.path_label.pack(anchor="w", pady=(3, 0))
        self.o_draw = tk.BooleanVar(value=True)
        self._t(tk.Checkbutton(inner, text="Draw it step by step (video + drawing)", variable=self.o_draw, font=(FONT, 10),
                               anchor="w", highlightthickness=0), "check").pack(anchor="w", pady=(4, 0))

        # ---- left, middle: two tabs
        tabs_card = self._card(left, fill="x", pady=(10, 0))
        nb = ttk.Notebook(tabs_card)
        nb.pack(fill="x", padx=4, pady=4)
        t1 = self._frame(nb)
        t2 = self._frame(nb)
        nb.add(t1, text="Drawing")
        nb.add(t2, text="Edit")

        # Drawing tab
        pad = self._frame(t1)
        pad.pack(fill="x", padx=10, pady=8)
        self.detail, self.speed = tk.StringVar(value="Normal"), tk.StringVar(value="Normal")
        row = self._frame(pad)
        row.pack(fill="x")
        self._label(row, "Detail", "muted", font=(FONT, 9)).grid(row=0, column=0, sticky="w")
        self._combo(row, self.detail, list(DETAIL), 10).grid(row=0, column=1, padx=(6, 14))
        self._label(row, "Speed", "muted", font=(FONT, 9)).grid(row=0, column=2, sticky="w")
        self._combo(row, self.speed, list(SPEED), 9).grid(row=0, column=3, padx=6)
        self.colors = tk.IntVar(value=24)
        self._scale(pad, "Colors (6 to 100)", self.colors, 6, 100, 280).pack(fill="x", pady=(4, 0))
        self.o_outline, self.o_fill = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        self.o_bg, self.o_rec = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        grid = self._frame(pad)
        grid.pack(fill="x")
        for i, (text, var) in enumerate((("Outlines first", self.o_outline), ("Fill in colors", self.o_fill),
                                         ("Draw the background", self.o_bg), ("Record the video", self.o_rec))):
            self._t(tk.Checkbutton(grid, text=text, variable=var, font=(FONT, 10), anchor="w", highlightthickness=0),
                    "check").grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 12))

        # Edit tab
        pad2 = self._frame(t2)
        pad2.pack(fill="x", padx=10, pady=8)
        self.layout, self.change = tk.StringVar(value=LAYOUTS[0]), tk.StringVar(value="Medium")
        r1 = self._frame(pad2)
        r1.pack(fill="x")
        self._label(r1, "Combine", "muted", font=(FONT, 9)).grid(row=0, column=0, sticky="w")
        self._combo(r1, self.layout, LAYOUTS, 12).grid(row=0, column=1, padx=(6, 12))
        self._label(r1, "AI change", "muted", font=(FONT, 9)).grid(row=0, column=2, sticky="w")
        self._combo(r1, self.change, list(CHANGE), 8).grid(row=0, column=3, padx=6)
        self.o_ai = tk.BooleanVar(value=True)
        self._t(tk.Checkbutton(pad2, text="Let the AI blend my pictures using my words", variable=self.o_ai,
                               font=(FONT, 10), anchor="w", highlightthickness=0), "check").pack(anchor="w", pady=(2, 0))
        self.e_bright, self.e_contrast, self.e_sat = tk.IntVar(value=0), tk.IntVar(value=100), tk.IntVar(value=100)
        self.e_blur, self.e_sharp = tk.IntVar(value=0), tk.IntVar(value=0)
        sc = self._frame(pad2)
        sc.pack(fill="x")
        for i, (text, var, lo, hi) in enumerate((("Brightness", self.e_bright, -100, 100), ("Contrast", self.e_contrast, 50, 150),
                                                 ("Color", self.e_sat, 0, 200), ("Blur", self.e_blur, 0, 10),
                                                 ("Sharpen", self.e_sharp, 0, 10))):
            self._scale(sc, text, var, lo, hi, 130).grid(row=i // 2, column=i % 2, padx=(0, 10), sticky="w")
        self.e_filter, self.e_rotate = tk.StringVar(value="None"), tk.StringVar(value="0")
        self.e_flip = tk.BooleanVar(value=False)
        cell = self._frame(sc)
        cell.grid(row=2, column=1, sticky="w")
        self._t(tk.Checkbutton(cell, text="Flip left-right", variable=self.e_flip, font=(FONT, 10), anchor="w",
                               highlightthickness=0), "check").pack(anchor="w")
        self._button(cell, "Reset edits", self.reset_edits).pack(anchor="w", pady=(4, 0))
        r3 = self._frame(pad2)
        r3.pack(fill="x", pady=(2, 0))
        self._label(r3, "Filter", "muted", font=(FONT, 9)).grid(row=0, column=0, sticky="w")
        self._combo(r3, self.e_filter, FILTERS, 12).grid(row=0, column=1, padx=(6, 12))
        self._label(r3, "Turn", "muted", font=(FONT, 9)).grid(row=0, column=2, sticky="w")
        self._combo(r3, self.e_rotate, ["0", "90", "180", "270"], 5).grid(row=0, column=3, padx=6)

        # ---- right: colors in use, the drawing paper, progress, download buttons
        self.pal = self._t(tk.Canvas(right, height=30, highlightthickness=0), "bg")
        self.pal.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        paper = self._frame(right, "card", highlightthickness=1)
        paper.grid(row=1, column=0, sticky="nsew")
        self.canvas = self._t(tk.Canvas(paper, width=self.CW, height=self.CH, highlightthickness=0), "canvas")
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.create_text(self.CW // 2, self.CH // 2, text="Your drawing will appear here", fill="#a8a29e",
                                font=(FONT, 14), tags="hint")

        prog = self._frame(right, "bg")
        prog.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.status = tk.StringVar(value="Ready. Type what you want drawn, then press Create & draw.")
        self.status_label = self._label(prog, "", "bglabel", textvariable=self.status, wraplength=640, justify="left",
                                        font=(FONT, 10), anchor="w")
        self.status_label.pack(fill="x", pady=(0, 6))
        pr = self._frame(prog, "bg")
        pr.pack(fill="x")
        self.bar = ttk.Progressbar(pr, mode="determinate", maximum=100, style="Drawable.Horizontal.TProgressbar")
        self.bar.pack(side="left", fill="x", expand=True)
        self.skip_btn = self._button(pr, "Skip ahead", self.on_skip, state="disabled")
        self.skip_btn.pack(side="right", padx=(8, 0))
        self.stop_btn = self._button(pr, "Stop", self.on_stop, state="disabled")
        self.stop_btn.pack(side="right", padx=(8, 0))

        self.results = self._frame(right, "bg")
        self.results.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.results.grid_remove()
        self.results_title = self._label(self.results, "Your drawing is ready", "bglabel", font=(FONT, 11, "bold"))
        self.results_title.pack(anchor="w", pady=(0, 6))
        btns = self._frame(self.results, "bg")
        btns.pack(anchor="w")
        self.dl_image_btn = self._button(btns, "Download drawing", self.download_image, "primary")
        self.dl_video_btn = self._button(btns, "Download video", self.download_video, "primary")
        self.dl_src_btn = self._button(btns, "Download picture", self.download_source)
        self.watch_btn = self._button(btns, "Watch video", self.watch_video)

    # ---- small helpers
    def set_status(self, text):
        self.status.set(text)
        bad = text.startswith(("Something went wrong", "A problem", "Could not"))
        try:
            self.status_label.config(fg="#ef4444" if bad else self.T["TEXT"])
        except Exception:
            pass

    def set_busy(self, busy):
        self.create_btn.config(state="disabled" if busy else "normal",
                               bg=self.T["ACCENT_OFF"] if busy else self.T["ACCENT"])
        self.stop_btn.config(state="normal" if busy else "disabled")
        self.skip_btn.config(state="disabled")

    def show(self, frame):
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        h, w = frame.shape[:2]
        s = min(cw / w, ch / h)
        if abs(s - 1) > 0.01:
            frame = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))),
                               interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        self.photo = self.ImageTk.PhotoImage(Image.fromarray(frame))
        if self.img_item is None:
            self.canvas.delete("hint")
            self.img_item = self.canvas.create_image(cw // 2, ch // 2, image=self.photo)
        else:
            self.canvas.coords(self.img_item, cw // 2, ch // 2)
            self.canvas.itemconfig(self.img_item, image=self.photo)

    def draw_palette(self, colors, active):
        """The strip of colors. It shrinks the swatches to fit, so even 100 colors fit on one line."""
        self.pal.delete("all")
        try:
            width = max(200, int(self.pal.winfo_width()))
        except Exception:
            width = 600
        n = max(1, len(colors))
        sw = max(4, min(24, (width - 8) // n))
        gap = 4 if sw >= 14 else (1 if sw >= 8 else 0)
        for i, col in enumerate(colors):
            x = 4 + i * (sw + gap)
            fill = mixhex(col, 0.0 if i <= active else 0.65, self.T["BG"])
            if i == active and sw >= 8:
                self.pal.create_rectangle(x, 2, x + sw, 26, fill=fill, outline=self.T["TEXT"], width=2)
            else:
                self.pal.create_rectangle(x, 4, x + sw, 24, fill=fill, outline=self.T["BORDER"] if sw >= 8 else fill)

    def clear_results(self):
        self.results.grid_remove()
        for b in (self.watch_btn, self.dl_video_btn, self.dl_image_btn, self.dl_src_btn):
            b.pack_forget()
        if self.video_path:
            try:
                os.remove(self.video_path)
            except OSError:
                pass
        self.video_path = None
        self.final_img = None
        self.source_img = None

    # ---- styles
    def set_style(self, name):
        self.style.set(name)
        self.style_photo = self.ImageTk.PhotoImage(style_thumb(name, (60, 40)))
        self.style_btn.config(text=name, image=self.style_photo, compound="left")

    def open_styles(self):
        """A window with one picture for each style, so you can see the look before you choose it."""
        T = self.T
        top = tk.Toplevel(self.root)
        top.title("Choose a style")
        top.configure(bg=T["BG"])
        try:
            top.transient(self.root)
            top.resizable(False, False)
            top.geometry(f"+{self.root.winfo_rootx() + 70}+{self.root.winfo_rooty() + 30}")
        except Exception:
            pass
        tk.Label(top, text="Pick a look for your picture", bg=T["BG"], fg=T["TEXT"],
                 font=(FONT, 13, "bold")).pack(pady=(12, 0))
        tk.Label(top, text="The small pictures show the look. Real results will differ a little.", bg=T["BG"],
                 fg=T["MUTED"], font=(FONT, 9)).pack(pady=(0, 6))
        grid = tk.Frame(top, bg=T["BG"])
        grid.pack(padx=10, pady=(0, 12))
        refs = []

        def choose(name):
            self.set_style(name)
            top.destroy()

        for i, name in enumerate(STYLE_NAMES):
            cell = tk.Frame(grid, bg=T["CARD"], highlightthickness=3, cursor="hand2",
                            highlightbackground=T["ACCENT"] if name == self.style.get() else T["BORDER"])
            cell.grid(row=i // 4, column=i % 4, padx=6, pady=6)
            photo = self.ImageTk.PhotoImage(style_thumb(name, (160, 107)))
            refs.append(photo)
            pic = tk.Label(cell, image=photo, bd=0, bg=T["CARD"])
            pic.pack()
            txt = tk.Label(cell, text=name, bg=T["CARD"], fg=T["TEXT"], font=(FONT, 10, "bold"), pady=5)
            txt.pack()
            for w in (cell, pic, txt):
                w.bind("<Button-1>", lambda e, n=name: choose(n))
        top.refs = refs
        top.bind("<Escape>", lambda e: top.destroy())
        try:
            top.grab_set()
        except Exception:
            pass

    # ---- pictures and edits
    def pick_images(self):
        ps = filedialog.askopenfilenames(title="Choose one or more pictures",
                                         filetypes=[("Pictures", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")])
        if ps:
            self.pics = list(ps)
            self._pics_label()

    def clear_images(self):
        self.pics = []
        self._pics_label()

    def _pics_label(self):
        n = len(self.pics)
        self.path_label.config(text="No pictures chosen" if n == 0 else
                               os.path.basename(self.pics[0]) if n == 1 else f"{n} pictures chosen (they will be combined)")

    def reset_edits(self):
        self.e_bright.set(0)
        self.e_contrast.set(100)
        self.e_sat.set(100)
        self.e_blur.set(0)
        self.e_sharp.set(0)
        self.e_filter.set("None")
        self.e_rotate.set("0")
        self.e_flip.set(False)

    def read_edits(self):
        return {"brightness": int(self.e_bright.get()), "contrast": int(self.e_contrast.get()),
                "saturation": int(self.e_sat.get()), "blur": int(self.e_blur.get()), "sharpen": int(self.e_sharp.get()),
                "filter": self.e_filter.get(), "rotate": int(self.e_rotate.get()), "flip": bool(self.e_flip.get())}

    # ---- start / stop
    def on_create(self):
        if self.create_btn["state"] == "disabled":
            return
        prompt = self.prompt.get("1.0", "end").strip()
        pics = list(self.pics)
        if not prompt and not pics:
            self.set_status("Type what you want drawn, or add your own pictures.")
            return
        draw = bool(self.o_draw.get())
        if draw and not self.o_outline.get() and not self.o_fill.get():
            self.set_status("Turn on Outlines first and/or Fill in colors.")
            return
        seconds = SPEED[self.speed.get()]
        opts = {"max_side": DETAIL[self.detail.get()], "colors": int(self.colors.get()),
                "outlines": self.o_outline.get(), "fill": self.o_fill.get(), "bg": self.o_bg.get(),
                "seconds": seconds, "record": bool(self.o_rec.get()) and seconds > 0, "draw": draw,
                "layout": self.layout.get(), "change": self.change.get(), "style": self.style.get(), "ai": bool(self.o_ai.get()),
                "edits": self.read_edits()}
        stop_flag.clear()
        self.clear_results()
        self.drawing = None
        self.set_busy(True)
        self.bar.configure(mode="indeterminate")
        self.bar.start(12)
        threading.Thread(target=self.work, args=(prompt, pics, opts), daemon=True).start()

    def work(self, prompt, pics, o):
        """Runs in the background: makes the picture (imagined, or combined from yours), edits it, plans the drawing."""
        say = lambda t: self.q.put(("status", t))
        try:
            if prompt:
                n = wanted_colors(prompt)
                if n:
                    o["colors"] = n
                    say(f"Your prompt asks for {n} colors - using that.")
            style, surprised = o["style"], o["style"] == SURPRISE
            if surprised:
                style = random.choice(list(STYLES))
                o["style"] = style
                if not pics or (prompt and o["ai"]):
                    say(f"Surprise! Using the {style} style.")
            if pics:
                say(f"Loading {len(pics)} picture{'s' if len(pics) > 1 else ''}...")
                imgs = [_fit_max(load_image(p), 1600) for p in pics]
                if len(imgs) > 1:
                    say(f"Combining {len(imgs)} pictures ({o['layout']})...")
                img = combine_images(imgs, o["layout"])
                if prompt and o["ai"]:
                    img = restyle_image(img, prompt, o["change"], say, o["style"])
                kind = "edited"
            else:
                img = generate_image(prompt, say, o["style"])
                kind = "imagined"
            if stop_flag.is_set():
                raise Stopped()
            if edits_changed(o["edits"]):
                say("Applying your edits...")
                img = apply_edits(img, o["edits"])
                kind = "edited"
            self.q.put(("source", img, kind, style if (surprised and (not pics or (prompt and o["ai"]))) else None))
            if not o["draw"]:
                self.q.put(("edited", img))
                return
            say("Planning the drawing...")
            plan = build_plan(img, o["max_side"], o["colors"], o["outlines"], o["fill"], o["bg"])
            self.q.put(("plan", plan, o))
        except Stopped:
            self.q.put(("stopped",))
        except Exception as e:
            self.q.put(("error", f"{type(e).__name__}: {e}"))

    def poll(self):
        """Handles messages from the background threads. One bad message can never stop this loop."""
        try:
            while True:
                try:
                    msg = self.q.get_nowait()
                except queue.Empty:
                    break
                try:
                    self.handle(msg)
                except Exception as e:
                    self.end_run(f"Something went wrong: {type(e).__name__}: {e}")
        finally:
            self.root.after(80, self.poll)

    def handle(self, msg):
        kind = msg[0]
        if kind == "status":
            self.set_status(msg[1])
        elif kind == "source":
            self.source_img, self.source_kind, self.surprise_style = msg[1], msg[2], msg[3]
        elif kind == "plan":
            self.begin_drawing(msg[1], msg[2])
        elif kind == "edited":
            self.finish_picture(msg[1])
        elif kind == "drawn":
            self.finish(msg[1], stopped=msg[2], error=msg[3])
        elif kind == "stopped":
            self.end_run("Stopped.")
        elif kind == "error":
            self.end_run("Something went wrong: " + msg[1])

    def begin_drawing(self, plan, o):
        self.bar.stop()
        self.bar.configure(mode="determinate", value=0)
        d = Drawing(plan, o["seconds"], o["record"])
        if o["record"] and d.video_failed:
            self.set_status("Video recording is not available on this PC. Drawing without it.")
        self.drawing = d
        self.palette_colors = ([plan["line_color"]] if d.has_outline else []) + list(plan["palette"])
        self.shown_swatch = 0
        self.draw_palette(self.palette_colors, 0)
        self.skip_btn.config(state="normal")
        self.img_item = None
        self.render = {"frame": None, "progress": 0.0, "label": "", "swatch": 0, "version": 0, "seen": 0, "active": True}
        self.render_lock = threading.Lock()
        # The drawing and the video recording run in their own thread, so this window can never freeze while it draws.
        threading.Thread(target=self.render_loop, args=(d,), daemon=True).start()
        self.root.after(40, self.watch_render)

    def render_loop(self, d):
        err = None
        try:
            while not d.finished and not stop_flag.is_set():
                t0 = time.perf_counter()
                frame = d.step()
                with self.render_lock:
                    r = self.render
                    r.update(frame=frame, progress=d.progress, label=d.label, swatch=d.swatch, version=r["version"] + 1)
                if not d.instant:
                    time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        try:
            d.finalize()
        except Exception as e:
            err = err or f"{type(e).__name__}: {e}"
        self.render["active"] = False
        self.q.put(("drawn", d, stop_flag.is_set(), err))

    def watch_render(self):
        """Shows the newest drawn frame (about 25 times a second). Cheap, so the window stays smooth."""
        try:
            r = self.render
            with self.render_lock:
                fresh = r["version"] != r["seen"]
                frame, prog, label, sw = r["frame"], r["progress"], r["label"], r["swatch"]
                r["seen"] = r["version"]
            if fresh and frame is not None:
                self.show(frame)
                self.bar.configure(value=prog * 100)
                if label:
                    self.set_status(label)
                if sw != self.shown_swatch:
                    self.shown_swatch = sw
                    self.draw_palette(self.palette_colors, sw)
        except Exception:
            pass
        if self.render["active"]:
            self.root.after(40, self.watch_render)

    def _style_note(self):
        return f" (Surprise style: {self.surprise_style})" if self.surprise_style else ""

    def _src_button_text(self):
        return "Download edited picture" if self.source_kind == "edited" else "Download imagined picture"

    def show_results(self, title, drawing_done):
        self.results_title.config(text=title)
        self.results.grid()
        if drawing_done:
            self.dl_image_btn.pack(side="left", padx=(0, 6))
            if self.video_path:
                self.dl_video_btn.pack(side="left", padx=(0, 6))
        if self.source_img is not None:
            self.dl_src_btn.config(text=self._src_button_text())
            self.dl_src_btn.pack(side="left", padx=(0, 6))
        if drawing_done and self.video_path:
            self.watch_btn.pack(side="left")

    def finish_picture(self, img):
        """Edit-only run (no drawing): just show the finished picture."""
        self.drawing = None
        self.final_img = None
        self.img_item = None
        self.show(np.array(img.convert("RGB")))
        self.bar.configure(value=100)
        self.end_run("Done! Your picture is ready." + self._style_note())
        self.show_results("Your picture is ready", False)

    def finish(self, d, stopped, error=None):
        self.drawing = None
        self.final_img = d.image()
        self.video_path = d.video_path if (d.video_path and os.path.exists(d.video_path)) else None
        self.show(np.array(self.final_img))
        self.bar.configure(value=100)
        self.draw_palette(self.palette_colors, len(self.palette_colors))
        if error:
            msg = "A problem stopped the drawing early (" + error + "). Here is what was drawn."
        elif stopped:
            msg = "Stopped. Here is what was drawn so far."
        else:
            msg = "Done! Your drawing is ready." + self._style_note()
        self.end_run(msg)
        self.show_results("Your drawing is ready", True)

    def on_ui_error(self, exc, val, tb):
        try:
            self.end_run(f"Something went wrong: {exc.__name__}: {val}")
        except Exception:
            pass

    def end_run(self, text):
        self.bar.stop()
        self.bar.configure(mode="determinate")
        self.set_status(text)
        self.set_busy(False)

    def on_stop(self):
        stop_flag.set()
        self.set_status("Stopping...")

    def on_skip(self):
        if self.drawing is not None:
            self.drawing.speed = 12.0

    # ---- results: watch and download
    def download_image(self):
        if self.final_img is None:
            return
        path = filedialog.asksaveasfilename(title="Save the drawing", defaultextension=".png", initialfile="drawing.png",
                                            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")])
        if path:
            try:
                self.final_img.save(path)
                self.set_status("Drawing saved: " + path)
            except Exception as e:
                self.set_status(f"Could not save the drawing: {e}")

    def download_source(self):
        if self.source_img is None:
            return
        path = filedialog.asksaveasfilename(title="Save the picture", defaultextension=".png", initialfile="picture.png",
                                            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")])
        if path:
            try:
                self.source_img.convert("RGB").save(path)
                self.set_status("Picture saved: " + path)
            except Exception as e:
                self.set_status(f"Could not save the picture: {e}")

    def download_video(self):
        if not self.video_path or not os.path.exists(self.video_path):
            return
        ext = os.path.splitext(self.video_path)[1]
        path = filedialog.asksaveasfilename(title="Save the drawing video", defaultextension=ext,
                                            initialfile="drawing" + ext, filetypes=[("Video", "*" + ext)])
        if path:
            try:
                shutil.copyfile(self.video_path, path)
                self.set_status("Video saved: " + path)
            except Exception as e:
                self.set_status(f"Could not save the video: {e}")

    def watch_video(self):
        if not self.video_path or not os.path.exists(self.video_path):
            return
        top = tk.Toplevel(self.root)
        top.title("Drawing video")
        top.configure(bg=self.T["BG"])
        label = tk.Label(top, bg=self.T["BG"])
        label.pack()
        state = {"cap": None, "photo": None, "gen": 0}

        def nxt(gen):
            if gen != state["gen"] or state["cap"] is None:
                return
            ok, bgr = state["cap"].read()
            if not ok:
                state["cap"].release()
                state["cap"] = None
                return
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            s = min(1.0, 700 / w, 600 / h)
            if s < 1:
                rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            state["photo"] = self.ImageTk.PhotoImage(Image.fromarray(rgb))
            label.configure(image=state["photo"])
            top.after(int(1000 / FPS), lambda: nxt(gen))

        def play():
            if state["cap"] is not None:
                state["cap"].release()
            state["gen"] += 1
            state["cap"] = cv2.VideoCapture(self.video_path)
            nxt(state["gen"])

        def close():
            state["gen"] += 1
            if state["cap"] is not None:
                state["cap"].release()
            top.destroy()

        self._button(top, "Play again", play).pack(pady=6)
        top.protocol("WM_DELETE_WINDOW", close)
        play()

    def on_close(self):
        stop_flag.set()
        cleanup_temp()
        self.root.destroy()


# ------------------------------------------------------------------ style preview pictures (built in, so the app stays one file)
STYLE_THUMBS_B64 = {
    "Storybook cartoon": (
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUEBAQEAwUEBAQGBQUGCA0ICAcHCBALDAkNExAUExIQEhIUFx0ZFBYcFhISGiMaHB4fISEhFBkk"
        "JyQgJh0gISD/2wBDAQUGBggHCA8ICA8gFRIVICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICD/wAAR"
        "CADIASwDASIAAhEBAxEB/8QAHAAAAgMBAQEBAAAAAAAAAAAAAAIFBgcEAwgB/8QAVxAAAQMDAQQEBgwKBQoGAwAAAQACAwQFEQYHEiExE0FR"
        "YRQWInGBlBUjMlVWdJGTsbLR0hc1NjdCUlehwdNic3WVswgkJjNDVHKDkvAnNDhTguGitPH/xAAcAQABBQEBAQAAAAAAAAAAAAAAAgMEBQYB"
        "Bwj/xAA5EQABAwIDBAYJBAIDAQAAAAABAAIDBBEFEiEUMUFRE2FxgZHRBhUiMlKSocHwIzOx4RZCNFNicv/aAAwDAQACEQMRAD8A1TCMJsIw"
        "t3dYGyXCMJsIwi6LJcIwmwjCLoslwjCbCMIuiyXCMJsIwi6LJcIwmwjCLoslwjCbCMIuiyXCMJsIwi6LJcIwmwjCLoslwmiilmlbFDG6SRxw"
        "1rBknzBdlJarhXFngtFNK1zg0Paw7oOccTyC1OyWKkstIGRND53D2yYji7u7h3KDVVjIBzPJT6WifOeQ5rOI9J6hlYHttrgD+s9rT8hOVHVd"
        "vrqB4ZWUssBPLfbgHzHrW3LyqaanrKd9PVQtlieMFrhkKsZir7+20W6laPwlmX2HG/WsNwjCnNRWN1luIjYXPppQXRPPPvB7xwULhXrJGyND"
        "27iqCSN0bix28JcIwmwjCcum7JcIwmwjCLoslwjCbCMIuiyXCMJsIwi6LJcIwmwjCLoslwjCbCMIuiyXCMJsIwi6LJ8IwnwjCRdLskwjCfCM"
        "IuiyTCMJ8Iwi6LJMLooqGouFbHSUzd6WQ4GeAHaT3K3ae0jHU07K66B3RvGWQDLSR1Enn6Fcqa3UFE7epKOGF2N3eYwBxHZnmqufEWRktZqf"
        "orWnw58gDnmw+qjLTpe2WyJpfC2pqMeVLK3PHuHV9Pept0cb4+jexrmfqkZCZCz75HvOZxuVomRsjGVgsFVb5o+jrIHT22JtNVNGQxvBj+7H"
        "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"
    ),
    "Coloring page": (
        "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"
        "ekphD6ErD7xKSpBIBwQceseeEI1xmUf/2Q=="
    ),
    "Watercolor": (
        "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"
        "C7SMDzwo7Ss4IEqrRPc7SU+0L1iQ7ELk3TMpnjWSSf3KBQYAQYJ8hXB5a0bSRFkS3Iv5SyIBiSElnFKhpduEPIR3v/7iY7JZh08BAAgBxAuY"
        "QKbXR6pNZzJJBut4CHGRHC53olMO1DnE+3ELo/a2CJjC4y8m74e/0qv8ikQQQhJa5sE/KO0Z5Q2kYP8AwmFwETef0C4b1sf/AFFEcwV3HFxK"
        "4D1i93+pNBmwXXD/AFDZeDlRY+e/ZOw7KgeZMEQUu02IBjyUdwDQDe9lOOJ9a6Q4VemUXhwu0AcL3iIIAmSsT0y7f0WlumMXW3MDGO4VdJU2"
        "d14IROZA8pTEkt/hMBUuAYQd7yRN00UXFIkG6zHbtxdAvaIWkWQ0i1gsKrrWsrvpFpJBXbGzM86u0GevDha3gqF0GSF4ndQABa1t+AlPUWke"
        "7EYhdrMtaNCQTOTx4S5kkTGF4D1AbSWsPieVSOoVmk4PnsktCWjWMxIE/KO4loJt5WN/X1hi54BOU519SILRPzyiwtGqdsAEi2SSl+oxglzx"
        "OFi1NTWeDufLVWXQQC837o6hOo33ailIc5wVLtZRALpsPCxgTtMy4fwpbBN+Ak6g6jSd1MAnawyqj1Gp2XiPtAMTGFAA4TDh8EJLYlsrLcGZ"
        "EpiZF7CYhE7ScRaVGNL3be+EJWCTk6QAPfieUSBum5BNl6DpKzYhvm6pLS1xD7eE6UJR8o75tbNh/wBWLX9zxdRcGaR5fJBFhlczMAONwusr"
        "0GV27HTJN1QOmaZrYLSJ8pYySRa6G9i1sbjK7Y3TjOjYIxjyvWCSJm/dKymxjAxos3ARJ3NDS7amN9ynzTU8kpr5ZHElt3XxPdCC4WC1ui9O"
        "Z1HWCm93sbc+V1Nf01on0C2k3aRghQs25jxT6JE7X43NsY/Uh4OBuXCBYHKYy2wzlWazTu0uqfScZ22Vd4xcYlTE01aK2UXFuLNPo1emx7jU"
        "9vbyts6ilE/UaR/C5Fpd4thNL/8A3CfjhNcbLvU5aWvjWPpujqams0zSR9UBIeo6VkAVQQLFcoXcFxM90zjgjGPKToR2fO5fiKOnHV9KCSHE"
        "zawXJ+p9FqdbqhqtLQfUpEZC2ek9Kqa2uCW7aQMlxXc0tPTp0hTFMQMQhSUHaLXR2tjYTnkSSPh1Ho/Ua7xTZpqhjwten6L6vVptd9ID5yvr"
        "jaNFpG2mB2tlWCN0chOed/BZWcPoSOjaJmk1DCKgV/8ArOniHSR2Wj6j6b9fS/XpN9zcwuNaCG7XXjKakpdzPbvI7Ovl6FVfB0o6vpXWBgnj"
        "snb1DSGB9UScLlH2H/hEe0gg8ZR0IjR5zOvKTOsf1DTGwqj/AJXM6t7Ha15bglUnaWl7SpuJ4E90qjRE3ORntxUZKqGgTmUpBJiPKYyIlsBK"
        "BMO3T5TyrC73GTnEBLBNo3eO6YD3E84nundSqBm4sgISb8D4wlLvFWVO9rS6IIESUWb6hDQCSUD2IN+/Cv0zgyo1zhjN06KTkkx+CEZ5Ywm6"
        "TfcsfoKop72j3C5XlcQ0w6QQcreFalt9rhficrH1YadS408chTNnDCCUomn5zi9bVxQy6789qv8AkrY1zztY0uPhO+hVpmXtiO4XX+l+nUDo"
        "zqXN3vceeF7eu6DTVOn1Km0NcwTI5WelvxWb06K6HEzlr+v1d6uj5+0kEyBdSxvtRMG9rYKUgm+4j4VmUZCCQGzC9eicBqW7wB2K8zSGggH4"
        "PKgLoNwYwnQl0yUiRrZvQzRy1dOzoiRNoPZY2t2f1DiCJjC8T9ZqG16bDV9hlM9xJuTunKmbGwskUkjS8vzmPewLFCFf3IHS7F8KPkC1/CA3"
        "C2bXTQALmJ5UEyRDEEtzg3VcEkB1h+qskGds3SgTMjGJSAafRepN6frt7o2kXXYV/UnTmUpFcF5uGgL54QWuvE4lUOpPdq21HfYBElQ82ljz"
        "S65Fpq8nl1oenHwaOu1R1eufV2iHGV5sC8oTJl2OApPuAFxwVLilFUitlJzk5PywgtDYkl3CguSXW4ULSRubkFG0S4gX44Thosc7QCFo9M6b"
        "U12pa3Z/bFySEnT9C/XagUmz5K77R6OlodMxjBeL+SmSlRbaGi88uuX9KLNPpaWm0zaVFoDVbBaNthPlEumPbfsjHukgG0hcDYRioqkIQQ2R"
        "eDyi1w3SfcUS2eIEzEYRLW0wLCCgcV1GNqNdTdfcIIXD9a6VU0VX6jBNJxyF3brAkNmO6o1VBmpoPbUAIIt4TouiBu6kdmFfK8HzIzG0X8oA"
        "EiwthenW6V+j1T6TwRyF5ZO9dzESi4ScZeUENbdoMR3UubTabIe0tLgb8gotwLhKNA69gRATQAJCUAzJAI8IhxcbAhAHq0RpjUbXT2W0G/24"
        "c0R3XOyWO3Nk+UD1PVP1H0Q727byp+vnjCLUka7huY19PDLFmj/2XatjG6mGkHnwqrgEgxPZRznuBda6Xadsc/woU3cm0ZjYyLLllkiqTfg8"
        "9Y1W1KTmF3mFcZtMXUIBdtMm/wC6YBomRKRu0cXJvs2b/ROu0+n0/pV5FIX+Fb1f1JS6ho3UdMIY7Lu65ipJpuYAIiEKFP6VLaSCB5yob1MT"
        "yerXcnrkM6w+gn2LWmwABLuLI+7wPEJWzgWM5PAQuy0x8qWV4QXSQInyU5kNJIGFFEAI5kkEi/BUIBjki9yoogA2LpnIlQXm8hRRIAWDI2e7"
        "uoQ7JwoolAVwaCZMoCMEgHsookAJsRLZtZQSP8ZJKiiAGvIF/PhNTp/UrBoP3fuoogdFW0jveldPbodGyGy91yVpzMA2OflRRRm7PQcUI44K"
        "MfBMG4nhMWwHA54UUQdQTaHGwyoWy0ySIUUQApcSTBgJSbe3jhRRAGJ17p39XpPrsEVKf7hcS6ztpUUXaD7GT5jHGOVSXyTbl0Huji8gA4Ci"
        "ieUYYBB7d5SbQZAsVFEANILQJIhBrGF+4NgkXKiiUAtG0Q1whEuAwVFEgAAtb5SEy0kA97KKJQCAYvIb3PCZobMBx8+FFEgAIdkuscIhgI/y"
        "CiiAP//Z"
    ),
    "Pencil sketch": (
        "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"
        "uBb2eXZbQ0NLmD//2Q=="
    ),
    "Anime": (
        "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"
    ),
    "Pixel art": (
        "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"
    ),
    "Oil painting": (
        "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"
    ),
    "3D cartoon": (
        "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"
        "rKc1uxCE25MuK//Z"
    ),
    "Realistic photo": (
        "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"
    ),
    "Surprise me": (
        "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"
        "qlQI4nojlH0QIAVKnSX/2Q=="
    ),
}


def style_thumb(name, size):
    b = STYLE_THUMBS_B64.get(name)
    if not b:
        return Image.new("RGB", size, (200, 200, 200))
    return Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB").resize(size, Image.LANCZOS)


_lock_socket = None


def only_one_copy():
    """A second copy of the app never opens: it closes straight away instead of starting again and again."""
    global _lock_socket
    import socket
    try:
        _lock_socket = socket.socket()
        _lock_socket.bind(("127.0.0.1", 47653))
        return True
    except OSError:
        return False


def raise_window_priority():
    """Windows: keep this window's thread ahead of the heavy image-model threads, so it stays responsive."""
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.SetThreadPriority(k.GetCurrentThread(), 1)
    except Exception:
        pass


def main():
    multiprocessing.freeze_support()                           # needed for the packaged Windows app
    if not only_one_copy():
        return
    raise_window_priority()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
