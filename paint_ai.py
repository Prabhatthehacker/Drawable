"""Drawable - an AI that imagines a picture and then draws it, step by step, on its own canvas.

Type what you want and press "Create & draw". It makes the picture on your PC, then draws it
outlines first, colors second. When it finishes you get the finished drawing and the full drawing
video, each with a Download button. Nothing is saved anywhere until you press a Download button.
"""
import atexit
import math
import multiprocessing
import os
import queue
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
STYLE = "colorful storybook illustration, clean smooth shapes, flat colors, simple clean outlines, "
DETAIL = {"Fast": 360, "Normal": 520, "Detailed": 720}          # longest side of the canvas, in pixels
SPEED = {"Slow": 120, "Normal": 45, "Fast": 15, "Instant": 0}   # seconds the drawing takes
FPS = 24
HOLD_SECONDS = 1.5                                              # the finished picture stays on screen this long in the video
WHITE, NONE = 254, 255                                          # label values: paper left white / dark outline pixel

# colors and font for the window
FONT = "Segoe UI"
BG, HEAD, BORDER = "#f5f3ee", "#1f2937", "#e3ded3"
TEXT, MUTED = "#1f2937", "#78716c"
ACCENT, ACCENT_DARK = "#e8590c", "#c94a08"

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
    return max(6, min(32, int(m.group(1)))) if m else None


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


def generate_image(prompt, status):
    p = get_pipe(status)
    text = STYLE + clean_prompt(prompt)
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
        pts = np.ascontiguousarray(pts[::max(1, len(pts) // 12000)], np.float32)
        k = int(max(1, min(colors, len(pts))))
        cv2.setRNGSeed(12345)
        _, _, cents = cv2.kmeans(pts, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5),
                                 3, cv2.KMEANS_PP_CENTERS)
        cents = list(cents)
        merged = True
        while merged:                                           # merge colors that look the same
            merged = False
            for i in range(len(cents)):
                for j in range(i + 1, len(cents)):
                    if np.linalg.norm(cents[i] - cents[j]) < 7:
                        cents[i] = (cents[i] + cents[j]) / 2
                        del cents[j]
                        merged = True
                        break
                if merged:
                    break
        cents = np.array(cents, np.float32)
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
        for _ in range(2):                                      # remove lonely specks
            scores = np.stack([cv2.blur((labels == c).astype(np.float32), (3, 3)) for c in range(k)], -1)
            own = np.take_along_axis(scores, labels[..., None].astype(np.int64), -1)[..., 0]
            labels = np.where(own * 9 <= 2.01, scores.argmax(-1).astype(np.uint8), labels)
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
def mix(color, amount):
    """Fade a color toward white (amount 0 = same, 1 = white)."""
    return "#%02x%02x%02x" % tuple(int(c * (1 - amount) + 255 * amount) for c in color)


class App:
    CW, CH = 600, 480

    def __init__(self, root):
        from PIL import ImageTk
        self.ImageTk = ImageTk
        self.root = root
        root.title("Drawable")
        w = min(1060, max(800, root.winfo_screenwidth() - 40))
        h = min(700, max(560, root.winfo_screenheight() - 90))
        root.geometry(f"{w}x{h}")
        root.minsize(min(w, 940), min(h, 600))
        self.q = queue.Queue()
        self.drawing = None
        self.final_img = None
        self.video_path = None
        self.photo = None
        self.img_item = None
        self.image_path = ""
        self.swatches = []
        self._build()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.report_callback_exception = self.on_ui_error        # a mistake inside a button never leaves the app stuck
        root.after(80, self.poll)

    # ---- layout
    def _button(self, parent, text, command, kind="plain", **kw):
        """A flat, clean button. kind: 'primary' (orange), 'plain' (white with a border)."""
        if kind == "primary":
            look = dict(bg=ACCENT, fg="white", activebackground=ACCENT_DARK, activeforeground="white",
                        disabledforeground="#fbe3d6", font=(FONT, 11, "bold"), pady=7)
        else:
            look = dict(bg="white", fg=TEXT, activebackground="#f1ede4", activeforeground=TEXT,
                        disabledforeground="#b5afa3", font=(FONT, 10), pady=4,
                        highlightthickness=1, highlightbackground=BORDER)
        look.update(kw)
        b = tk.Button(parent, text=text, command=command, relief="flat", bd=0, cursor="hand2",
                      padx=12, **look)
        if kind == "primary":
            b.bind("<Enter>", lambda e: b["state"] == "normal" and b.config(bg=ACCENT_DARK))
            b.bind("<Leave>", lambda e: b["state"] == "normal" and b.config(bg=ACCENT))
        else:
            b.bind("<Enter>", lambda e: b["state"] == "normal" and b.config(bg="#f1ede4"))
            b.bind("<Leave>", lambda e: b["state"] == "normal" and b.config(bg="white"))
        return b

    def _card(self, parent, **pack):
        card = tk.Frame(parent, bg="white", highlightthickness=1, highlightbackground=BORDER)
        card.pack(**pack)
        return card

    def _build(self):
        root = self.root
        root.configure(bg=BG)
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TCombobox", fieldbackground="white", background="white", padding=4)
        style.configure("Drawable.Horizontal.TProgressbar", troughcolor="#e8e4da", background=ACCENT,
                        lightcolor=ACCENT, darkcolor=ACCENT, bordercolor="#e8e4da", thickness=10)
        root.option_add("*TCombobox*Listbox.font", (FONT, 10))

        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        # ---- header
        head = tk.Frame(root, bg=HEAD)
        head.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(head, text="Drawable", bg=HEAD, fg="white", font=(FONT, 18, "bold")).pack(side="left", padx=(18, 10), pady=10)
        tk.Label(head, text="Describe a picture. Watch it get drawn.", bg=HEAD, fg="#cbd5e1",
                 font=(FONT, 10)).pack(side="left", pady=10)

        # ---- left: what to draw + options
        left = tk.Frame(root, bg=BG)
        left.grid(row=1, column=0, sticky="ns", padx=(14, 8), pady=14)
        right = tk.Frame(root, bg=BG)
        right.grid(row=1, column=1, sticky="nsew", padx=(0, 14), pady=14)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ask = self._card(left, fill="x")
        inner = tk.Frame(ask, bg="white")
        inner.pack(fill="x", padx=12, pady=12)
        tk.Label(inner, text="What should it draw?", bg="white", fg=TEXT, font=(FONT, 10, "bold")).pack(anchor="w")
        self.prompt = tk.Text(inner, width=38, height=5, wrap="word", font=(FONT, 10), undo=True, relief="flat", bd=0,
                              bg="#faf8f3", fg=TEXT, insertbackground=TEXT, padx=8, pady=6,
                              highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT)
        self.prompt.pack(fill="x", pady=(4, 8))
        self.prompt.bind("<Control-Return>", lambda e: (self.on_create(), "break")[1])
        self.create_btn = self._button(inner, "Create & draw", self.on_create, "primary")
        self.create_btn.pack(fill="x")
        up = tk.Frame(inner, bg="white")
        up.pack(fill="x", pady=(10, 0))
        self._button(up, "Use my own picture", self.pick_image).pack(side="left")
        self._button(up, "Clear", self.clear_image).pack(side="left", padx=6)
        self.path_label = tk.Label(inner, text="No picture chosen", bg="white", fg=MUTED, font=(FONT, 9))
        self.path_label.pack(anchor="w", pady=(4, 0))

        optc = self._card(left, fill="x", pady=(10, 0))
        oi = tk.Frame(optc, bg="white")
        oi.pack(fill="x", padx=12, pady=10)
        tk.Label(oi, text="Options", bg="white", fg=TEXT, font=(FONT, 10, "bold")).pack(anchor="w")
        self.detail, self.speed = tk.StringVar(value="Normal"), tk.StringVar(value="Normal")
        row = tk.Frame(oi, bg="white")
        row.pack(fill="x", pady=(6, 0))
        tk.Label(row, text="Detail", bg="white", fg=MUTED, font=(FONT, 9)).grid(row=0, column=0, sticky="w")
        ttk.Combobox(row, textvariable=self.detail, values=list(DETAIL), width=10, state="readonly").grid(row=0, column=1, padx=(6, 14))
        tk.Label(row, text="Speed", bg="white", fg=MUTED, font=(FONT, 9)).grid(row=0, column=2, sticky="w")
        ttk.Combobox(row, textvariable=self.speed, values=list(SPEED), width=9, state="readonly").grid(row=0, column=3, padx=6)
        self.colors = tk.IntVar(value=20)
        crow = tk.Frame(oi, bg="white")
        crow.pack(fill="x", pady=(2, 0))
        tk.Label(crow, text="Colors", bg="white", fg=MUTED, font=(FONT, 9)).pack(side="left")
        tk.Scale(crow, from_=6, to=32, orient="horizontal", variable=self.colors, length=200, bg="white", fg=TEXT,
                 troughcolor="#e8e4da", highlightthickness=0, bd=0, activebackground=ACCENT,
                 font=(FONT, 9)).pack(side="left", padx=8)
        self.o_outline, self.o_fill = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        self.o_bg, self.o_rec = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        for text, var in (("Outlines first", self.o_outline), ("Fill in colors", self.o_fill),
                          ("Draw the background", self.o_bg), ("Record the video", self.o_rec)):
            tk.Checkbutton(oi, text=text, variable=var, bg="white", fg=TEXT, activebackground="white",
                           selectcolor="white", font=(FONT, 10), anchor="w", highlightthickness=0).pack(anchor="w")

        prog = self._card(left, fill="x", pady=(10, 0))
        pi = tk.Frame(prog, bg="white")
        pi.pack(fill="x", padx=12, pady=10)
        self.status = tk.StringVar(value="Ready. Type what you want drawn, then press Create & draw.")
        self.status_label = tk.Label(pi, textvariable=self.status, wraplength=300, justify="left", bg="white", fg=TEXT,
                                     font=(FONT, 10))
        self.status_label.pack(anchor="w", pady=(0, 6))
        self.bar = ttk.Progressbar(pi, mode="determinate", maximum=100, style="Drawable.Horizontal.TProgressbar")
        self.bar.pack(fill="x")
        ctl = tk.Frame(pi, bg="white")
        ctl.pack(fill="x", pady=(8, 0))
        self.stop_btn = self._button(ctl, "Stop", self.on_stop, state="disabled")
        self.stop_btn.pack(side="left")
        self.skip_btn = self._button(ctl, "Skip ahead", self.on_skip, state="disabled")
        self.skip_btn.pack(side="left", padx=6)

        # ---- right: colors in use, the drawing paper, and the download buttons
        self.pal = tk.Canvas(right, height=34, highlightthickness=0, bg=BG)
        self.pal.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        paper = tk.Frame(right, bg="white", highlightthickness=1, highlightbackground=BORDER)
        paper.grid(row=1, column=0, sticky="nsew")
        self.canvas = tk.Canvas(paper, width=self.CW, height=self.CH, bg="#fbfaf7", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.create_text(self.CW // 2, self.CH // 2, text="Your drawing will appear here", fill="#a8a29e",
                                font=(FONT, 14), tags="hint")

        self.results = tk.Frame(right, bg=BG)
        self.results.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.results.grid_remove()
        tk.Label(self.results, text="Your drawing is ready", bg=BG, fg=TEXT, font=(FONT, 11, "bold")).pack(side="left", padx=(0, 12))
        self.dl_image_btn = self._button(self.results, "Download drawing", self.download_image, "primary")
        self.dl_video_btn = self._button(self.results, "Download video", self.download_video, "primary")
        self.watch_btn = self._button(self.results, "Watch video", self.watch_video)

    # ---- small helpers
    def set_status(self, text):
        self.status.set(text)
        bad = text.startswith(("Something went wrong", "A problem", "Could not"))
        try:
            self.status_label.config(fg="#b91c1c" if bad else TEXT)
        except Exception:
            pass

    def set_busy(self, busy):
        self.create_btn.config(state="disabled" if busy else "normal", bg="#f1b79a" if busy else ACCENT)
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
        self.pal.delete("all")
        for i, col in enumerate(colors):
            x = 4 + i * 28
            shown = col if i <= active else mix(col, 0.6)        # colors not used yet look faded
            self.pal.create_rectangle(x, 4, x + 24, 28, fill=mix(col, 0) if i <= active else shown,
                                      outline="#111111" if i == active else "#bbbbbb", width=3 if i == active else 1)

    def clear_results(self):
        self.results.grid_remove()
        for b in (self.watch_btn, self.dl_video_btn, self.dl_image_btn):
            b.pack_forget()
        if self.video_path:
            try:
                os.remove(self.video_path)
            except OSError:
                pass
        self.video_path = None
        self.final_img = None

    # ---- picking a picture
    def pick_image(self):
        p = filedialog.askopenfilename(title="Choose a picture",
                                       filetypes=[("Pictures", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")])
        if p:
            self.image_path = p
            self.path_label.config(text=os.path.basename(p))

    def clear_image(self):
        self.image_path = ""
        self.path_label.config(text="No picture chosen")

    # ---- start / stop
    def on_create(self):
        if self.create_btn["state"] == "disabled":
            return
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt and not self.image_path:
            self.set_status("Type what you want drawn, or choose your own picture.")
            return
        if not self.o_outline.get() and not self.o_fill.get():
            self.set_status("Turn on Outlines first and/or Fill in colors.")
            return
        seconds = SPEED[self.speed.get()]
        opts = {"max_side": DETAIL[self.detail.get()], "colors": int(self.colors.get()),
                "outlines": self.o_outline.get(), "fill": self.o_fill.get(), "bg": self.o_bg.get(),
                "seconds": seconds, "record": bool(self.o_rec.get()) and seconds > 0}
        stop_flag.clear()
        self.clear_results()
        self.drawing = None
        self.set_busy(True)
        self.bar.configure(mode="indeterminate")
        self.bar.start(12)
        threading.Thread(target=self.work, args=(prompt, self.image_path, opts), daemon=True).start()

    def work(self, prompt, path, o):
        """Runs in the background: imagines (or loads) the picture and plans the drawing."""
        say = lambda t: self.q.put(("status", t))
        try:
            if prompt:
                n = wanted_colors(prompt)
                if n:
                    o["colors"] = n
                    say(f"Your prompt asks for {n} colors - using that.")
                img = generate_image(prompt, say)
            else:
                say("Loading your picture...")
                img = load_image(path)
            if stop_flag.is_set():
                raise Stopped()
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
        elif kind == "plan":
            self.begin_drawing(msg[1], msg[2])
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
        self.shown_swatch = -1
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
            msg = "Done! Your drawing is ready."
        self.end_run(msg)
        self.results.grid()
        self.dl_image_btn.pack(side="left", padx=(0, 6))
        if self.video_path:
            self.dl_video_btn.pack(side="left", padx=(0, 6))
            self.watch_btn.pack(side="left")

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
        label = tk.Label(top)
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

        ttk.Button(top, text="Play again", command=play).pack(pady=6)
        top.protocol("WM_DELETE_WINDOW", close)
        play()

    def on_close(self):
        stop_flag.set()
        cleanup_temp()
        self.root.destroy()


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
