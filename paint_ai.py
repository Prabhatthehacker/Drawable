"""Drawable - local AI that draws (and colors) pictures in MS Paint by controlling the mouse.

Made for Windows 10 Paint (maximized). Everything runs on your PC.
Give control with the green button. Take it back any time: click the red bar, press ESC, or move the mouse.
"""
import os, sys, threading, time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np, cv2, pyautogui, pygetwindow as gw
from PIL import Image

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True
for _n in ("MINIMUM_DURATION", "MINIMUM_SLEEP"):
    if hasattr(pyautogui, _n):
        setattr(pyautogui, _n, 0)

MODEL = "stabilityai/sd-turbo"
VIDEO_DIR = r"C:\Users\DELL\OneDrive\Desktop\Drawing Videos"
pipe = None
stop_flag = False
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class GenStopped(Exception):
    pass


ui_hooks = {}


# ---------------------------------------------------------------- Paint layout
# Windows 10 Paint, maximized (screen coordinates, taken from a real screenshot).
SWATCH_X0, SWATCH_Y0, SWATCH_D = 763, 62, 22   # first palette swatch + spacing
PENCIL = (243, 70)
HOME_TAB = (85, 35)
COLOR1_OFFSET = (-75, 8)                         # "Color 1" box relative to first swatch
WHITE_INDEX = 10                                 # bottom-left swatch in the palette

# Paint's standard 20 swatches (row 1 then row 2). Used only to recognise the layout;
# the real colors are read from your screen.
EXPECTED = [
    (0, 0, 0), (127, 127, 127), (136, 0, 21), (237, 28, 36), (255, 127, 39),
    (255, 242, 0), (34, 177, 76), (0, 162, 232), (63, 72, 204), (163, 73, 164),
    (255, 255, 255), (195, 195, 195), (185, 122, 87), (255, 174, 201), (255, 201, 14),
    (239, 228, 176), (181, 230, 29), (153, 217, 234), (112, 146, 190), (200, 191, 231),
]

DETAIL = {"Fast": 300, "Normal": 420, "Detailed": 520}
SPEED = {"Slow": 0.010, "Normal": 0.004, "Fast": 0.0015}   # pause after every mouse move (seconds)


# ---------------------------------------------------------------- image model
def get_pipe(status):
    global pipe
    if pipe is None:
        import torch
        from diffusers import AutoPipelineForText2Image
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        kw = {"torch_dtype": torch.float16} if dev == "cuda" else {}
        status("Loading image model (first time takes about a minute)...")
        try:
            p = AutoPipelineForText2Image.from_pretrained(MODEL, local_files_only=True, **kw)
        except Exception:
            p = AutoPipelineForText2Image.from_pretrained(MODEL, **kw)
        pipe = p.to(dev)
    return pipe


def generate_image(prompt, status):
    p = get_pipe(status)
    text = (prompt + ", simple cartoon illustration, flat solid colors, thick black outlines, "
            "centered, isolated on a plain white background")
    args = dict(num_inference_steps=4, guidance_scale=0.0, height=512, width=512)

    def cb(_pipe, step, _t, kwargs):
        if stop_flag or esc_pressed():
            raise GenStopped()
        status(f"Generating image... step {step + 1}/4 (CPU can take a minute or two)")
        return kwargs

    status("Generating image... (CPU can take a minute or two)")
    try:
        return p(text, callback_on_step_end=cb, **args).images[0]
    except TypeError:
        return p(text, **args).images[0]


def load_image(path):
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


# ---------------------------------------------------------------- image -> drawing plan
def quantize(rgb, palette):
    """Map every pixel to the nearest Paint palette color; returns an index map."""
    rgb = cv2.bilateralFilter(rgb, 7, 40, 7)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pal = np.array(palette, np.uint8).reshape(1, -1, 3)
    pal_lab = cv2.cvtColor(pal, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    wgt = np.array([1.0, 1.6, 1.6], np.float32)     # favour matching the hue over the brightness
    d = (((lab[:, :, None, :] - pal_lab[None, None, :, :]) * wgt) ** 2).sum(-1)
    labels = d.argmin(-1).astype(np.uint8)
    # majority filter: removes speckles so regions come out clean
    scores = np.stack([cv2.blur((labels == k).astype(np.float32), (5, 5))
                       for k in range(len(palette))], -1)
    return scores.argmax(-1).astype(np.uint8)


def drop_background(labels):
    """Anything of one color that touches the picture's border is background: leave it white."""
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    bg = int(np.bincount(border, minlength=20).argmax())
    if bg == WHITE_INDEX:
        return labels
    n, cc = cv2.connectedComponents((labels == bg).astype(np.uint8))
    edge_ids = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
    edge_ids = edge_ids[edge_ids > 0]
    labels = labels.copy()
    labels[np.isin(cc, edge_ids)] = WHITE_INDEX
    return labels


def runs_from_mask(mask):
    """Horizontal runs (y, x0, x1) covering every True pixel."""
    m = np.pad((mask > 0).astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    ys, xs = np.nonzero(d == 1)
    _, xe = np.nonzero(d == -1)
    return list(zip(ys.tolist(), xs.tolist(), (xe - 1).tolist()))


def fill_runs(labels, k):
    return runs_from_mask(labels == k)


def line_mask(rgb):
    """The picture's own dark lines (cartoon / coloring-page outlines)."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = (g < 100).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 6          # ignore specks
    return keep[cc]


def outline_strokes(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    edges = cv2.Canny(g, 60, 150)
    edges[:4, :] = 0                          # the picture's own edge is not an outline
    edges[-4:, :] = 0
    edges[:, :4] = 0
    edges[:, -4:] = 0
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    strokes = []
    for c in cnts:
        if len(c) < 18:
            continue
        c = cv2.approxPolyDP(c, 1.0, False).reshape(-1, 2)
        if len(c) >= 2:
            strokes.append(c)
    if not strokes:
        return []
    ordered, cur = [], np.array([0, 0])
    starts = np.array([s[0] for s in strokes])
    alive = np.ones(len(strokes), bool)
    for _ in range(len(strokes)):
        d = np.where(alive, np.linalg.norm(starts - cur, axis=1), np.inf)
        i = int(d.argmin())
        alive[i] = False
        ordered.append(strokes[i])
        cur = strokes[i][-1]
    return ordered


def build_plan(img, palette, box, detail, do_fill, do_outline, keep_bg=True):
    x1, y1, x2, y2 = box
    cw, ch = x2 - x1, y2 - y1
    w0, h0 = img.size
    s = min((cw - 20) / w0, (ch - 20) / h0, DETAIL.get(detail, 420) / max(w0, h0))
    w, h = max(8, int(w0 * s)), max(8, int(h0 * s))
    rgb = cv2.resize(np.array(img.convert("RGB")), (w, h), interpolation=cv2.INTER_AREA)
    ox, oy = x1 + (cw - w) // 2, y1 + (ch - h) // 2
    plan = {"outline_runs": [], "outlines": [], "fills": [], "origin": (ox, oy), "size": (w, h)}
    line = np.zeros((h, w), bool)
    if do_outline:
        lm = line_mask(rgb)
        if lm.mean() >= 0.004:                # the picture has real black lines: draw those
            line = lm
            plan["outline_runs"] = runs_from_mask(line)
        else:                                  # photo-like picture: trace edges instead
            plan["outlines"] = outline_strokes(rgb)
            m = np.zeros((h, w), np.uint8)
            for st in plan["outlines"]:
                cv2.polylines(m, [st.reshape(-1, 1, 2).astype(np.int32)], False, 255, 1, cv2.LINE_8)
            line = m > 0
    if not do_fill:
        target = np.full((h, w), WHITE_INDEX, np.uint8)
        target[line] = 0
        plan["target"] = target
    if do_fill:
        src = rgb
        if line.any():                         # paint the lines out so colors reach right up to them
            wide = cv2.dilate(line.astype(np.uint8), np.ones((3, 3), np.uint8)) * 255
            src = cv2.inpaint(rgb, wide, 3, cv2.INPAINT_TELEA)
        labels = quantize(src, palette)
        if not keep_bg:
            labels = drop_background(labels)
        labels[line] = 255                     # colors never paint over the outlines
        target = labels.copy()
        target[target == 255] = WHITE_INDEX
        target[line] = 0                       # outlines are black
        plan["target"] = target
        order = sorted(range(len(palette)), key=lambda k: -int((labels == k).sum()))
        for k in order:
            if k == WHITE_INDEX:
                continue                       # canvas is already white
            runs = fill_runs(labels, k)
            if runs:
                plan["fills"].append((k, runs))
    return plan


# ---------------------------------------------------------------- Paint helpers
def find_paint():
    wins = [w for w in gw.getAllWindows() if w.title and w.visible]
    for w in wins:
        if w.title.endswith("- Paint"):
            return w
    for w in wins:
        if "Paint" in w.title and "Draw" not in w.title and "3D" not in w.title:
            return w
    return None


def esc_pressed():
    try:
        import ctypes
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000)
    except Exception:
        return False


def countdown_point(msg, status, secs=5):
    for i in range(secs, 0, -1):
        status(f"{msg} ({i})")
        time.sleep(1)
    return pyautogui.position()


def make_layout(x0, y0, d, pencil, home):
    k = d / SWATCH_D
    return {"x0": x0, "y0": y0, "d": d, "pencil": pencil, "home": home,
            "c1": (int(x0 + COLOR1_OFFSET[0] * k), int(y0 + COLOR1_OFFSET[1] * k))}


def swatch_xy(L, i):
    row, col = divmod(i, 10)
    return int(L["x0"] + col * L["d"]), int(L["y0"] + row * L["d"])


def read_palette(L, shot):
    cols = []
    for i in range(20):
        x, y = swatch_xy(L, i)
        if 0 <= x < shot.width and 0 <= y < shot.height:
            cols.append(shot.getpixel((x, y))[:3])
        else:
            cols.append((0, 0, 0))
    return cols


def layout_matches(cols):
    good = sum(1 for c, e in zip(cols, EXPECTED) if max(abs(a - b) for a, b in zip(c, e)) <= 45)
    return good >= 17


def detect_layout(status):
    shot = pyautogui.screenshot()
    L = make_layout(SWATCH_X0, SWATCH_Y0, SWATCH_D, PENCIL, HOME_TAB)
    cols = read_palette(L, shot)
    if layout_matches(cols):
        return L, cols
    # Different layout / resolution: ask the user to point at three things.
    px, py = countdown_point("Hover the PENCIL tool in Paint", status)
    bx, by = countdown_point("Hover the BLACK swatch (top-left of the colors)", status)
    rx, _ = countdown_point("Hover the PURPLE swatch (top-right of the colors)", status)
    d = (rx - bx) / 9.0
    if d < 8:
        raise RuntimeError("Could not read the color palette positions.")
    L = make_layout(bx, by, d, (px, py), None)
    shot = pyautogui.screenshot()
    return L, read_palette(L, shot)


def find_canvas(status):
    shot = pyautogui.screenshot()
    a = np.array(shot.convert("L"))
    mask = (a > 250).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = None
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        if w >= 300 and h >= 200 and y > 120:
            box = (x + 3, y + 3, x + w - 3, y + h - 3)
    if box is None:
        sw, sh = shot.size
        box = (8, 172, sw - 20, sh - 85)
    return box


class Driver:
    """Moves the mouse in Paint (smoothly, with small pauses) and notices when you take over."""

    def __init__(self, status, delay=0.004):
        self.status = status
        self.delay = delay
        self.last = None
        self.stopped = False
        self.reason = ""
        self.warn = False

    def _stop(self, why):
        self.stopped = True
        self.reason = why
        return False

    def move(self, x, y):
        if self.stopped:
            return False
        if stop_flag:
            return self._stop("you pressed Take Back Control")
        if esc_pressed():
            return self._stop("you pressed ESC")
        if self.last is not None:
            p = pyautogui.position()
            if abs(p.x - self.last[0]) > 4 or abs(p.y - self.last[1]) > 4:
                return self._stop("you moved the mouse")
        pyautogui.moveTo(x, y)
        self.last = (x, y)
        time.sleep(self.delay)
        return True

    def click(self, x, y):
        if not self.move(x, y):
            return False
        time.sleep(0.05)
        pyautogui.mouseDown()
        time.sleep(0.03)
        pyautogui.mouseUp()
        time.sleep(0.08)
        return True

    def path(self, pts):
        if not self.move(*pts[0]):
            return False
        time.sleep(0.015)                          # let the pointer settle before pressing
        pyautogui.mouseDown()
        time.sleep(0.015 + self.delay)
        ok = True
        try:
            for p in pts[1:]:
                px, py = self.last
                steps = int(max(abs(p[0] - px), abs(p[1] - py)) // 25) + 1   # glide, do not teleport
                for i in range(1, steps + 1):
                    ix = round(px + (p[0] - px) * i / steps)
                    iy = round(py + (p[1] - py) * i / steps)
                    if not self.move(ix, iy):
                        ok = False
                        break
                if not ok:
                    break
        finally:
            time.sleep(self.delay)
            pyautogui.mouseUp()
            time.sleep(self.delay)
        return ok

    def select_color(self, L, cols, i):
        want = cols[i]
        for _ in range(3):
            if not self.click(*swatch_xy(L, i)):
                return False
            time.sleep(0.12)
            try:
                got = pyautogui.pixel(*L["c1"])
                if max(abs(a - b) for a, b in zip(got, want)) <= 30:
                    return True
            except Exception:
                return True
        self.warn = True                      # could not confirm; keep going
        return True


class Recorder:
    """Records only the drawing area of the Paint canvas to a video file in VIDEO_DIR."""

    def __init__(self, region, timelapse=False):
        x, y, x2, y2 = region
        self.region = (x, y, (x2 - x) // 2 * 2, (y2 - y) // 2 * 2)   # video sizes must be even
        self.fps = 12
        self.timelapse = timelapse
        self.interval = 0.25 if timelapse else 1.0 / self.fps        # time-lapse plays about 3x faster
        self.path = None
        self.frames = 0
        self.vw = None
        self._stop = threading.Event()
        self._t = None

    def start(self):
        folder = VIDEO_DIR
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception:
            return False
        stamp = time.strftime("%Y%m%d_%H%M%S")
        w, h = self.region[2], self.region[3]
        for ext, codec in (("mp4", "mp4v"), ("avi", "MJPG")):
            path = os.path.join(folder, f"drawing_{stamp}.{ext}")
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), self.fps, (w, h))
            if vw.isOpened():
                self.vw, self.path = vw, path
                break
        if self.vw is None:
            return False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return True

    def _frame(self):
        img = pyautogui.screenshot(region=self.region)
        frame = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        try:                                   # small ring that shows where the pen is
            mx, my = pyautogui.position()
            cx, cy = mx - self.region[0], my - self.region[1]
            if 0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]:
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), 1, cv2.LINE_AA)
        except Exception:
            pass
        return frame

    def _loop(self):
        t0 = time.time()
        written = 0
        while not self._stop.is_set():
            t1 = time.time()
            try:
                frame = self._frame()
            except Exception:
                time.sleep(0.05)
                continue
            if self.timelapse:
                n = 1
            else:                              # real time: repeat frames if capturing is slow
                n = max(1, int((time.time() - t0) * self.fps) + 1 - written)
            for _ in range(min(n, 60)):
                self.vw.write(frame)
            written += n
            self.frames += 1
            time.sleep(max(0.0, self.interval - (time.time() - t1)))

    def stop(self):
        self._stop.set()
        if self._t is not None:
            self._t.join(3)
        if self.vw is not None:
            self.vw.release()
        return self.path if self.frames else None


def repair(plan, L, cols, dr, status):
    """Look at what Paint actually drew and redraw any pixels that were missed."""
    ox, oy = plan["origin"]
    w, h = plan["size"]
    target = plan["target"]
    pal = np.array(cols, np.int16)
    for attempt in range(3):
        if dr.stopped:
            return
        time.sleep(0.4)
        shot = np.array(pyautogui.screenshot(region=(ox, oy, w, h)).convert("RGB")).astype(np.int16)
        if shot.shape[:2] != target.shape:
            return
        bad = np.abs(shot - pal[target]).max(-1) > 40
        n = int(bad.sum())
        if n < 4 or n > 0.4 * bad.size:       # nothing to fix, or the canvas is not where we think
            return
        status(f"Fixing {n} missed pixels (round {attempt + 1}/3)...")
        for k in np.unique(target[bad]):
            runs = runs_from_mask(bad & (target == k))
            if not dr.select_color(L, cols, int(k)):
                return
            for y, xa, xb in runs:
                pts = [(ox + xa, oy + y)] if xb <= xa else [(ox + xa, oy + y), (ox + xb, oy + y)]
                if not dr.path(pts):
                    return


def draw_plan(plan, L, cols, status, delay=0.004):
    dr = Driver(status, delay)
    ox, oy = plan["origin"]
    nf = sum(len(r) for _, r in plan["fills"])
    total = nf + len(plan["outlines"]) + len(plan["outline_runs"])
    done = 0
    if L.get("home"):
        dr.click(*L["home"])
        time.sleep(0.3)
    if not dr.click(*L["pencil"]):
        return dr
    time.sleep(0.2)

    def tick():
        nonlocal done
        done += 1
        if done % 25 == 0:
            status(f"Drawing... {done}/{total} strokes (ESC or move the mouse to stop)")

    # 1) outline everything in black
    def draw_runs(runs):
        for y, xa, xb in runs:
            pts = [(ox + xa, oy + y)] if xb <= xa else [(ox + xa, oy + y), (ox + xb, oy + y)]
            if not dr.path(pts):
                return False
            tick()
        return True

    if plan["outline_runs"] or plan["outlines"]:
        status("Step 1/2: drawing outlines...")
        if not dr.select_color(L, cols, 0):               # black
            return dr
        if not draw_runs(plan["outline_runs"]):
            return dr
        for st in plan["outlines"]:
            pts = [(ox + int(px), oy + int(py)) for px, py in st]
            if not dr.path(pts):
                return dr
            tick()
    # 2) then color it in
    if plan["fills"]:
        status("Step 2/2: coloring...")
    for k, runs in plan["fills"]:
        if not dr.select_color(L, cols, k):
            return dr
        if not draw_runs(runs):
            return dr
    if not dr.stopped:
        repair(plan, L, cols, dr, status)
    return dr


# ---------------------------------------------------------------- the app
def request_stop():
    global stop_flag
    stop_flag = True


def run(prompt, image_path, detail, speed, do_fill, do_outline, keep_bg, record, timelapse, status, done, ui):
    global stop_flag
    stop_flag = False
    try:
        ui(lambda: ui_hooks["compact"](True))    # control given: this window becomes the Take Back Control bar
        time.sleep(0.4)

        # A typed prompt ALWAYS wins: the AI imagines the picture from scratch.
        # An uploaded image is used only when the prompt box is empty.
        if prompt:
            if image_path:
                status("Using your prompt and ignoring the uploaded image...")
                time.sleep(1.0)
            status(f'Imagining "{prompt}" from scratch... (CPU can take a few minutes)')
            img = generate_image(prompt, status)
            try:
                img.save(os.path.join(SCRIPT_DIR, "last_imagined.png"))
            except Exception:
                pass
        else:
            status("Loading your uploaded image...")
            img = load_image(image_path)
        if stop_flag:
            status("Stopped.")
            return

        status("Looking for Paint window...")
        win = find_paint()
        if win is None:
            status("Waiting... open MS Paint (a blank canvas) now.")
            while win is None and not stop_flag:
                time.sleep(0.5)
                win = find_paint()
        if stop_flag:
            status("Stopped.")
            return

        try:
            if win.isMinimized:
                win.restore()
            win.maximize()
        except Exception:
            pass
        for _ in range(3):
            try:
                win.activate()
            except Exception:
                pass
            time.sleep(0.8)
            active = gw.getActiveWindow()
            if active is not None and "Paint" in (active.title or ""):
                break
        time.sleep(0.5)

        L, cols = detect_layout(status)
        box = find_canvas(status)
        plan = build_plan(img, cols, box, detail, do_fill, do_outline, keep_bg)
        n = sum(len(r) for _, r in plan["fills"]) + len(plan["outlines"]) + len(plan["outline_runs"])
        mins = n * (5 * SPEED.get(speed, 0.004) + 0.012) / 60
        status(f"Drawing {n} strokes (about {max(1, round(mins))} min).")
        rec = None
        if record:
            ox, oy = plan["origin"]
            pw, ph = plan["size"]
            region = (max(box[0], ox - 8), max(box[1], oy - 8), min(box[2], ox + pw + 8), min(box[3], oy + ph + 8))
            rec = Recorder(region, timelapse)
            if not rec.start():
                rec = None
                status(f"Could not create or open the video folder, so nothing is recorded: {VIDEO_DIR}")
                time.sleep(2)
        try:
            dr = draw_plan(plan, L, cols, status, SPEED.get(speed, 0.004))
        finally:
            video = rec.stop() if rec else None
        if dr.stopped:
            msg = f"Control given back because {dr.reason}."
        elif dr.warn:
            msg = "Done! (Paint's color box could not be confirmed - check the colors.)"
        else:
            msg = "Done!"
        if video:
            msg += f" Video saved: {video}"
        status(msg)
    except GenStopped:
        status("Control given back before drawing started.")
    except pyautogui.FailSafeException:
        status("Emergency stop (mouse hit a screen corner).")
    except Exception as e:
        status(f"Error: {e}")
    finally:
        try:
            pyautogui.mouseUp()
        except Exception:
            pass
        ui(lambda: ui_hooks["compact"](False))
        done()


def main():
    global root
    root = tk.Tk()
    root.title("Drawable")
    W, H = 480, 330
    home_geo = f"{W}x{H}+{root.winfo_screenwidth() - W - 10}+{root.winfo_screenheight() - H - 60}"
    root.geometry(home_geo)
    root.attributes("-topmost", True)
    img_path = tk.StringVar()

    def ui(fn):
        root.after(0, fn)

    body = tk.Frame(root)
    body.pack(fill="both", expand=True)

    # the small bar this same window turns into while it has control
    bar = tk.Frame(root, bg="#222222")
    bar_label = tk.Label(bar, text="", bg="#222222", fg="white", font=("Segoe UI", 9), wraplength=340)
    bar_label.pack(fill="x", pady=(3, 0))
    bar_btn = tk.Button(bar, bg="#d9534f", fg="white", activebackground="#b52b27", activeforeground="white",
                        font=("Segoe UI", 9, "bold"))
    bar_btn.pack(fill="x", padx=4, pady=4)

    def bar_click():
        request_stop()
        bar_btn.config(text="Control given back...", state="disabled")

    bar_btn.config(command=bar_click)

    def set_compact(on):
        if on:
            body.pack_forget()
            bar_btn.config(text="TAKE BACK CONTROL  (or ESC / move the mouse)", state="normal")
            bar_label.config(text="Control given. Starting...")
            bar.pack(fill="both", expand=True)
            cw = 360
            root.geometry(f"{cw}x70+{root.winfo_screenwidth() - cw - 4}+0")
        else:
            bar.pack_forget()
            body.pack(fill="both", expand=True)
            root.geometry(home_geo)
            root.attributes("-topmost", True)

    ui_hooks["compact"] = set_compact

    tk.Label(body, text="Prompt (e.g. 'a cat sitting on a moon')").pack(pady=(10, 0))
    prompt_entry = tk.Entry(body, width=48)
    prompt_entry.pack(pady=4)
    path_label = tk.Label(body, text="No image uploaded", fg="gray")
    path_label.pack()

    def upload():
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if p:
            img_path.set(p)
            path_label.config(text=os.path.basename(p))

    def clear_upload():
        img_path.set("")
        path_label.config(text="No image uploaded")

    row = tk.Frame(body)
    row.pack(pady=4)
    tk.Button(row, text="Upload Image", command=upload).pack(side="left", padx=4)
    tk.Button(row, text="Clear", command=clear_upload).pack(side="left", padx=4)

    opts = tk.Frame(body)
    opts.pack(pady=2)
    fill_var, line_var, detail_var = tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.StringVar(value="Normal")
    tk.Checkbutton(opts, text="Fill colors", variable=fill_var).pack(side="left", padx=4)
    tk.Checkbutton(opts, text="Outlines", variable=line_var).pack(side="left", padx=4)
    bg_var = tk.BooleanVar(value=True)
    tk.Checkbutton(opts, text="Draw background", variable=bg_var).pack(side="left", padx=4)
    tk.Label(opts, text="Detail:").pack(side="left", padx=(8, 0))
    ttk.Combobox(opts, textvariable=detail_var, values=list(DETAIL), width=8, state="readonly").pack(side="left")
    speed_var = tk.StringVar(value="Slow")
    tk.Label(opts, text="Speed:").pack(side="left", padx=(8, 0))
    ttk.Combobox(opts, textvariable=speed_var, values=list(SPEED), width=7, state="readonly").pack(side="left")

    rec_row = tk.Frame(body)
    rec_row.pack(pady=2)
    rec_var, lapse_var = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
    tk.Checkbutton(rec_row, text="Record video of the drawing", variable=rec_var).pack(side="left", padx=4)
    tk.Checkbutton(rec_row, text="Fast-forward (time-lapse)", variable=lapse_var).pack(side="left", padx=4)

    status_label = tk.Label(body, text="Ready. In Paint: open a blank canvas, maximize it.", fg="blue", wraplength=440)
    status_label.pack(pady=6)

    go_btn = tk.Button(body, text="Give Control & Draw", width=24, bg="#4CAF50", fg="white")
    go_btn.pack()
    tk.Label(body, text="Control ends when you click Take Back Control, press ESC, or move the mouse.",
             fg="gray").pack(pady=6)

    def set_status(t):
        def f():
            status_label.config(text=t)
            bar_label.config(text=t[:95])
        root.after(0, f)

    def finished():
        root.after(0, lambda: go_btn.config(state="normal"))

    def start():
        p, ip = prompt_entry.get().strip(), img_path.get()
        if not p and not ip:
            set_status("Enter a prompt or upload an image.")
            return
        if not fill_var.get() and not line_var.get():
            set_status("Turn on Fill colors and/or Outlines.")
            return
        go_btn.config(state="disabled")
        threading.Thread(target=run, daemon=True,
                         args=(p, ip, detail_var.get(), speed_var.get(), fill_var.get(), line_var.get(), bg_var.get(), rec_var.get(), lapse_var.get(),
                               set_status, finished, ui)).start()

    go_btn.config(command=start)
    root.mainloop()


if __name__ == "__main__":
    main()
