DRAWABLE AI
**Type what you imagine. Watch it get drawn.
Drawable is a free Windows app with an AI inside. You describe a picture (or choose one of your own) and it draws it stroke by stroke: outlines first, then colors. When it finishes you get the finished drawing and a video of the whole drawing, each with its own Download button. It can also edit existing images according to your instructions.

- Free, no account, nothing to install
- Runs on your own computer, no graphics card needed
- Your words and pictures stay on your computer

---

## Download and use it (easy steps)

1. Go to the **[Releases page](../../releases/latest)** of this repository.
2. Under **Assets**, click **Drawable.exe** to download it.
3. Double-click **Drawable.exe**.
   - Windows may say **"Windows protected your PC"** because the app isn't signed yet.
     Click **More info**, then **Run anyway**.
   - The first start takes a little while (it is one big file). Please wait.
4. Type what you want drawn, then press **Create & draw**.
   - The **first time**, it downloads its AI model (about **5 GB**). You need internet and free space. After that it works without internet.
   - Making a picture from words takes about **1 to 2 minutes** on a normal laptop.
5. Watch it draw. When it says **Done**, press:
   - **Download drawing** to save the picture
   - **Download video** to save the video of the whole drawing
   - **Watch video** to play it inside the app

Nothing is saved on your computer until you press a Download button.

### Tips
- **Use my own picture**: choose any picture and Drawable will draw that instead. This is much faster, and good if you want a very detailed picture.
- **Colors**: pick how many colors it uses (6 to 32).
- **Speed**: slow, normal, fast or instant. Press **Skip ahead** to speed up the drawing.
- The AI is small so it can run on a laptop. Its pictures are simpler than big online image makers.

### Show the Drawable logo on the app (optional)
Apps built by the newest workflow already have the logo. If your `Drawable.exe` shows a plain icon (it can also happen with older builds, or because Windows hasn't refreshed), you can give it the logo yourself:

1. Download **Drawable.ico** from this repository (click it, then the download button).
2. Right-click `Drawable.exe`, choose **Show more options** (Windows 11) then **Send to** and **Desktop (create shortcut)**.
3. Right-click the new shortcut on your desktop and choose **Properties**.
4. On the **Shortcut** tab, click **Change Icon...**, then **Browse...**.
5. Pick **Drawable.ico**, click **Open**, **OK**, then **Apply**.

Use that shortcut to open Drawable. You can pick any other icon you like the same way, it is all yours.

(An icon is separate from Windows' "unsigned app" warning. The icon only changes how the app looks.)

### If something goes wrong
| Problem | What to try |
|---|---|
| Windows blocks the app | Click **More info**, then **Run anyway** |
| It says "Something went wrong" | Close the app and open it again. If it keeps happening, open an Issue and copy the message |
| Nothing happens for a long time | The first run downloads about 5 GB. Keep the window open |
| The picture doesn't match my words | Try shorter, clearer words, or use your own picture |
| An error about a missing DLL | Install the [Microsoft Visual C++ runtime](https://aka.ms/vs/17/release/vc_redist.x64.exe), restart your PC, try again |

---

## Make your own copy of the app (optional)

You don't need this to use Drawable. It is only if you want to build the app yourself.

1. Click **Fork** (top right) to copy this repository to your own GitHub account.
2. Open the **Actions** tab and allow workflows if GitHub asks.
3. Click **Build Drawable (single Windows exe)**, then **Run workflow**.
4. Wait until it finishes (it can take a long time).
5. Your app is now on your repository's **Releases** page and in the run's **Artifacts** box, as **Drawable.exe**.

### Run it from the code instead
1. Install Python 3.10 or newer from python.org (tick **Add Python to PATH**).
2. In this folder, run:
   ```
   pip install torch
   pip install -r requirements.txt
   python paint_ai.py
   ```

---

## What is in this repository

| File | What it is |
|---|---|
| `paint_ai.py` | The whole app |
| `requirements.txt` | The libraries the app uses |
| `.github/workflows/build-windows-app.yml` | Builds `Drawable.exe` on GitHub |
| `index.html` | The download website (optional, for GitHub Pages) |
| `banner.png` | The picture at the top of this page (optional) |
| `Drawable.ico` | The Drawable logo as an icon (used when building the app) |
| `README.md` | This page (optional, you can delete it) |

---

Made with Python, OpenCV, PyTorch and the [sd-turbo](https://huggingface.co/stabilityai/sd-turbo) image model.
