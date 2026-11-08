# Drawable

Welcome to the Drawable repository!

Drawable is an AI that you can download and use to draw and color pictures in your Microsoft Paint application.

How does it work? First, you just follow the steps below and make sure you keep a Microsoft Paint window open.

# STEPS

## 1. Download and Extract the Files

1. Download the complete ZIP file containing all the Drawable files.
2. Once downloaded on your PC, extract the ZIP file.
3. Create a new folder named **Drawable**.
4. Inside that folder, create another folder named **Drawable**.
5. Move all the files extracted from the ZIP file into this second **Drawable** folder.

Your folder structure should look like this:

Drawable/
└── Drawable/
    ├── setup.bat
    ├── check.bat
    ├── run.bat
    └── Other application files...
```

## 2. Install the Required Software

Before running Drawable, make sure you have the following:

**Microsoft Visual C++ Redistributable (Required)**

1. Download `VC_redist.x64.exe` from the official Microsoft website:

   https://aka.ms/vs/17/release/vc_redist.x64.exe

2. Run the downloaded installer.
3. Complete the installation.
4. Restart your PC if requested.

**Python (Recommended Version: 3.14)**

- The recommended Python version for Drawable is **Python 3.14**.
- Install Python 3.14 before proceeding with setup.

## 3. Run the Setup

1. Double-click `setup.bat`.
2. Make sure you have an internet connection.
3. The setup downloads approximately 5 GB of required AI model files.
4. The model is cached in your user folder, so it only needs to be downloaded once.

## 4. Check the Installation

1. Double-click `check.bat`.
2. Every line should display **OK** or **FOUND**.
3. If PyTorch fails with `WinError 126`, make sure the Microsoft Visual C++ Redistributable is installed.
4. Restart your PC if necessary.

---

# EVERY TIME YOU WANT TO DRAW

1. Open Microsoft Paint.
2. Create a BLANK canvas.
3. MAXIMIZE the Paint window.
4. Double-click `run.bat`.
5. Type a prompt describing what you want to draw, or click **Upload Image** (this is much faster when using a CPU).
6. Choose your options:
   - Fill Colors
   - Outlines
   - Detail: Fast / Normal / Detailed
7. Click **Draw in Paint**.
8. When Windows asks for permission to control your PC, click **Yes**.

The Drawable window minimizes, and the mouse starts drawing your picture automatically in MS Paint.

---

# EXTRA STEP — DRAWING VIDEOS

Drawable can also record the drawing process!

To keep your drawing recordings organized:

1. Go to your Desktop.
2. Create a new folder named **Drawing Videos**.
3. Keep this folder on your Desktop so your drawing videos can be stored there.

You can use this folder to organize and watch recordings of your AI drawing sessions.

---

# STOPPING THE DRAWING

You can stop Drawable at any time:

- Press **ESC**.
- Move the mouse yourself to take control back.
- Move the mouse quickly into a screen corner.

Once drawing stops, the Drawable window comes back.

---

# IMPORTANT NOTES

- Drawable uses Paint's 20 standard colors.
- The picture is filled with colors first and then outlined in black.
- Do not touch or move the mouse while Drawable is working. Moving it counts as taking control back.
- You can use the Upload Image option for faster drawing, especially on a CPU.
- Keep your **Drawing Videos** folder on your Desktop for easy access to your drawing recordings.
- Make sure the Microsoft Visual C++ Redistributable is installed before running Drawable.
- Python 3.14 is the recommended version.

**Enjoy creating pictures with Drawable!**
