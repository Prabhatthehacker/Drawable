DRAWABLE - AI that draws and colors pictures in MS Paint on your PC
===================================================================

FIRST TIME
1. Double-click setup.bat (needs internet; downloads ~5 GB; model is cached
   in your user folder, so it is only downloaded once).
2. Double-click check.bat - every line should say OK / FOUND.
   (If torch fails with WinError 126: install
   https://aka.ms/vs/17/release/vc_redist.x64.exe and restart the PC.)

EVERY TIME
1. Open MS Paint, make a BLANK canvas, and MAXIMIZE the window.
2. Double-click run.bat.
3. Type a prompt (or click Upload Image - this is much faster on a CPU).
4. Pick options: Fill colors, Outlines, Detail (Fast / Normal / Detailed).
5. Click "Draw in Paint" and say Yes to the control request.
   The Drawable window minimizes and the mouse starts drawing.

STOPPING
- Press ESC, or just move the mouse yourself, or fling it into a screen corner.
- When it stops, the Drawable window comes back.

NOTES
- It uses Paint's 20 standard colors; the picture is filled with them first,
  then outlined in black.
- Do not touch the mouse while it draws (that counts as taking control back).
- Made for Windows 10 Paint. If the palette is not where it expects, it asks
  you to hover over the pencil, the black swatch and the purple swatch.
- Generating from a prompt on a CPU takes one to a few minutes.
