"""Blog Figures 3 and 4 relabeled for the hardware we actually ran (A100s instead of H100s).

Reads the original figure PNGs out of the draft .docx (read-only; the .docx is never modified), repaints the header and
replica-caption text in the same monospace font, and redraws the replica row with 8 replicas. Writes:
    images/blog_fig3_cloud_gpu_1xA100.png     (was: PLANNER ON A CLOUD GPU · 1x H100 · 80GB)
    images/blog_fig4_ray_cluster_8xA100.png   (was: 4 NODES x 8 H100, 32 warm replicas)
"""
import io
import os
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCX = os.path.join(os.path.dirname(HERE), "Real-Time JEPA World Models.docx")
OUT = os.path.join(HERE, "images")
_zip = zipfile.ZipFile(DOCX)


class _Src:   # SRC + "image4.png" -> the figure inside the .docx
    def __add__(self, name):
        return io.BytesIO(_zip.read("word/media/" + name))


SRC = _Src()
HDR, CAP = (124, 139, 134), (82, 98, 93)


def header(im, text, size=24.3):
    d = ImageDraw.Draw(im)
    d.rectangle([248, 72, 1150, 102], fill="white")
    f = ImageFont.truetype(MONO, size)
    d.text((254 - f.getbbox(text)[0], 79 - f.getbbox(text)[1]), text, font=f, fill=HDR)


def caption(im, text, size=22.8, cx=899, top=466):
    d = ImageDraw.Draw(im)
    d.rectangle([560, 458, 1320, 496], fill="white")
    f = ImageFont.truetype(MONO, size)
    b = f.getbbox(text)
    d.text((cx - (b[2] + b[0]) / 2, top - b[1]), text, font=f, fill=CAP)


def selftest():
    # redraw the ORIGINAL strings with our parameters: should reproduce the originals closely
    for f, t in (("image4.png", "PLANNER ON A CLOUD GPU · 1× H100 · 80GB"), ("image2.png", "PLANNER ON A RAY CLUSTER · 4 NODES × 8 H100")):
        o = Image.open(SRC + f).convert("RGB"); im = o.copy(); header(im, t)
        diff = np.abs(np.asarray(o, int) - np.asarray(im, int))[72:103, 248:1150].sum(-1)
        print(f, "header redraw: pixels differing >60:", int((diff > 60).sum()), "of", int((np.abs(np.asarray(o,int)[72:103,248:1150]-255).sum(-1) > 60).sum()), "ink px")
    o = Image.open(SRC + "image2.png").convert("RGB"); im = o.copy(); caption(im, "32 warm replicas · 4 nodes × 8 H100")
    diff = np.abs(np.asarray(o, int) - np.asarray(im, int))[458:497, 560:1320].sum(-1)
    print("caption redraw: pixels differing >60:", int((diff > 60).sum()))


selftest()
# Figure 3: one cloud GPU
im = Image.open(SRC + "image4.png").convert("RGB")
header(im, "PLANNER ON A CLOUD GPU · 1× A100 · 40GB")
im.save(os.path.join(OUT, "blog_fig3_cloud_gpu_1xA100.png"))
# Figure 4: the Ray cluster we ran: 1 node x 8 A100, 8 warm replicas (8 tiles instead of 6)
im = Image.open(SRC + "image2.png").convert("RGB")
header(im, "PLANNER ON A RAY CLUSTER · 1 NODE × 8 A100")
tile = im.crop((600, 384, 687, 444))
ImageDraw.Draw(im).rectangle([480, 382, 1310, 446], fill="white")
for i in range(8):
    im.paste(tile, (500 + 100 * i, 384))
caption(im, "8 warm replicas · 1 node × 8 A100")
im.save(os.path.join(OUT, "blog_fig4_ray_cluster_8xA100.png"))
print("saved")
