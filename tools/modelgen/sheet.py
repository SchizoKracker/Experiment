import sys
from PIL import Image
pre = sys.argv[1]; ks = sys.argv[2].split(",")
ims = [Image.open(f"{pre}_{k}.png").convert("RGB") for k in ks]
ims = [im.resize((int(im.width * 640 / im.height), 640)) for im in ims]
o = Image.new("RGB", (sum(i.width for i in ims), 640)); x = 0
for im in ims: o.paste(im, (x, 0)); x += im.width
o.save(pre + "_" + "".join(k[0] for k in ks) + ".png")
