#!/usr/bin/env python3
"""Stacks PNGs vertically into one, scaled to the narrowest one's width.

    python stack.py OUT.png TOP.png BOTTOM.png [...]
"""

import sys

from PIL import Image


def main():
    out, parts = sys.argv[1], sys.argv[2:]
    images = [Image.open(path).convert("RGB") for path in parts]
    width = min(image.width for image in images)
    images = [image if image.width == width else
              image.resize((width, round(image.height * width / image.width)), Image.LANCZOS) for image in images]
    canvas = Image.new("RGB", (width, sum(image.height for image in images)), images[0].getpixel((5, 5)))
    top = 0
    for image in images:
        canvas.paste(image, (0, top))
        top += image.height
    canvas.save(out, optimize=True)
    print(out, canvas.size)


if __name__ == "__main__":
    main()
