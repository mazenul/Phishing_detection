"""
Generate SurfShield notification icons (48x48 PNG).
Pure Python — no third-party libraries required.
Run once before loading the extension:
    python generate_icons.py
"""
import os
import struct
import zlib

ICONS = {
    "safe_48.png":       (0, 200, 83),     # green  #00c853
    "suspicious_48.png": (255, 152, 0),    # orange #ff9800
    "unsafe_48.png":     (244, 67, 54),    # red    #f44336
    "loading_48.png":    (117, 117, 117),  # gray   #757575
}

SIZE = 48
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    payload = tag + data
    return (
        struct.pack(">I", len(data))
        + payload
        + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    )


def make_circle_png(r: int, g: int, b: int, size: int = SIZE) -> bytes:
    """Create a solid-color filled circle on a transparent background."""
    cx = cy = (size - 1) / 2
    radius = size / 2 - 2

    raw_rows = []
    for y in range(size):
        row = bytearray([0])        # filter byte = None
        for x in range(size):
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if dist <= radius:
                alpha = 255
                # Soft anti-alias on the edge
                if dist > radius - 1:
                    alpha = int(255 * (radius - dist + 1))
                row += bytes([r, g, b, max(0, min(255, alpha))])
            else:
                row += bytes([0, 0, 0, 0])   # transparent
        raw_rows.append(bytes(row))

    raw_data   = b"".join(raw_rows)
    compressed = zlib.compress(raw_data, 9)

    sig  = b"\x89PNG\r\n\x1a\n"
    ihdr = _png_chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0),
        # 8-bit depth, colour type 6 = RGBA
    )
    idat = _png_chunk(b"IDAT", compressed)
    iend = _png_chunk(b"IEND", b"")

    return sig + ihdr + idat + iend


def main():
    for filename, (r, g, b) in ICONS.items():
        path = os.path.join(OUT_DIR, filename)
        with open(path, "wb") as f:
            f.write(make_circle_png(r, g, b))
        print(f"  Created: {path}")
    print("\nAll icons generated. You can now load the extension in Chrome.")


if __name__ == "__main__":
    main()
