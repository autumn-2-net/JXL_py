"""Test cross-format decode: PNG->JXL->JPEG and JPEG->JXL->PNG."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print("importing jxlpy...", flush=True)
import jxlpy

print("=== Test 1: small synthetic PNG -> JXL -> JPEG ===", flush=True)
import numpy as np
arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
print("  encode array to jxl...", flush=True)
t0 = time.time()
jxl = jxlpy.encode(arr)
print(f"  encoded {len(jxl)} bytes in {time.time()-t0:.2f}s", flush=True)

print("  decode_to_jpeg (no reconstruction, should fallback)...", flush=True)
t0 = time.time()
jpg = jxlpy.decode_to_jpeg(jxl)
print(f"  got {len(jpg)} bytes in {time.time()-t0:.2f}s", flush=True)
jpeg_magic = b'\xff\xd8'
print(f"  JPEG valid: {jpg[:2] == jpeg_magic}", flush=True)

print("  decode_to_png...", flush=True)
t0 = time.time()
png = jxlpy.decode_to_png(jxl)
print(f"  got {len(png)} bytes in {time.time()-t0:.2f}s", flush=True)
png_magic = b'\x89PNG\r\n\x1a\n'
print(f"  PNG valid: {png[:8] == png_magic}", flush=True)

print()
print("=== Test 2: JPEG file -> JXL (transcode) -> PNG ===", flush=True)
jpeg_path = Path("test_img/wallhaven-vpyekp.jpg")  # smallest JPEG
jpeg_bytes = jpeg_path.read_bytes()
print(f"  loaded {jpeg_path.name}: {len(jpeg_bytes)} bytes", flush=True)

print("  encode to jxl...", flush=True)
t0 = time.time()
jxl2 = jxlpy.encode(jpeg_bytes)
print(f"  encoded {len(jxl2)} bytes in {time.time()-t0:.2f}s", flush=True)

print("  decode_to_png...", flush=True)
t0 = time.time()
png2 = jxlpy.decode_to_png(jxl2)
print(f"  got {len(png2)} bytes in {time.time()-t0:.2f}s", flush=True)
print(f"  PNG valid: {png2[:8] == png_magic}", flush=True)

print("  decode_to_jpeg (should reconstruct original)...", flush=True)
t0 = time.time()
rt_jpeg = jxlpy.decode_to_jpeg(jxl2)
print(f"  got {len(rt_jpeg)} bytes in {time.time()-t0:.2f}s", flush=True)
print(f"  bit-exact match: {rt_jpeg == jpeg_bytes}", flush=True)

print()
print("=== Test 3: PNG file -> JXL -> JPEG ===", flush=True)
png_path = Path("test_img/png/wallhaven-49yekn.png")  # ~1.2MB
png_bytes = png_path.read_bytes()
print(f"  loaded {png_path.name}: {len(png_bytes)} bytes", flush=True)

print("  encode to jxl...", flush=True)
t0 = time.time()
jxl3 = jxlpy.encode(png_bytes)
print(f"  encoded {len(jxl3)} bytes in {time.time()-t0:.2f}s", flush=True)

print("  decode_to_jpeg...", flush=True)
t0 = time.time()
jpg3 = jxlpy.decode_to_jpeg(jxl3)
print(f"  got {len(jpg3)} bytes in {time.time()-t0:.2f}s", flush=True)
print(f"  JPEG valid: {jpg3[:2] == jpeg_magic}", flush=True)

print()
print("All done.")
