"""Pack a compact, deterministic 256x256 ImageNet penalty archive for fast Google Drive / Colab Pro upload.

Why: Uploading 50,000 loose ImageNet images (6.25 GB) to Google Drive is slow and reading loose
files over Drive FUSE bottlenecks GPU training. Packing a deterministic seed-42 subset (default
10,000 images resized to 512x512 JPEG or 256x256 JPEG) produces a single ~150-300 MB zip that
uploads quickly and extracts to Colab's local NVMe SSD (/content/) in seconds.
"""

from __future__ import annotations

import argparse
import io
import random
import zipfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".JPEG")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=r"C:\Users\anees\Desktop\Sem 5\DL\Dataset\ImageNet")
    ap.add_argument("--out", default=r"C:\Users\anees\Desktop\Sem 5\DL\Endsem\imagenet_penalty_10k.zip")
    ap.add_argument("--count", type=int, default=10000)
    ap.add_argument("--size", type=int, default=320, help="Resize shorter/square side before zipping")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    all_imgs = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix in IMAGE_SUFFIXES)
    if not all_imgs:
        raise FileNotFoundError(f"No images found under {src}")

    rng = random.Random(args.seed)
    chosen = rng.sample(all_imgs, min(args.count, len(all_imgs)))
    print(f"Packing {len(chosen):,} / {len(all_imgs):,} images (seed={args.seed}, size={args.size}x{args.size}) -> {args.out}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for i, p in enumerate(tqdm(chosen, desc="Packing")):
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB").resize((args.size, args.size), Image.BICUBIC)
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=88)
                    zf.writestr(f"imagenet_penalty/{i:05d}.jpg", buf.getvalue())
            except Exception:
                continue

    mb = out_path.stat().st_size / (1024 ** 2)
    print(f"Done! Archive size: {mb:.1f} MB -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
