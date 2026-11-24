"""Read-only MVTec LOCO AD combined dataset loading, split auditing, and ImageNet penalty corpus.

Strictly preserves Assignment 1/2/3 invariants:
  - Normal-only training (`train/good`, 1,778 pooled images across 5 categories)
  - Normal-only validation (`validation/good`, 305 pooled images)
  - Logical-only test evaluation (`test/good` [575] vs `test/logical_anomalies` [561] = 1,136)
  - `test/structural_anomalies` (432 images) are physically excluded with tripwire assertions
  - ZERO spatial augmentations (flips/rotations/crops corrupt logical layouts);
    appearance-only `RandomAugment` (brightness/contrast/saturation) is applied strictly
    to the AutoEncoder training view, matching EfficientAD / PUAD.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CATEGORIES: List[str] = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]

CATEGORY_LABELS: Dict[str, str] = {
    "breakfast_box": "Breakfast Box",
    "juice_bottle": "Juice Bottle",
    "pushpins": "Pushpins",
    "screw_bag": "Screw Bag",
    "splicing_connectors": "Splicing Connectors",
}

GOOD = "good"
LOGICAL = "logical_anomalies"
STRUCTURAL = "structural_anomalies"
EVALUATED_TYPES = (GOOD, LOGICAL)

EXPECTED_COUNTS: Dict[str, Dict[str, int]] = {
    "breakfast_box": {"train": 351, "val": 62, "test_good": 102, "test_logical": 83, "test_structural": 90},
    "juice_bottle": {"train": 335, "val": 54, "test_good": 94, "test_logical": 142, "test_structural": 94},
    "pushpins": {"train": 372, "val": 69, "test_good": 138, "test_logical": 91, "test_structural": 81},
    "screw_bag": {"train": 360, "val": 60, "test_good": 122, "test_logical": 137, "test_structural": 82},
    "splicing_connectors": {"train": 360, "val": 60, "test_good": 119, "test_logical": 108, "test_structural": 85},
}

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".JPEG")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_imagenet_normalization() -> transforms.Normalize:
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


class RandomAugment:
    """Appearance-only augmentation (verbatim from EfficientAD / PUAD). No spatial transforms."""

    def __call__(self, img: Image.Image) -> Image.Image:
        i_aug = int(torch.randint(1, 4, (1,)).item())
        lamda = float((torch.rand(1) * 0.4 + 0.8).item())
        if i_aug == 1:
            return transforms.functional.adjust_brightness(img, lamda)
        if i_aug == 2:
            return transforms.functional.adjust_contrast(img, lamda)
        return transforms.functional.adjust_saturation(img, lamda)


def resolve_category_dir(dataset_root: str | Path, category: str) -> Path:
    """Resolve `<root>/<cat>` or doubled `<root>/<cat>/<cat>` from MVTec tar extraction."""
    root = Path(dataset_root)
    for candidate in (root / category / category, root / category):
        if (candidate / "train" / "good").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find MVTec LOCO category '{category}' under '{dataset_root}'."
    )


def list_images(directory: str | Path) -> List[Path]:
    p = Path(directory)
    if not p.is_dir():
        return []
    return sorted(f for f in p.iterdir() if f.is_file() and f.suffix in IMAGE_SUFFIXES)


def audit_dataset(dataset_root: str | Path, categories: Sequence[str] = CATEGORIES) -> Dict:
    """Verify dataset accessibility and assert exact split counts across all 5 categories."""
    per_cat: Dict[str, Dict[str, int]] = {}
    totals = {
        "n_train_normal": 0,
        "n_validation_normal": 0,
        "n_test_good": 0,
        "n_test_logical": 0,
        "n_test_structural_excluded": 0,
    }
    warnings: List[str] = []

    for cat in categories:
        cat_dir = resolve_category_dir(dataset_root, cat)
        counts = {
            "n_train": len(list_images(cat_dir / "train" / GOOD)),
            "n_val": len(list_images(cat_dir / "validation" / GOOD)),
            "n_test_good": len(list_images(cat_dir / "test" / GOOD)),
            "n_test_logical": len(list_images(cat_dir / "test" / LOGICAL)),
            "n_test_structural": len(list_images(cat_dir / "test" / STRUCTURAL)),
        }
        per_cat[cat] = counts
        totals["n_train_normal"] += counts["n_train"]
        totals["n_validation_normal"] += counts["n_val"]
        totals["n_test_good"] += counts["n_test_good"]
        totals["n_test_logical"] += counts["n_test_logical"]
        totals["n_test_structural_excluded"] += counts["n_test_structural"]

        exp = EXPECTED_COUNTS.get(cat)
        if exp:
            for k_exp, k_got in (
                ("train", "n_train"),
                ("val", "n_val"),
                ("test_good", "n_test_good"),
                ("test_logical", "n_test_logical"),
            ):
                if counts[k_got] != exp[k_exp]:
                    warnings.append(f"{cat}: {k_got}={counts[k_got]} != expected {exp[k_exp]}")

        if counts["n_test_good"] == 0 or counts["n_test_logical"] == 0:
            raise RuntimeError(f"Invalid test split for {cat}: {counts}")

    totals["n_images_final_auroc"] = totals["n_test_good"] + totals["n_test_logical"]
    totals["structural_included_in_final_auroc"] = 0
    return {"dataset_root": str(dataset_root), "per_category": per_cat, "pooled": totals, "warnings": warnings}


class CombinedNormalImages(Dataset):
    """Pooled normal images (`train/good` or `validation/good`), optionally cached in RAM."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        img_size: int = 256,
        normalize: Optional[Callable] = None,
        augment: bool = False,
        categories: Sequence[str] = CATEGORIES,
        cache_in_ram: bool = True,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError(f"split must be 'train' or 'validation', got {split!r}")
        self.img_size = img_size
        self.normalize = normalize
        self.augment = augment
        self.cache_in_ram = cache_in_ram
        self._augmenter = RandomAugment()
        self._to_tensor = transforms.ToTensor()
        self._resize = transforms.Resize((img_size, img_size))

        self.cache: List[Image.Image] = []
        self.categories_per_item: List[str] = []
        self.paths: List[Path] = []
        for cat in categories:
            cat_dir = resolve_category_dir(dataset_root, cat)
            imgs = list_images(cat_dir / split / GOOD)
            if not imgs:
                raise FileNotFoundError(f"No images found in {cat_dir / split / GOOD}")
            for p in imgs:
                if cache_in_ram:
                    with Image.open(p) as im:
                        self.cache.append(self._resize(im.convert("RGB")))
                self.categories_per_item.append(cat)
                self.paths.append(p)

    def __len__(self) -> int:
        return len(self.paths)

    def _finalize(self, img: Image.Image) -> torch.Tensor:
        t = self._to_tensor(img)
        return self.normalize(t) if self.normalize is not None else t

    def __getitem__(self, index: int):
        if self.cache_in_ram:
            img = self.cache[index]
        else:
            with Image.open(self.paths[index]) as im:
                img = self._resize(im.convert("RGB"))
        clean = self._finalize(img)
        if not self.augment:
            return clean
        return clean, self._finalize(self._augmenter(img))


@dataclass
class CombinedTestItem:
    path: Path
    label: int          # 0 = good, 1 = logical_anomalies
    anomaly_type: str   # "good" | "logical_anomalies"
    category: str
    unique_name: str


def build_combined_logical_test_index(
    dataset_root: str | Path,
    categories: Sequence[str] = CATEGORIES,
) -> List[CombinedTestItem]:
    """Build pooled logical-only test index. Structural anomalies are physically excluded."""
    items: List[CombinedTestItem] = []
    for cat in categories:
        cat_dir = resolve_category_dir(dataset_root, cat)
        for p in list_images(cat_dir / "test" / GOOD):
            items.append(CombinedTestItem(p, 0, GOOD, cat, f"{cat}/{p.name}"))
        for p in list_images(cat_dir / "test" / LOGICAL):
            items.append(CombinedTestItem(p, 1, LOGICAL, cat, f"{cat}/{p.name}"))

    leaked = [i for i in items if i.anomaly_type == STRUCTURAL]
    if leaked:
        raise AssertionError(f"TRIPWIRE: {len(leaked)} structural images leaked into test index!")
    if not items:
        raise FileNotFoundError(f"No test images found under {dataset_root}")
    return items


class CombinedTestImages(Dataset):
    """Streams test images once per evaluation (never cached during training)."""

    def __init__(
        self,
        items: Sequence[CombinedTestItem],
        img_size: int = 256,
        normalize: Optional[Callable] = None,
    ) -> None:
        self.items = list(items)
        ops = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        if normalize is not None:
            ops.append(normalize)
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str, str]:
        item = self.items[index]
        with Image.open(item.path) as im:
            tensor = self.transform(im.convert("RGB"))
        return tensor, item.label, item.anomaly_type, item.category


class PenaltyImages(Dataset):
    """Out-of-distribution ImageNet corpus for EfficientAD's L_penalty term."""

    def __init__(self, root: str | Path, normalize: Callable, img_size: int = 256) -> None:
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"ImageNet penalty corpus directory not found: {root}")
        self.paths = sorted(p for p in root_path.rglob("*") if p.is_file() and p.suffix in IMAGE_SUFFIXES)
        if not self.paths:
            raise FileNotFoundError(f"No images found in ImageNet penalty directory: {root}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.RandomGrayscale(p=0.3),
                transforms.CenterCrop((img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as im:
            return self.transform(im.convert("RGB"))
