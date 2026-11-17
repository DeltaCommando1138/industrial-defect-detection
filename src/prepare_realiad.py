"""Create reproducible Real-IAD train/validation/test CSV files."""

import csv
import hashlib
import re
from pathlib import Path


DATA_DIR = Path("data/realiad_256")
OUTPUT_DIR = Path("data/splits")
SEED = "realiad-256-v1"


def choose_split(category: str, sample_id: str) -> str:
    """Keep every camera view of a sample in the same split."""
    key = f"{SEED}:{category}:{sample_id}".encode()
    value = int(hashlib.sha256(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "val"
    return "test"


def main() -> None:
    if not DATA_DIR.is_dir():
        raise SystemExit(f"Dataset not found: {DATA_DIR}")

    rows = {name: [] for name in ("train", "val", "test")}

    for image_path in sorted(DATA_DIR.rglob("*.jpg")):
        relative = image_path.relative_to(DATA_DIR)
        parts = relative.parts
        category, label = parts[0], parts[1]
        sample_id = image_path.parent.name
        defect_type = parts[2] if label == "NG" else ""
        camera_match = re.search(r"_(C\d+)_", image_path.name)
        mask_path = image_path.with_suffix(".png")

        row = {
            "image_path": image_path.as_posix(),
            "mask_path": mask_path.as_posix() if mask_path.exists() else "",
            "category": category,
            "label": label,
            "defect_type": defect_type,
            "sample_id": sample_id,
            "camera": camera_match.group(1) if camera_match else "",
        }
        rows[choose_split(category, sample_id)].append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "image_path",
        "mask_path",
        "category",
        "label",
        "defect_type",
        "sample_id",
        "camera",
    ]
    for split, split_rows in rows.items():
        output_path = OUTPUT_DIR / f"{split}.csv"
        with output_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"{split}: {len(split_rows):,} images -> {output_path}")


if __name__ == "__main__":
    main()
