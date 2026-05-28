"""Check completed_manifest vs on-disk metadata for path collisions."""
import csv
import hashlib
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"


def slugify(text: str, fallback: str = "item", max_len: int = 70) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return (slug or fallback)[:max_len]


def stable_hash(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def seq_dir(row: dict) -> Path:
    path_hash = stable_hash(f"{row['dataset_root']}/{row['relative_path']}", length=8)
    start_tag = f"t{int(round(float(row['start_second']))):03d}s"
    name = f"sequence_{slugify(row['dataset_root'], max_len=18)}_{path_hash}_{start_tag}"
    cow_folder = f"cow_{row['cow_id']}_{row['cow_health_status'].lower()}"
    return OUT / "sequences" / row["cow_health_status"].lower() / cow_folder / name


def main() -> None:
    rows = list(csv.DictReader((OUT / "completed_manifest.csv").open(encoding="utf-8")))
    by_path: dict[str, list[str]] = {}
    missing: list[str] = []

    for row in rows:
        path = str(seq_dir(row))
        by_path.setdefault(path, []).append(row["sequence_index"])
        meta = Path(path) / "metadata.json"
        if not meta.is_file():
            missing.append(row["sequence_index"])
        elif json.loads(meta.read_text(encoding="utf-8")).get("sequence_index") != int(row["sequence_index"]):
            missing.append(row["sequence_index"])

    collisions = {k: v for k, v in by_path.items() if len(v) > 1}
    print(f"manifest rows: {len(rows)}")
    print(f"paths with >1 manifest row: {len(collisions)}")
    print(f"missing or overwritten indices: {len(missing)}")
    if collisions:
        print("\nExample collisions:")
        for path, indices in list(collisions.items())[:5]:
            print(f"  {Path(path).name}: {indices}")


if __name__ == "__main__":
    main()
