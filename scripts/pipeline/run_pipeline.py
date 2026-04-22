# scripts/pipeline/run_pipeline.py

import argparse
import os
import shutil
import subprocess
import sys
import time
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)

PYTHON = sys.executable
STAGING_DIR = Path("data/staging")
STAGING_BACKUP = Path("data/staging_full_backup")

DRIVE_INDEX_PATH = Path("data/outputs/drive_index.csv")
CHUNKS_DIR = Path("data/outputs/chunks")

# Files cleared between chunks so a previous chunk's state doesn't break the next one.
PER_CHUNK_RESET = [
    "data/outputs/speciesnet_results.json",
    "data/outputs/speciesnet_results.csv",
    "data/outputs/speciesnet_review.csv",
    "data/outputs/manifest.csv",
    "data/outputs/manifest_new.csv",
    "data/outputs/metadata.csv",
    "data/outputs/ml_outputs.csv",
    "data/outputs/.download_progress.csv",
    "data/outputs/download_log.csv",
]


def parse_id_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


def make_run_tag(drive_root: str | None, start_folders: str | None) -> str:
    ids = parse_id_list(start_folders)
    if ids:
        return "_".join(ids)
    return drive_root or ""


def infer_index_path(args: argparse.Namespace) -> str:
    if args.index:
        return args.index
    if args.out_index:
        return args.out_index
    if args.per_folder:
        tag = make_run_tag(args.drive_root, args.start_folders)
        if tag:
            return str(Path("data/outputs") / f"drive_index_{tag}.csv")
    return "data/outputs/drive_index.csv"


def ensure_python_311() -> None:
    if sys.version_info[:2] != (3, 11):
        print("ERROR: This pipeline must be run with Python 3.11 to match dependencies.")
        print(f"Current Python: {sys.version}")
        sys.exit(1)


def run_step(name: str, cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print(f"STEP: {name}")
    print("CMD:", " ".join(cmd))
    print("=" * 80)
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        print(f"\nERROR: Step failed: {name} (exit code {res.returncode})")
        sys.exit(res.returncode)


def copy_staging_backup() -> None:
    if not STAGING_DIR.exists():
        return
    STAGING_BACKUP.parent.mkdir(parents=True, exist_ok=True)
    if STAGING_BACKUP.exists():
        shutil.rmtree(STAGING_BACKUP)
    shutil.copytree(STAGING_DIR, STAGING_BACKUP)


def restore_staging_backup() -> None:
    if not STAGING_BACKUP.exists():
        return
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    shutil.copytree(STAGING_BACKUP, STAGING_DIR)


def manifest_has_rows(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    try:
        with p.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for _ in reader:
                return True
    except Exception:
        return False
    return False


def count_index_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        if next(reader, None) is None:
            return 0
        return sum(1 for _ in reader)


def split_index_into_chunks(src: Path, chunks_dir: Path, chunk_size: int) -> list[Path]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with src.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return paths

        def _flush(buf: list[list[str]], idx: int) -> Path:
            out = chunks_dir / f"chunk_{idx:04d}_index.csv"
            with out.open("w", newline="", encoding="utf-8") as wf:
                w = csv.writer(wf)
                w.writerow(header)
                w.writerows(buf)
            return out

        buffer: list[list[str]] = []
        n = 0
        for row in reader:
            buffer.append(row)
            if len(buffer) >= chunk_size:
                n += 1
                paths.append(_flush(buffer, n))
                buffer = []
        if buffer:
            n += 1
            paths.append(_flush(buffer, n))
    return paths


def append_csv_rows(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    first_write = not dst.exists()
    with src.open("r", newline="", encoding="utf-8") as sf:
        reader = csv.reader(sf)
        header = next(reader, None)
        if not header:
            return
        mode = "w" if first_write else "a"
        with dst.open(mode, newline="", encoding="utf-8") as df:
            w = csv.writer(df)
            if first_write:
                w.writerow(header)
            for row in reader:
                w.writerow(row)


def clear_per_chunk_state() -> None:
    for p in PER_CHUNK_RESET:
        path = Path(p)
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)


def run_chunked_auto(args: argparse.Namespace, chunk_paths: list[Path]) -> None:
    """Run auto-mode pipeline one chunk at a time. Assumes the full index
    already exists on disk at args.index."""
    cumulative_manifest = CHUNKS_DIR / "cumulative_manifest.csv"
    cumulative_metadata = CHUNKS_DIR / "cumulative_metadata.csv"
    cumulative_ml_outputs = CHUNKS_DIR / "cumulative_ml_outputs.csv"

    for p in (cumulative_manifest, cumulative_metadata, cumulative_ml_outputs):
        if p.exists():
            p.unlink()

    total = len(chunk_paths)
    for i, chunk_index in enumerate(chunk_paths, start=1):
        print(f"\n{'#' * 80}")
        print(f"# CHUNK {i}/{total}: {chunk_index.name}")
        print(f"{'#' * 80}")

        clear_per_chunk_state()

        run_step(f"[Chunk {i}/{total}] Download Images",
                 [PYTHON, "scripts/pipeline/download_drive.py", "--index", str(chunk_index)])

        run_step(f"[Chunk {i}/{total}] Create Manifest",
                 [PYTHON, "scripts/pipeline/make_manifest.py",
                  "--batch_size", str(args.batch_size)])

        manifest_path = "data/outputs/manifest.csv"

        run_step(f"[Chunk {i}/{total}] Extract Metadata (EXIF)",
                 [PYTHON, "scripts/pipeline/extract_metadata.py", "--manifest", manifest_path])
        run_step(f"[Chunk {i}/{total}] Run SpeciesNet",
                 [PYTHON, "scripts/ml/run_speciesnet.py"])
        run_step(f"[Chunk {i}/{total}] Postprocess SpeciesNet",
                 [PYTHON, "scripts/ml/postprocess_speciesnet.py",
                  "--burst_window", str(args.ml_burst_window)])
        run_step(f"[Chunk {i}/{total}] Parse ML Results",
                 [PYTHON, "scripts/ml/run_inference.py", "--provider", "speciesnet"])
        run_step(f"[Chunk {i}/{total}] Extract Metadata (merge ML)",
                 [PYTHON, "scripts/pipeline/extract_metadata.py", "--manifest", manifest_path])

        chunk_archive = CHUNKS_DIR / f"chunk_{i:04d}"
        chunk_archive.mkdir(parents=True, exist_ok=True)
        for fname in ("manifest.csv", "metadata.csv", "ml_outputs.csv",
                      "speciesnet_results.csv", "speciesnet_results.json",
                      "speciesnet_review.csv"):
            src = Path("data/outputs") / fname
            if src.exists():
                shutil.copy2(src, chunk_archive / fname)

        append_csv_rows(Path("data/outputs/manifest.csv"), cumulative_manifest)
        append_csv_rows(Path("data/outputs/metadata.csv"), cumulative_metadata)
        append_csv_rows(Path("data/outputs/ml_outputs.csv"), cumulative_ml_outputs)

    print(f"\n{'#' * 80}")
    print(f"# FINALIZING: merging {total} chunks into final output")
    print(f"{'#' * 80}")

    if cumulative_manifest.exists():
        shutil.copy2(cumulative_manifest, "data/outputs/manifest.csv")
    if cumulative_metadata.exists():
        shutil.copy2(cumulative_metadata, "data/outputs/metadata.csv")
    if cumulative_ml_outputs.exists():
        shutil.copy2(cumulative_ml_outputs, "data/outputs/ml_outputs.csv")

    run_step("Generate Output CSVs", [
        PYTHON, "scripts/pipeline/make_output.py",
        "--manifest", "data/outputs/manifest.csv",
        "--burst_seconds", str(args.burst_seconds),
        "--burst_export", args.burst_export,
    ])

    if args.upload:
        upload_cmd = [PYTHON, "scripts/drive_upload/upload_to_drive.py"]
        if args.overwrite:
            upload_cmd += ["--overwrite"]
        if args.upload_target:
            upload_cmd += ["--target_folder", args.upload_target]
        run_step("Upload Results to Drive", upload_cmd)


def build_steps(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = []

    if args.mode == "auto":
        if not args.index:
            index_cmd = [PYTHON, "scripts/pipeline/build_index.py"]
            if args.drive_root:
                index_cmd += ["--drive_root", args.drive_root]
            if args.start_folders:
                index_cmd += ["--start_folders", args.start_folders]
            if args.out_index:
                index_cmd += ["--out", args.out_index]
            if args.per_folder:
                index_cmd += ["--per_folder"]
            if args.resume:
                index_cmd += ["--resume"]
            if args.max_files is not None:
                index_cmd += ["--max_files", str(args.max_files)]
            steps.append(("Index Drive", index_cmd))

        download_cmd = [PYTHON, "scripts/pipeline/download_drive.py", "--index", infer_index_path(args)]
        if args.resume:
            download_cmd += ["--resume"]
        if args.max_downloads is not None:
            download_cmd += ["--max_downloads", str(args.max_downloads)]
        steps.append(("Download Images", download_cmd))

    if args.mode == "auto":
        steps.append((
            "Create Manifest (new-only)",
            [PYTHON, "scripts/pipeline/make_manifest.py",
             "--cache", args.cache,
             "--new_out", args.new_manifest,
             "--write_new_only",
             "--batch_size", str(args.batch_size)]
        ))
    else:
        steps.append(("Create Manifest", [
            PYTHON, "scripts/pipeline/make_manifest.py",
            "--batch_size", str(args.batch_size)
        ]))

    manifest_to_process = args.new_manifest if (args.mode == "auto" and args.use_new_manifest_for_outputs) else "data/outputs/manifest.csv"
    if args.mode == "auto" and args.use_new_manifest_for_outputs and not manifest_has_rows(manifest_to_process):
        manifest_to_process = "data/outputs/manifest.csv"

    # Extract EXIF metadata before SpeciesNet postprocessing so that
    # postprocess_speciesnet.py has metadata.csv available for burst timestamp grouping.
    steps.append(("Extract Metadata (EXIF)", [PYTHON, "scripts/pipeline/extract_metadata.py", "--manifest", manifest_to_process]))

    steps.append(("Run SpeciesNet", [PYTHON, "scripts/ml/run_speciesnet.py"]))
    steps.append(("Postprocess SpeciesNet", [PYTHON, "scripts/ml/postprocess_speciesnet.py", "--burst_window", str(args.ml_burst_window)]))
    steps.append(("Parse ML Results", [PYTHON, "scripts/ml/run_inference.py", "--provider", "speciesnet"]))

    # Re-run extract_metadata now that ml_outputs.csv exists so the final
    # metadata.csv has ML columns (species, has_animal, model_certainty) merged in.
    steps.append(("Extract Metadata (merge ML)", [PYTHON, "scripts/pipeline/extract_metadata.py", "--manifest", manifest_to_process]))

    steps.append(("Generate Output CSVs", [
        PYTHON, "scripts/pipeline/make_output.py",
        "--manifest", manifest_to_process,
        "--burst_seconds", str(args.burst_seconds),
        "--burst_export", args.burst_export
    ]))

    if args.upload:
        upload_cmd = [PYTHON, "scripts/drive_upload/upload_to_drive.py"]
        if args.overwrite:
            upload_cmd += ["--overwrite"]
        if args.upload_target:
            upload_cmd += ["--target_folder", args.upload_target]
        steps.append(("Upload Results to Drive", upload_cmd))

    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the wildlife pipeline end-to-end.")

    parser.add_argument("--mode", default="auto", choices=["auto", "manual"],
                        help="Auto: index+download+process NEW images with caching. Manual: process selected folder only.")
    parser.add_argument("--folder", default=None,
                        help="Manual mode only: local folder path of images to stage.")

    parser.add_argument("--drive_root", default=None, help="Drive root folder ID (auto mode).")
    parser.add_argument("--start_folders", default=None, help="Comma-separated folder IDs to start indexing from.")
    parser.add_argument("--index", default=None, help="Use an existing drive index CSV instead of building one.")
    parser.add_argument("--out_index", default=None, help="Output drive index CSV path.")
    parser.add_argument("--per_folder", action="store_true", help="Name index output using folder tag.")
    parser.add_argument("--resume", action="store_true", help="Resume download/index (auto mode).")
    parser.add_argument("--max_files", type=int, default=None, help="Max files to index (auto mode).")
    parser.add_argument("--max_downloads", type=int, default=None, help="Max files to download (auto mode).")

    parser.add_argument("--batch_size", type=int, default=0, help="Optional batch manifest size (>0 to write batches).")
    parser.add_argument("--cache", default="data/outputs/cache/processed_file_ids.txt", help="Cache file to track processed file_ids.")
    parser.add_argument("--new_manifest", default="data/outputs/manifest_new.csv", help="Output path for new-only manifest.")
    parser.add_argument("--use_new_manifest_for_outputs", action="store_true",
                        help="Use new-only manifest for metadata+output steps (falls back if empty).")

    parser.add_argument("--ml_burst_window", type=int, default=300, help="Burst window seconds for SpeciesNet postprocess.")
    parser.add_argument("--burst_seconds", type=int, default=300, help="Burst duration seconds for output.")
    parser.add_argument("--burst_export", default="all", choices=["all", "first", "middle", "last"],
                        help="Which burst images to export in output.")

    parser.add_argument("--chunk_size", type=int, default=0,
                        help="Split large runs into chunks of this many images. 0 = off. "
                             "If total indexed > chunk_size, the pipeline processes chunks "
                             "sequentially to cap disk usage.")

    parser.add_argument("--upload", action="store_true",
                        help="Upload output CSVs to Google Drive after pipeline completes (production).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing Drive CSVs instead of appending (used with --upload).")
    parser.add_argument("--upload_target", default=None,
                        help="Drive folder ID. When set with --upload, every by_location CSV "
                             "is uploaded to this single folder instead of being mapped via "
                             "drive_index.csv. Intended for --mode manual where there is no Drive index.")

    args = parser.parse_args()

    ensure_python_311()

    if args.mode == "manual":
        if not args.folder:
            print("ERROR: --folder is required in manual mode.")
            sys.exit(2)
        src = Path(args.folder)
        if not src.exists():
            print(f"ERROR: Folder does not exist: {src}")
            sys.exit(2)
        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        copied = 0
        for p in src.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                continue
            rel = p.relative_to(src)
            dst = STAGING_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            copied += 1
        print(f"Copied {copied} image(s) from {src} → {STAGING_DIR} (preserving subfolder structure).")

    if args.mode == "manual":
        copy_staging_backup()

    start = time.time()

    if args.mode == "auto" and args.chunk_size > 0:
        if not args.index:
            index_cmd = [PYTHON, "scripts/pipeline/build_index.py"]
            if args.drive_root:
                index_cmd += ["--drive_root", args.drive_root]
            if args.start_folders:
                index_cmd += ["--start_folders", args.start_folders]
            if args.out_index:
                index_cmd += ["--out", args.out_index]
            if args.per_folder:
                index_cmd += ["--per_folder"]
            if args.resume:
                index_cmd += ["--resume"]
            if args.max_files is not None:
                index_cmd += ["--max_files", str(args.max_files)]
            run_step("Index Drive", index_cmd)
            args.index = infer_index_path(args)

        total = count_index_rows(Path(args.index))
        print(f"\nTotal images in index: {total}")

        if total > args.chunk_size:
            print(f"Chunking activated: {total} > chunk_size {args.chunk_size}")
            chunk_paths = split_index_into_chunks(Path(args.index), CHUNKS_DIR, args.chunk_size)
            print(f"Split into {len(chunk_paths)} chunks of up to {args.chunk_size} images each")
            run_chunked_auto(args, chunk_paths)
        else:
            print(f"Below chunk threshold ({args.chunk_size}): running single-pass")
            steps = build_steps(args)
            for name, cmd in steps:
                run_step(name, cmd)
    else:
        steps = build_steps(args)
        for name, cmd in steps:
            run_step(name, cmd)

    elapsed = time.time() - start
    print(f"\nDONE in {elapsed/60:.1f} minutes")

    if args.mode == "manual":
        restore_staging_backup()
    else:
        # Auto mode: clear staging after a successful run to free disk space.
        # Re-downloads are prevented by data/outputs/.download_progress.csv.
        if STAGING_DIR.exists():
            shutil.rmtree(STAGING_DIR)
            print(f"Cleared {STAGING_DIR} (outputs preserved in data/outputs/)")


if __name__ == "__main__":
    main()