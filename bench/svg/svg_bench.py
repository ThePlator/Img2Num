"""
Benchmark Img2Num's SVG output size, emit time and render fidelity.

Workflow (from the repository root, see ``bench/svg/README.md``)::

    # 1. Freeze the K-Means labels for every corpus image (once per machine).
    uv run --group bench bench/svg/svg_bench.py prepare

    # 2. Emit SVGs with the code you want to measure.
    git switch dev
    uv run --group bench bench/svg/svg_bench.py run bench/svg/results/before
    git switch my-branch
    uv run --group bench bench/svg/svg_bench.py run bench/svg/results/after

    # 3. Compare: sizes, timings and a pixel diff of both renders.
    uv run --group bench bench/svg/svg_bench.py compare \
        bench/svg/results/before bench/svg/results/after

K-Means picks its initial centroids randomly, so running the full pipeline
twice gives two different SVGs. ``prepare`` runs the bilateral filter and
K-Means once and caches the labels; ``run`` only calls ``labels_to_svg`` on
those cached labels. Differences between two runs are then caused by the
code under test, not by clustering noise.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = BENCH_DIR / "corpus"
DEFAULT_CACHE = BENCH_DIR / ".cache"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Pipeline parameters. These mirror ImageToSvgConfig's defaults so the
# benchmark measures what users get out of the box from image_to_svg().
SIGMA_SPATIAL = 3.0
SIGMA_RANGE = 50.0
KMEANS_K = 16
KMEANS_MAX_ITER = 100
COLOR_SPACE = 0  # CIE LAB
MIN_AREA = 100
MIN_THICKNESS = 0

# A pixel counts as "changed" when any channel differs by more than this
# (0-255). Rasterizers anti-alias edges, so tiny coordinate changes cause
# small per-pixel deltas that are invisible.
DEFAULT_TOLERANCE = 8


def load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.ascontiguousarray(np.asarray(img.convert("RGBA"), dtype=np.uint8))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_images(corpus: Path) -> list[Path]:
    return sorted(p for p in corpus.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def git_describe() -> str:
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--exclude", "*"],
            cwd=BENCH_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def import_img2num():
    try:
        import img2num
    except ImportError:
        sys.exit(
            "img2num is not importable. Run this script with "
            "`uv run --group bench ...` from the repository root, or install "
            "the Python package into your environment first."
        )
    return img2num


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def numpy_labels(rgba: np.ndarray, k: int, max_iter: int, seed: int = 0) -> np.ndarray:
    """Median pre-smoothing + seeded k-means++ / Lloyd in RGB.

    A fallback for machines where img2num's own filter / K-Means misbehave
    (e.g. no usable GPU adapter). It never calls img2num and is fully
    deterministic, so every machine gets the same labels. It approximates the
    real pipeline; prefer the default labeler when it works.
    """
    from PIL import ImageFilter

    height, width = rgba.shape[:2]
    smooth = Image.fromarray(rgba[..., :3]).filter(ImageFilter.MedianFilter(5))
    px = np.asarray(smooth, dtype=np.float32).reshape(-1, 3)
    rng = np.random.default_rng(seed)

    px_sq = (px**2).sum(axis=1, keepdims=True)

    def nearest(centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # |p - c|^2 = |p|^2 - 2 p.c + |c|^2, without an (N, k, 3) temporary.
        d = px_sq - 2 * px @ centroids.T + (centroids**2).sum(axis=1)
        idx = d.argmin(axis=1)
        return idx, np.maximum(d[np.arange(len(px)), idx], 0)

    centroids = px[rng.integers(len(px))][None, :]
    for _ in range(1, k):
        _, dist = nearest(centroids)
        total = dist.sum()
        if total == 0:  # fewer distinct colors than k
            break
        pick = rng.choice(len(px), p=dist / total)
        centroids = np.vstack([centroids, px[pick]])

    labels, _ = nearest(centroids)
    for _ in range(max_iter):
        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, px)
        counts = np.bincount(labels, minlength=len(centroids))[:, None]
        centroids = np.where(counts > 0, sums / np.maximum(counts, 1), centroids)
        new_labels, _ = nearest(centroids)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return labels.reshape(height, width).astype(np.int32)


def cmd_prepare(args: argparse.Namespace) -> None:
    img2num = import_img2num()
    args.cache.mkdir(parents=True, exist_ok=True)

    for path in corpus_images(args.corpus):
        out = args.cache / f"{path.stem}.npz"
        digest = sha256(path)
        if out.exists() and not args.force:
            with np.load(out) as cached:
                labeler = str(cached["labeler"]) if "labeler" in cached else ""
                if str(cached["source_sha256"]) == digest and labeler == args.labeler:
                    print(f"  {path.name:32s} cached")
                    continue

        t0 = time.perf_counter()
        rgba = load_rgba(path)
        if args.labeler == "img2num":
            filtered = img2num.bilateral_filter(
                rgba.copy(), SIGMA_SPATIAL, SIGMA_RANGE, COLOR_SPACE
            )
            _, labels = img2num.kmeans(
                filtered, KMEANS_K, KMEANS_MAX_ITER, COLOR_SPACE
            )
        else:
            labels = numpy_labels(rgba, KMEANS_K, KMEANS_MAX_ITER)
        np.savez_compressed(
            out,
            labels=np.asarray(labels, dtype=np.int32),
            labeler=np.array(args.labeler),
            source_sha256=np.array(digest),
        )
        print(f"  {path.name:32s} {time.perf_counter() - t0:6.2f}s")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@dataclass
class ImageResult:
    name: str
    width: int
    height: int
    svg_bytes: int
    svg_gzip_bytes: int
    path_count: int
    emit_ms_median: float
    emit_ms_min: float
    deterministic: bool


def emit_svg(img2num, rgba: np.ndarray, labels: np.ndarray) -> str:
    """The call under test. Later optimizations add their options here."""
    return img2num.labels_to_svg(rgba, labels, MIN_AREA, MIN_THICKNESS)


def cmd_run(args: argparse.Namespace) -> None:
    img2num = import_img2num()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[ImageResult] = []
    for path in corpus_images(args.corpus):
        if args.only and path.stem not in args.only:
            continue
        cache = args.cache / f"{path.stem}.npz"
        if not cache.exists():
            sys.exit(f"No cached labels for {path.name}. Run `prepare` first.")
        with np.load(cache) as cached:
            if str(cached["source_sha256"]) != sha256(path):
                sys.exit(f"{path.name} changed since `prepare`. Re-run `prepare`.")
            labels = np.ascontiguousarray(cached["labels"])
        rgba = load_rgba(path)
        height, width = rgba.shape[:2]

        svg = emit_svg(img2num, rgba, labels)  # warm-up; this output is kept
        timings = []
        deterministic = True
        for _ in range(args.repeat):
            t0 = time.perf_counter_ns()
            svg_i = emit_svg(img2num, rgba, labels)
            timings.append((time.perf_counter_ns() - t0) / 1e6)
            deterministic &= svg_i == svg

        data = svg.encode("utf-8")
        (out_dir / f"{path.stem}.svg").write_bytes(data)
        r = ImageResult(
            name=path.stem,
            width=width,
            height=height,
            svg_bytes=len(data),
            svg_gzip_bytes=len(gzip.compress(data, compresslevel=9, mtime=0)),
            path_count=svg.count("<path"),
            emit_ms_median=statistics.median(timings),
            emit_ms_min=min(timings),
            deterministic=deterministic,
        )
        results.append(r)
        print(
            f"  {r.name:30s} {r.svg_bytes:>10,d} B  gz {r.svg_gzip_bytes:>9,d} B  "
            f"{r.path_count:>6d} paths  {r.emit_ms_median:8.1f} ms"
            + ("" if deterministic else "  (output varies between runs)")
        )

    meta = {
        "git": git_describe(),
        "img2num_version": getattr(img2num, "__version__", "unknown"),
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "repeat": args.repeat,
        "params": {
            "sigma_spatial": SIGMA_SPATIAL,
            "sigma_range": SIGMA_RANGE,
            "kmeans_k": KMEANS_K,
            "kmeans_max_iter": KMEANS_MAX_ITER,
            "color_space": COLOR_SPACE,
            "min_area": MIN_AREA,
            "min_thickness": MIN_THICKNESS,
        },
    }
    payload = {"meta": meta, "images": [asdict(r) for r in results]}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    total = sum(r.svg_bytes for r in results)
    total_gz = sum(r.svg_gzip_bytes for r in results)
    print(f"Total: {total:,d} B (gzip {total_gz:,d} B) -> {out_dir}")
    unstable = [r.name for r in results if not r.deterministic]
    if unstable:
        print(
            "Warning: labels_to_svg gave different SVGs for identical input on: "
            + ", ".join(unstable)
            + ". Their render diffs include run-to-run noise; see README."
        )


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def rasterize(svg_path: Path) -> np.ndarray:
    import resvg_py

    png = resvg_py.svg_to_bytes(svg_path=str(svg_path), background="#ffffff")
    with Image.open(io.BytesIO(bytes(png))) as img:
        return np.asarray(img.convert("RGB"), dtype=np.int16)


@dataclass
class RenderDiff:
    mean_abs: float
    max_abs: int
    changed_pct: float


def render_diff(a: Path, b: Path, tolerance: int) -> RenderDiff:
    ra, rb = rasterize(a), rasterize(b)
    if ra.shape != rb.shape:
        # A changed canvas size is always a visible change.
        return RenderDiff(float("inf"), 255, 100.0)
    diff = np.abs(ra - rb)
    per_px = diff.max(axis=-1)
    return RenderDiff(
        mean_abs=float(diff.mean()),
        max_abs=int(diff.max()),
        changed_pct=float((per_px > tolerance).mean() * 100),
    )


def is_stable(*results: dict) -> bool:
    return all(r.get("deterministic", True) for r in results)


def load_run(run_dir: Path) -> tuple[dict, dict[str, dict]]:
    path = run_dir / "results.json"
    if not path.exists():
        sys.exit(f"{path} not found. Run `run {run_dir}` first.")
    payload = json.loads(path.read_text())
    return payload["meta"], {r["name"]: r for r in payload["images"]}


def pct(before: float, after: float) -> str:
    if before == 0:
        return "n/a"
    return f"{(after - before) / before * 100:+.1f}%"


def kib(n: int) -> str:
    return f"{n / 1024:,.1f}"


def cmd_compare(args: argparse.Namespace) -> None:
    meta_a, run_a = load_run(args.before)
    meta_b, run_b = load_run(args.after)
    names = [n for n in run_a if n in run_b]
    missing = sorted(set(run_a) ^ set(run_b))
    tol = args.tolerance

    lines = [
        f"Before: `{meta_a['git']}` · After: `{meta_b['git']}` · "
        f"{meta_b['machine']}, emit time = best of {meta_b['repeat']} runs · "
        f"changed px = any channel differs by more than {tol}/255",
        "",
        "| Image | Size (KiB) | Δ size | Gzip (KiB) | Δ gzip | Paths "
        "| Emit (ms) | Δ time | Mean diff | Max diff | Changed px |",
        "| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]

    tot = {"a": 0, "b": 0, "ga": 0, "gb": 0, "ta": 0.0, "tb": 0.0}
    worst = RenderDiff(0.0, 0, 0.0)
    # Worst changed-pixel share among images whose output is deterministic;
    # the --max-changed-pct gate uses this so run-to-run noise can't trip it.
    worst_stable_pct = 0.0
    for name in names:
        a, b = run_a[name], run_b[name]
        d = render_diff(
            args.before / f"{name}.svg", args.after / f"{name}.svg", tol
        )
        worst = RenderDiff(
            max(worst.mean_abs, d.mean_abs),
            max(worst.max_abs, d.max_abs),
            max(worst.changed_pct, d.changed_pct),
        )
        if is_stable(run_a[name], run_b[name]):
            worst_stable_pct = max(worst_stable_pct, d.changed_pct)
        tot["a"] += a["svg_bytes"]
        tot["b"] += b["svg_bytes"]
        tot["ga"] += a["svg_gzip_bytes"]
        tot["gb"] += b["svg_gzip_bytes"]
        tot["ta"] += a["emit_ms_min"]
        tot["tb"] += b["emit_ms_min"]
        mark = "" if is_stable(a, b) else " †"
        paths = (
            str(b["path_count"])
            if a["path_count"] == b["path_count"]
            else f"{a['path_count']} → {b['path_count']}"
        )
        lines.append(
            f"| {name}{mark} | {kib(b['svg_bytes'])} "
            f"| {pct(a['svg_bytes'], b['svg_bytes'])} "
            f"| {kib(b['svg_gzip_bytes'])} "
            f"| {pct(a['svg_gzip_bytes'], b['svg_gzip_bytes'])} | {paths} "
            f"| {b['emit_ms_min']:.1f} "
            f"| {pct(a['emit_ms_min'], b['emit_ms_min'])} "
            f"| {d.mean_abs:.3f} | {d.max_abs} | {d.changed_pct:.3f}% |"
        )

    lines.append(
        f"| **Total** | **{kib(tot['b'])}** | **{pct(tot['a'], tot['b'])}** "
        f"| **{kib(tot['gb'])}** | **{pct(tot['ga'], tot['gb'])}** | "
        f"| **{tot['tb']:.1f}** | **{pct(tot['ta'], tot['tb'])}** "
        f"| {worst.mean_abs:.3f} (worst) | {worst.max_abs} (worst) "
        f"| {worst.changed_pct:.3f}% (worst) |"
    )
    lines += [
        "",
        f"**Summary:** {kib(tot['a'])} KiB → {kib(tot['b'])} KiB raw "
        f"({pct(tot['a'], tot['b'])}), {kib(tot['ga'])} KiB → {kib(tot['gb'])} KiB "
        f"gzipped ({pct(tot['ga'], tot['gb'])}); emit time "
        f"{pct(tot['ta'], tot['tb'])}; worst image has "
        f"{worst.changed_pct:.3f}% changed pixels.",
    ]
    if not all(is_stable(run_a[n], run_b[n]) for n in names):
        lines += [
            "",
            "† `labels_to_svg` output varies between runs for this image, so part "
            "of its diff is run-to-run noise, not the change under test.",
        ]
    if missing:
        lines += ["", f"Not in both runs (skipped): {', '.join(missing)}"]

    report = "\n".join(lines) + "\n"
    print(report)
    if args.markdown:
        args.markdown.write_text(report)
    if args.max_changed_pct is not None and worst_stable_pct > args.max_changed_pct:
        sys.exit(
            f"Render diff too large: {worst_stable_pct:.3f}% changed pixels "
            f"> allowed {args.max_changed_pct}%"
        )


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="Where `prepare` stores frozen K-Means labels (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Run filter + K-Means once and cache labels")
    p.add_argument("--force", action="store_true", help="Recompute cached labels")
    p.add_argument(
        "--labeler",
        choices=["img2num", "numpy"],
        default="img2num",
        help="How to produce labels. `img2num` runs the real bilateral filter + "
        "K-Means; `numpy` is a deterministic fallback for machines where those "
        "fail (default: %(default)s)",
    )
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("run", help="Emit SVGs from cached labels and measure them")
    p.add_argument("out", type=Path, help="Output directory for SVGs + results.json")
    p.add_argument("--repeat", type=int, default=5, help="Timed runs per image")
    p.add_argument(
        "--only", action="append", help="Only this image (no suffix); repeatable"
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="Compare two runs (sizes, timing, renders)")
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument(
        "--tolerance",
        type=int,
        default=DEFAULT_TOLERANCE,
        help="Per-channel delta (0-255) above which a pixel counts as changed "
        "(default: %(default)s)",
    )
    p.add_argument("--markdown", type=Path, help="Also write the table to this file")
    p.add_argument(
        "--max-changed-pct",
        type=float,
        help="Exit non-zero if any image with deterministic output has more "
        "changed pixels than this (percent)",
    )
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
