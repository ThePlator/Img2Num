# SVG output benchmark

Tooling for [#637](https://github.com/Ryan-Millard/Img2Num/issues/637): measure how big Img2Num's SVG output is, how long it takes to emit, and whether an optimization visibly changes the result.

- `corpus/`: 26 permissively licensed images (photos, illustrations, logos and stress cases, 32 px to 1600 px wide). Sources and licenses are in [`corpus/SOURCES.md`](corpus/SOURCES.md).
- `svg_bench.py`: the measurement script (`prepare`, `run` and `compare`).
- `build_corpus.py`: rebuilds `corpus/` from its sources. You only need it to audit or extend the corpus.

## Quick start

Run everything from the repository root. The `bench` dependency group adds Pillow and [resvg](https://github.com/linebender/resvg) (used to rasterize SVGs) on top of the locally built `img2num` Python package.

```sh
# 1. Freeze K-Means labels for every corpus image (once per machine).
uv run --group bench bench/svg/svg_bench.py prepare

# 2. Emit SVGs with the baseline and with your change.
git switch dev
uv run --group bench bench/svg/svg_bench.py run bench/svg/results/before
git switch <your-branch>
uv run --group bench bench/svg/svg_bench.py run bench/svg/results/after

# 3. Compare. Prints a Markdown table you can paste into your PR.
uv run --group bench bench/svg/svg_bench.py compare \
  bench/svg/results/before bench/svg/results/after --markdown bench/svg/results/table.md
```

`uv run` rebuilds `img2num` from your working tree, so the `run` step measures whatever is checked out. `bench/svg/.cache/` and `bench/svg/results/` are git-ignored.

## What is measured

`run` writes one SVG per image plus `results.json` with:

| Metric           | Meaning                                                                       |
| ---------------- | ----------------------------------------------------------------------------- |
| `svg_bytes`      | Raw UTF-8 size of the SVG.                                                    |
| `svg_gzip_bytes` | Size after `gzip -9`, which is roughly what a web server sends.               |
| `path_count`     | Number of `<path>` elements.                                                  |
| `emit_ms_*`      | Wall time of `labels_to_svg`, median and best of `--repeat` runs (default 5). |

`compare` rasterizes both SVGs of every image with resvg at native size on a white background and reports:

| Column     | Meaning                                                                               |
| ---------- | ------------------------------------------------------------------------------------- |
| Mean diff  | Mean absolute per-channel difference (0-255).                                         |
| Max diff   | Largest single-channel difference.                                                    |
| Changed px | Share of pixels where any channel differs by more than `--tolerance` (default 8/255). |

The timing column uses the best of N runs, which is less noisy than the median on a shared machine. On a busy machine single images still vary by up to ±30% between identical runs, while the corpus total stays within a few percent. Judge emitter speed by the total, and raise `--repeat` when you need tighter numbers. Pass `--max-changed-pct` to make `compare` exit non-zero when any image changes more than an agreed amount. Only images with deterministic output count toward this check (see below).

## Why labels are frozen

`img2num.kmeans` seeds its centroids randomly, so the full pipeline produces a different SVG every run. `prepare` runs the bilateral filter and K-Means once with `image_to_svg`'s default settings and caches the labels in `.cache/`. `run` then only calls `labels_to_svg`, the stage that contains the SVG emitter. Before/after differences therefore come from the code under test, as long as both runs share the same cache.

If `img2num.bilateral_filter` or `img2num.kmeans` do not work on your machine (for example a container without a usable GPU adapter), use `prepare --labeler numpy`. It is a deterministic stand-in (5x5 median filter + seeded k-means++ in RGB, k=16) that never calls img2num. It approximates the real clustering closely enough to benchmark the emitter, and every machine gets the same labels.

## Known noise: `labels_to_svg` is not fully deterministic

Even with identical labels, `labels_to_svg` occasionally emits different SVGs for the same input. Region adjacency is stored in a `std::set<Node_ptr>`, which is ordered by heap address, so the merge order of equally-good candidates can change between runs. In practice this affects some grayscale and texture images (brick, camera, cell, grass, gravel) and changes their size by about 0.1%. `run` detects it by comparing repeated outputs, and `compare` marks affected rows with †. Comparing a run against a second run of the same code shows this noise floor.

## Corpus

| Category      | Images                                                                                                                  |
| ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Photographs   | astronaut, cat, coffee, rocket, Hubble deep field, retina (1411x1411), grayscale camera, clock and cell                  |
| Textures      | brick, grass, gravel                                                                                                    |
| Illustrations | five Twemoji graphics, a horse silhouette, flat overlapping shapes, a pixel-art sprite, a wide banner                   |
| Logos         | concentric rings (nested even-odd holes), a shield badge, a 32x32 icon                                                  |
| Stress cases  | a smooth gradient (banding into many thin regions) and a grid of identical icons (the best case for `<use>` dedup) |

To add an image, extend `build_corpus.py` with a permissively licensed source, run `uv run bench/svg/build_corpus.py`, and add the file's license to `REUSE.toml`.
