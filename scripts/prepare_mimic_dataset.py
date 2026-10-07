#!/usr/bin/env python3
"""Build fixed-length MIMIC-IV ICU vital-sign windows.

The script streams the large ``chartevents.csv.gz`` file, aggregates six vital
signs into 15-minute bins, splits by subject, imputes from train-only statistics,
and writes one float32 ``(length, 6)`` NumPy file per window.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = ("heart_rate", "resp_rate", "spo2", "sbp", "dbp", "map")
ITEM_TO_CHANNEL = {
    220045: 0,  # Heart Rate
    220210: 1,  # Respiratory Rate
    220277: 2,  # O2 saturation pulseoxymetry
    220050: 3, 220179: 3,  # arterial / non-invasive systolic BP
    220051: 4, 220180: 4,  # arterial / non-invasive diastolic BP
    220052: 5, 220181: 5,  # arterial / non-invasive mean BP
}
VALID_RANGE = np.asarray(
    [(20, 250), (4, 80), (50, 100), (40, 300), (20, 200), (30, 220)],
    dtype=np.float64,
)


def split_subject(subject_id: int, seed: int) -> str:
    digest = hashlib.blake2b(
        f"{seed}:{int(subject_id)}".encode(), digest_size=8
    ).digest()
    value = int.from_bytes(digest, "little") / float(2**64)
    return "train" if value < 0.8 else ("val" if value < 0.9 else "test")


def load_cohort(icu_dir: Path, min_hours: float, seed: int, max_stays: int) -> pd.DataFrame:
    path = icu_dir / "icustays.csv.gz"
    stays = pd.read_csv(
        path,
        usecols=["subject_id", "stay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    ).dropna()
    duration = (stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600
    stays = stays.loc[duration >= min_hours].copy()
    stays["split"] = [split_subject(x, seed) for x in stays["subject_id"]]
    stays = stays.sort_values(["subject_id", "stay_id"], kind="stable")
    if max_stays > 0:
        stays = stays.head(max_stays)
    if stays.empty:
        raise RuntimeError(f"No ICU stays are at least {min_hours:g} hours long")
    return stays.reset_index(drop=True)


def initialise_db(db_path: Path, cohort: pd.DataFrame, overwrite: bool) -> sqlite3.Connection:
    if db_path.exists() and overwrite:
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS bins ("
        "stay_id INTEGER, bin INTEGER, channel INTEGER, total REAL, n INTEGER, "
        "PRIMARY KEY(stay_id, bin, channel))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
    )
    signature = hashlib.sha256(
        cohort[["subject_id", "stay_id", "intime", "outtime", "split"]]
        .to_csv(index=False).encode()
    ).hexdigest()
    old = conn.execute("SELECT value FROM meta WHERE key='cohort_signature'").fetchone()
    if old and old[0] != signature:
        raise RuntimeError(
            f"Cache cohort differs: {db_path}. Use --overwrite-cache to rebuild it."
        )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key,value) VALUES('cohort_signature',?)", (signature,)
    )
    conn.commit()
    return conn


def aggregate_events(
    chart_path: Path,
    cohort: pd.DataFrame,
    conn: sqlite3.Connection,
    bin_minutes: int,
    chunksize: int,
) -> None:
    complete = conn.execute("SELECT value FROM meta WHERE key='events_complete'").fetchone()
    if complete and complete[0] == "1":
        print("[cache] using completed event aggregation")
        return
    # A previous interrupted scan may contain only a non-uniform prefix of the
    # source file. Restart cleanly so partial duplicate aggregates cannot bias
    # bin means.
    partial_count = conn.execute("SELECT COUNT(*) FROM bins").fetchone()[0]
    if partial_count:
        print(f"[cache] discarding {partial_count} partial aggregate rows")
        conn.execute("DELETE FROM bins")
        conn.commit()

    lookup = cohort.set_index("stay_id")[["intime", "outtime"]]
    allowed_stays = set(int(x) for x in cohort["stay_id"])
    allowed_items = set(ITEM_TO_CHANNEL)
    sql = (
        "INSERT INTO bins(stay_id,bin,channel,total,n) VALUES(?,?,?,?,?) "
        "ON CONFLICT(stay_id,bin,channel) DO UPDATE SET "
        "total=total+excluded.total,n=n+excluded.n"
    )
    usecols = ["stay_id", "charttime", "itemid", "valuenum"]
    reader = pd.read_csv(
        chart_path,
        usecols=usecols,
        chunksize=chunksize,
        dtype={"stay_id": "Int64", "itemid": "Int64", "valuenum": "float64"},
    )
    for chunk_no, chunk in enumerate(reader, 1):
        chunk = chunk[
            chunk["stay_id"].isin(allowed_stays)
            & chunk["itemid"].isin(allowed_items)
            & chunk["valuenum"].notna()
        ].copy()
        if chunk.empty:
            if chunk_no % 20 == 0:
                print(f"[events] chunks={chunk_no}")
            continue
        chunk["stay_id"] = chunk["stay_id"].astype(np.int64)
        chunk["itemid"] = chunk["itemid"].astype(np.int64)
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.join(lookup, on="stay_id").dropna(subset=["charttime", "intime"])
        chunk = chunk[(chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] < chunk["outtime"])]
        chunk["channel"] = chunk["itemid"].map(ITEM_TO_CHANNEL).astype(np.int8)
        lo = VALID_RANGE[chunk["channel"].to_numpy(), 0]
        hi = VALID_RANGE[chunk["channel"].to_numpy(), 1]
        vals = chunk["valuenum"].to_numpy()
        chunk = chunk[(vals >= lo) & (vals <= hi)]
        if chunk.empty:
            continue
        seconds = (chunk["charttime"] - chunk["intime"]).dt.total_seconds()
        chunk["bin"] = (seconds // (bin_minutes * 60)).astype(np.int64)
        grouped = chunk.groupby(["stay_id", "bin", "channel"])["valuenum"].agg(["sum", "count"])
        rows = [
            (int(a), int(b), int(c), float(s), int(n))
            for (a, b, c), (s, n) in grouped.iterrows()
        ]
        conn.executemany(sql, rows)
        conn.commit()
        if chunk_no % 10 == 0:
            print(f"[events] chunks={chunk_no}, retained_partial_groups={len(rows)}")

    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('events_complete','1')")
    conn.commit()


def train_medians(conn: sqlite3.Connection, train_stays: set[int]) -> np.ndarray:
    values = [[] for _ in CHANNELS]
    cur = conn.execute("SELECT stay_id,channel,total,n FROM bins")
    for stay_id, channel, total, count in cur:
        if int(stay_id) in train_stays:
            values[int(channel)].append(float(total) / int(count))
    medians = []
    for channel, vals in zip(CHANNELS, values):
        if not vals:
            raise RuntimeError(f"No train observations for channel {channel}")
        medians.append(float(np.median(np.asarray(vals))))
    return np.asarray(medians, dtype=np.float32)


def load_stay(conn: sqlite3.Connection, stay_id: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.full((n_bins, len(CHANNELS)), np.nan, dtype=np.float32)
    cur = conn.execute(
        "SELECT bin,channel,total,n FROM bins WHERE stay_id=? AND bin<?",
        (int(stay_id), int(n_bins)),
    )
    for bin_idx, channel, total, count in cur:
        arr[int(bin_idx), int(channel)] = float(total) / int(count)
    return arr, np.isfinite(arr)


def impute(arr: np.ndarray, medians: np.ndarray, interp_limit: int) -> np.ndarray:
    frame = pd.DataFrame(arr)
    frame = frame.interpolate(
        method="linear", axis=0, limit=interp_limit, limit_direction="both"
    )
    frame = frame.ffill(limit=interp_limit).bfill(limit=interp_limit)
    out = frame.to_numpy(dtype=np.float32)
    missing = ~np.isfinite(out)
    if missing.any():
        out[missing] = np.take(medians, np.where(missing)[1])
    return out


def write_windows(
    output_root: Path,
    cohort: pd.DataFrame,
    conn: sqlite3.Connection,
    lengths: list[int],
    stride_by_length: dict[int, int],
    medians: np.ndarray,
    interp_limit: int,
    max_missing: float,
    max_windows_per_stay: int,
    bin_minutes: int,
) -> dict:
    totals = {length: {split: 0 for split in ("train", "val", "test")} for length in lengths}
    writers = {}
    for length in lengths:
        root = output_root / f"MIMICIV{length}"
        for split in ("train", "val", "test"):
            path = root / split
            path.mkdir(parents=True, exist_ok=True)
            manifest = (root / f"manifest_{split}.csv").open("w", newline="")
            writer = csv.writer(manifest)
            writer.writerow(["file", "source_hash", "start_bin", "observed_fraction"])
            writers[(length, split)] = (path, manifest, writer)

    try:
        for row_no, row in enumerate(cohort.itertuples(index=False), 1):
            duration_bins = int(
                (row.outtime - row.intime).total_seconds() // (bin_minutes * 60)
            )
            raw, observed = load_stay(conn, int(row.stay_id), duration_bins)
            if not observed.any():
                continue
            filled = impute(raw, medians, interp_limit)
            source_hash = hashlib.sha256(
                f"{int(row.subject_id)}:{int(row.stay_id)}".encode()
            ).hexdigest()[:16]
            for length in lengths:
                starts = list(range(0, duration_bins - length + 1, stride_by_length[length]))
                if max_windows_per_stay > 0:
                    starts = starts[:max_windows_per_stay]
                for start in starts:
                    obs_fraction = float(observed[start : start + length].mean())
                    if 1.0 - obs_fraction > max_missing:
                        continue
                    split = str(row.split)
                    idx = totals[length][split]
                    file_name = f"{idx:07d}.npy"
                    path, _, writer = writers[(length, split)]
                    np.save(path / file_name, filled[start : start + length].astype(np.float32))
                    writer.writerow([file_name, source_hash, start, f"{obs_fraction:.6f}"])
                    totals[length][split] += 1
            if row_no % 1000 == 0:
                print(f"[windows] stays={row_no}/{len(cohort)} totals={totals}")
    finally:
        for _, handle, _ in writers.values():
            handle.close()
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-root", type=Path, default=Path("data/mimic-iv-3.1"))
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--bin-minutes", type=int, default=15)
    parser.add_argument("--stride", type=int, nargs="*", default=None,
                        help="one stride per length; defaults to half-window")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--interp-limit", type=int, default=8)
    parser.add_argument("--max-missing", type=float, default=0.80)
    parser.add_argument("--max-stays", type=int, default=0)
    parser.add_argument("--max-windows-per-stay", type=int, default=8)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lengths = sorted(set(args.lengths))
    if args.stride is None:
        strides = {x: max(1, x // 2) for x in lengths}
    elif len(args.stride) != len(lengths):
        raise ValueError("--stride must contain one value per --lengths entry")
    else:
        strides = dict(zip(lengths, args.stride))
    icu_dir = args.mimic_root / "icu"
    # Each requested length gets every stay long enough for that length; using
    # the shortest threshold avoids unnecessarily discarding 32--64 h stays
    # from the 128-step benchmark when 256 is built in the same run.
    min_hours = min(lengths) * args.bin_minutes / 60
    cohort = load_cohort(icu_dir, min_hours, args.seed, args.max_stays)
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_root / "mimic_iv_vitals_cache.sqlite"
    conn = initialise_db(cache_path, cohort, args.overwrite_cache)
    try:
        aggregate_events(
            icu_dir / "chartevents.csv.gz", cohort, conn, args.bin_minutes, args.chunksize
        )
        medians = train_medians(
            conn, set(cohort.loc[cohort["split"] == "train", "stay_id"].astype(int))
        )
        totals = write_windows(
            args.output_root, cohort, conn, lengths, strides, medians,
            args.interp_limit, args.max_missing, args.max_windows_per_stay,
            args.bin_minutes,
        )
    finally:
        conn.close()

    metadata = {
        "dataset": "MIMIC-IV v3.1 ICU chartevents",
        "channels": list(CHANNELS),
        "item_to_channel": {str(k): CHANNELS[v] for k, v in ITEM_TO_CHANNEL.items()},
        "bin_minutes": args.bin_minutes,
        "lengths": lengths,
        "stride": strides,
        "split": "subject-level deterministic 80/10/10",
        "seed": args.seed,
        "train_medians": dict(zip(CHANNELS, map(float, medians))),
        "valid_ranges": dict(zip(CHANNELS, VALID_RANGE.tolist())),
        "max_missing": args.max_missing,
        "counts": totals,
    }
    for length in lengths:
        path = args.output_root / f"MIMICIV{length}" / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
