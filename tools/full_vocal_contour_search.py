#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import librosa
import numpy as np
import requests
from scipy import signal

BAD_EDITION = re.compile(r"\b(remix|instrumental|karaoke|nightcore|sped\s*up|slowed|8d audio|tribute|cover version)\b", re.I)
BAD_VIDEO = re.compile(r"\b(cover|karaoke|reaction|tutorial|nightcore|sped\s*up|slowed|8d audio|lyrics? video reaction)\b", re.I)


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").casefold()).strip()


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").casefold()).strip("-")[:80]


def base_title(s: str) -> str:
    x = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", (s or "").casefold())
    return re.sub(r"\s+", " ", x).strip()


def run(cmd, *, timeout=None, check=True, capture=True):
    return subprocess.run(cmd, timeout=timeout, check=check, capture_output=capture, text=True)


def enumerate_catalog(artists, outdir: pathlib.Path):
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 Chrome/150 Safari/537.36"
    rows = []
    for artist in artists:
        try:
            r = session.get(
                "https://itunes.apple.com/search",
                params={"term": artist, "entity": "song", "limit": 200, "country": "US"},
                timeout=45,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print("CATALOG_FAIL", artist, repr(exc), flush=True)
            continue
        seen = set()
        count = 0
        for x in data.get("results", []):
            if norm_text(x.get("artistName")) != norm_text(artist):
                continue
            title = (x.get("trackName") or "").strip()
            if not title or BAD_EDITION.search(title):
                continue
            edition = "acoustic" if "acoustic" in title.casefold() else "demo" if "demo" in title.casefold() else "studio"
            key = (base_title(title), edition)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "artist": artist,
                "title": title,
                "apple_id": x.get("trackId"),
                "release_date": x.get("releaseDate"),
                "preview_url": x.get("previewUrl"),
            })
            count += 1
        print("CATALOG", artist, count, flush=True)
    (outdir / "catalog.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print("CATALOG_TOTAL", len(rows), flush=True)
    return rows


def download_one(row, audio_dir: pathlib.Path):
    prefix = f"{slug(row['artist'])}__{row.get('apple_id') or 'x'}__{slug(row['title'])}"
    outtmpl = str(audio_dir / (prefix + ".%(ext)s"))
    query = f"ytsearch3:{row['artist']} {row['title']} official audio"
    cmd = [
        "yt-dlp", "--no-playlist", "--match-filter", "duration >= 55 & duration <= 620",
        "--extract-audio", "--audio-format", "m4a", "--audio-quality", "7",
        "--max-filesize", "30M", "--no-warnings", "--quiet", "--print-json",
        "-o", outtmpl, query,
    ]
    try:
        p = run(cmd, timeout=220, check=False)
        if p.returncode != 0:
            return None
        lines = [line for line in p.stdout.splitlines() if line.lstrip().startswith("{")]
        if not lines:
            return None
        meta = json.loads(lines[-1])
        video_title = meta.get("title") or ""
        if BAD_VIDEO.search(video_title):
            for f in audio_dir.glob(prefix + ".*"):
                f.unlink(missing_ok=True)
            return None
        files = list(audio_dir.glob(prefix + ".*"))
        if not files:
            return None
        return {
            **row,
            "filename": files[0].name,
            "youtube_id": meta.get("id"),
            "youtube_title": video_title,
            "channel": meta.get("channel"),
            "duration": meta.get("duration"),
        }
    except Exception as exc:
        return None


def download_catalog(rows, audio_dir: pathlib.Path, outdir: pathlib.Path, workers=4):
    downloaded = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(download_one, row, audio_dir) for row in rows]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            result = fut.result()
            if result:
                downloaded.append(result)
            if i % 20 == 0:
                print("DOWNLOAD_PROGRESS", i, len(downloaded), flush=True)
    downloaded.sort(key=lambda r: (r["artist"].casefold(), r["title"].casefold()))
    (outdir / "downloaded.json").write_text(json.dumps(downloaded, indent=2, ensure_ascii=False))
    print("DOWNLOADED_TOTAL", len(downloaded), flush=True)
    return downloaded


def separate_vocals(downloaded, audio_dir: pathlib.Path, sep_dir: pathlib.Path, batch_size=24):
    files = [audio_dir / row["filename"] for row in downloaded if (audio_dir / row["filename"]).exists()]
    for start in range(0, len(files), batch_size):
        batch = files[start:start + batch_size]
        print("DEMUCS_BATCH", start, len(batch), flush=True)
        cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "--jobs", "2", "-o", str(sep_dir)] + [str(p) for p in batch]
        p = run(cmd, timeout=7200, check=False)
        if p.returncode != 0:
            print("DEMUCS_FAILURE", start, p.stderr[-3000:], flush=True)


def detect_taps(y, sr):
    peaks, props = signal.find_peaks(np.abs(y), distance=int(0.40 * sr), height=max(0.12, float(np.max(np.abs(y))) * 0.18), prominence=0.10)
    return peaks / sr


def extract_pitch(path, *, query=False, sr=12000, hop=240):
    y, _ = librosa.load(path, sr=sr, mono=True)
    if not query:
        sos = signal.butter(4, [120, 2200], btype="bandpass", fs=sr, output="sos")
        y = signal.sosfiltfilt(sos, y)
        y = librosa.effects.harmonic(y, margin=2.0)
    frame = 2048
    f0, voiced, prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("A2") if not query else 250,
        fmax=librosa.note_to_hz("C6") if not query else 2000,
        sr=sr,
        frame_length=frame,
        hop_length=hop,
        fill_na=np.nan,
    )
    midi = librosa.hz_to_midi(f0)
    times = librosa.times_like(midi, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=True)[0]
    valid = np.isfinite(midi) & (prob >= (0.50 if not query else 0.60))
    if not query and np.any(valid):
        floor = np.percentile(rms[valid], 12)
        valid &= rms >= floor
    midi[~valid] = np.nan
    return times, midi, prob, y, sr


def prepare_query(query_path):
    times, midi, prob, y, sr = extract_pitch(query_path, query=True, sr=12000, hop=120)
    taps = detect_taps(y, sr)
    valid = np.isfinite(midi)
    for tap in taps:
        valid &= np.abs(times - tap) > 0.075
    valid &= (times >= 0.55) & (times <= 11.10)
    # Remove obvious tracker glitches while preserving the genuine large rises.
    med = signal.medfilt(np.where(np.isfinite(midi), midi, np.nanmedian(midi[valid])), 5)
    good = valid & (np.abs(midi - med) < 4.5)
    grid = np.arange(0.60, 11.051, 0.05)
    src_t = times[good]
    src_m = midi[good]
    values = np.interp(grid, src_t, src_m)
    nearest = np.min(np.abs(grid[:, None] - src_t[None, :]), axis=1)
    values[nearest > 0.20] = np.nan
    filled = np.interp(np.arange(len(values)), np.where(np.isfinite(values))[0], values[np.isfinite(values)])
    filled = signal.medfilt(filled, 5)
    filled[~np.isfinite(values)] = np.nan
    # Confidence weights fall near tap gaps and at the uncertain extreme endpoints.
    weights = np.where(np.isfinite(filled), 1.0, 0.0)
    weights[(grid < 0.72) | (grid > 10.85)] *= 0.55
    # Keep actual absolute contour, but all matching is transposition invariant.
    segments = {
        "whole": (grid, filled, weights),
        "repeated_core": (grid[(grid >= 0.70) & (grid <= 7.18)], filled[(grid >= 0.70) & (grid <= 7.18)], weights[(grid >= 0.70) & (grid <= 7.18)]),
        "last_two": (grid[(grid >= 4.85) & (grid <= 10.90)], filled[(grid >= 4.85) & (grid <= 10.90)], weights[(grid >= 4.85) & (grid <= 10.90)]),
        "ending": (grid[(grid >= 7.20) & (grid <= 10.90)], filled[(grid >= 7.20) & (grid <= 10.90)], weights[(grid >= 7.20) & (grid <= 10.90)]),
    }
    report = {
        "tap_times": taps.tolist(),
        "grid": grid.tolist(),
        "midi": [None if not np.isfinite(x) else float(x) for x in filled],
        "weights": weights.tolist(),
    }
    return segments, report


def nearest_octave_adjust(target, reference):
    # Predominant vocal trackers occasionally jump by one octave. Choose the nearest octave
    # to the query after global transposition, but penalize octave switching separately.
    choices = np.stack([target - 12, target, target + 12], axis=1)
    idx = np.argmin(np.abs(choices - reference[:, None]), axis=1)
    adjusted = choices[np.arange(len(target)), idx]
    return adjusted, idx - 1


def huber(x, delta=2.5):
    a = np.abs(x)
    return np.where(a <= delta, 0.5 * a * a / delta, a - 0.5 * delta)


def match_segment(qt, qv, qw, tt, tv):
    qvalid = np.isfinite(qv) & (qw > 0)
    if qvalid.sum() < 20:
        return {"score": 999.0}
    q0 = qt[0]
    rel = qt - q0
    best = None
    finite_t = np.isfinite(tv)
    if finite_t.sum() < 50:
        return {"score": 999.0}
    valid_times = tt[finite_t]
    valid_values = tv[finite_t]
    for scale in np.arange(0.72, 1.361, 0.025):
        span = rel[-1] * scale
        max_start = tt[-1] - span
        if max_start <= 0:
            continue
        for start in np.arange(0, max_start, 0.10):
            sample_t = start + rel * scale
            vals = np.interp(sample_t, valid_times, valid_values, left=np.nan, right=np.nan)
            nearest = np.minimum(np.abs(sample_t - np.interp(sample_t, valid_times, valid_times)), 99)
            # Determine whether interpolation bridged long unvoiced gaps.
            inds = np.searchsorted(valid_times, sample_t)
            left = np.clip(inds - 1, 0, len(valid_times) - 1)
            right = np.clip(inds, 0, len(valid_times) - 1)
            gap = valid_times[right] - valid_times[left]
            vals[gap > 0.34] = np.nan
            use = qvalid & np.isfinite(vals)
            coverage = float(np.sum(qw[use]) / max(np.sum(qw[qvalid]), 1e-9))
            if coverage < 0.66:
                continue
            transpose = float(np.nanmedian(vals[use] - qv[use]))
            raw = vals - transpose
            adjusted, octave_state = nearest_octave_adjust(raw, qv)
            err = adjusted - qv
            weighted = qw[use]
            mae = float(np.sum(np.abs(err[use]) * weighted) / np.sum(weighted))
            robust = float(np.sum(huber(err[use]) * weighted) / np.sum(weighted))
            # Compare local movement over 100 ms; this preserves glides without overfitting frame noise.
            dstep = 2
            dmask = use[dstep:] & use[:-dstep]
            if dmask.sum() >= 8:
                qd = qv[dstep:] - qv[:-dstep]
                td = adjusted[dstep:] - adjusted[:-dstep]
                dw = np.minimum(qw[dstep:], qw[:-dstep])
                deriv = float(np.sum(np.abs((td - qd)[dmask]) * dw[dmask]) / np.sum(dw[dmask]))
                direction = float(np.mean(np.sign(td[dmask]) != np.sign(qd[dmask])))
            else:
                deriv, direction = 9.0, 1.0
            if use.sum() >= 10:
                corr = float(np.corrcoef(qv[use], adjusted[use])[0, 1])
                if not np.isfinite(corr):
                    corr = -1.0
            else:
                corr = -1.0
            switches = float(np.mean(np.diff(octave_state[use]) != 0)) if use.sum() > 2 else 1.0
            p90 = float(np.percentile(np.abs(err[use]), 90))
            score = (
                0.58 * mae + 0.36 * robust + 0.38 * deriv + 1.25 * direction
                + 0.75 * (1.0 - corr) + 1.8 * (1.0 - coverage)
                + 0.55 * switches + 0.08 * p90
            )
            rec = {
                "score": float(score), "start": float(start), "scale": float(scale),
                "coverage": coverage, "transpose": transpose, "mae": mae,
                "robust_error": robust, "derivative_error": deriv,
                "direction_mismatch": direction, "correlation": corr,
                "octave_switch_rate": switches, "p90_error": p90,
            }
            if best is None or score < best["score"]:
                best = rec
    return best or {"score": 999.0}


def analyze_one(row, sep_dir: pathlib.Path, query_segments):
    stem = pathlib.Path(row["filename"]).stem
    vocal = sep_dir / "htdemucs" / stem / "vocals.wav"
    if not vocal.exists():
        return None
    try:
        tt, tv, _, _, _ = extract_pitch(vocal, query=False, sr=12000, hop=240)
        matches = {name: match_segment(*segment, tt, tv) for name, segment in query_segments.items()}
        # Whole phrase dominates, but a strong core or ending can survive imperfect memory.
        composite = min(
            matches["whole"]["score"],
            matches["repeated_core"]["score"] + 0.55,
            matches["last_two"]["score"] + 0.70,
            matches["ending"]["score"] + 1.15,
        )
        winning = min(matches, key=lambda name: matches[name]["score"] + {"whole": 0, "repeated_core": .55, "last_two": .70, "ending": 1.15}[name])
        return {**row, "composite": float(composite), "winning_segment": winning, "matches": matches, "vocal_path": str(vocal)}
    except Exception as exc:
        return None


def rank(downloaded, sep_dir, query_path, outdir, workers=4):
    segments, query_report = prepare_query(query_path)
    (outdir / "query_contour.json").write_text(json.dumps(query_report, indent=2))
    rows = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(analyze_one, row, sep_dir, segments) for row in downloaded]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            result = fut.result()
            if result:
                rows.append(result)
            if i % 20 == 0:
                print("ANALYSIS_PROGRESS", i, len(rows), flush=True)
    rows.sort(key=lambda r: r["composite"])
    (outdir / "rankings.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    lines = []
    for r in rows:
        win = r["matches"][r["winning_segment"]]
        lines.append(
            f"{r['composite']:.6f}\t{r['winning_segment']}\t{r['artist']}\t{r['title']}\t"
            f"start={win.get('start', -1):.2f}\tscale={win.get('scale', -1):.3f}\t"
            f"mae={win.get('mae', 99):.3f}\tdir={win.get('direction_mismatch', 1):.3f}\t"
            f"corr={win.get('correlation', -1):.3f}\tcoverage={win.get('coverage', 0):.3f}"
        )
    (outdir / "rankings.tsv").write_text("\n".join(lines))
    print("\n".join(lines[:50]), flush=True)
    return rows


def package_top(rows, outdir, top_dir, n=30):
    top_dir.mkdir(parents=True, exist_ok=True)
    for rank_idx, row in enumerate(rows[:n], 1):
        src = pathlib.Path(row["vocal_path"])
        if not src.exists():
            continue
        dst = top_dir / f"{rank_idx:02d}__{slug(row['artist'])}__{slug(row['title'])}.mp3"
        run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-codec:a", "libmp3lame", "-b:a", "96k", str(dst)], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artists-json", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    artists = json.loads(args.artists_json)
    work = pathlib.Path(args.work)
    audio_dir = work / "audio"
    sep_dir = work / "separated"
    outdir = work / "results"
    top_dir = work / "top_vocals"
    for d in [audio_dir, sep_dir, outdir, top_dir]:
        d.mkdir(parents=True, exist_ok=True)
    rows = enumerate_catalog(artists, outdir)
    downloaded = download_catalog(rows, audio_dir, outdir)
    separate_vocals(downloaded, audio_dir, sep_dir)
    ranked = rank(downloaded, sep_dir, args.query, outdir)
    package_top(ranked, outdir, top_dir)


if __name__ == "__main__":
    main()
