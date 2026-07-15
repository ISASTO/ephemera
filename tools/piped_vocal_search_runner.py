#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import subprocess
import sys
import urllib.parse

import requests

import tools.full_vocal_contour_search as core

PIPED_HOSTS = [
    "https://pipedapi.kavin.rocks",
    "https://pipedapi.adminforge.de",
    "https://api.piped.private.coffee",
    "https://pipedapi.leptons.xyz",
    "https://pipedapi-libre.kavin.rocks",
    "https://pipedapi.nosebs.ru",
    "https://piped-api.codespace.cz",
    "https://pipedapi.reallyaweso.me",
    "https://pipedapi.ducks.party",
    "https://pipedapi.drgns.space",
    "https://api.piped.yt",
]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/150 Safari/537.36"
BAD = re.compile(r"\b(cover|karaoke|reaction|tutorial|remix|nightcore|sped\s*up|slowed|8d|tribute|instrumental|live cover)\b", re.I)
VID = re.compile(r"[?&]v=([A-Za-z0-9_-]{11})")


def tokens(s):
    return set(re.sub(r"[^a-z0-9]+", " ", (s or "").casefold()).split())


def video_id(url):
    m = VID.search(url or "")
    return m.group(1) if m else None


def score_item(item, row):
    title = item.get("title") or ""
    if BAD.search(title):
        return -1000
    wanted = tokens(row["artist"] + " " + row["title"])
    got = tokens(title)
    overlap = len(wanted & got) / max(len(wanted), 1)
    artist_overlap = len(tokens(row["artist"]) & got) / max(len(tokens(row["artist"])), 1)
    title_overlap = len(tokens(row["title"]) & got) / max(len(tokens(row["title"])), 1)
    duration = int(item.get("duration") or 0)
    duration_bonus = 0 if 55 <= duration <= 620 else -2
    official = 0.7 if any(x in title.casefold() for x in ["official audio", "official video", "topic"]) else 0
    return 4 * overlap + 3 * artist_overlap + 4 * title_overlap + official + duration_bonus


def get_json(session, url, timeout=10):
    r = session.get(url, timeout=timeout, headers={"User-Agent": UA, "Accept": "application/json"})
    r.raise_for_status()
    return r.json()


def query_host(host, row):
    session = requests.Session()
    q = urllib.parse.quote(f"{row['artist']} {row['title']}", safe="")
    items = []
    for filt in ["music_songs", "videos"]:
        try:
            data = get_json(session, f"{host}/search?q={q}&filter={filt}", 12)
            items.extend((data or {}).get("items") or [])
            if items:
                break
        except Exception:
            continue
    candidates = []
    for item in items:
        vid = video_id(item.get("url"))
        if not vid:
            continue
        s = score_item(item, row)
        if s < 2.0:
            continue
        candidates.append((s, vid, item))
    candidates.sort(reverse=True, key=lambda x: x[0])
    for s, vid, item in candidates[:3]:
        try:
            data = get_json(session, f"{host}/streams/{vid}", 15)
        except Exception:
            continue
        streams = []
        for stream in (data or {}).get("audioStreams") or []:
            url = stream.get("url")
            if not url:
                continue
            fmt = (stream.get("format") or "").upper()
            if "HLS" in fmt or "DASH" in fmt:
                continue
            streams.append((int(stream.get("bitrate") or 0), url, stream))
        if streams:
            streams.sort(reverse=True, key=lambda x: x[0])
            return {
                "host": host,
                "url": streams[0][1],
                "video_id": vid,
                "video_title": item.get("title"),
                "duration": item.get("duration"),
                "score": s,
            }
    return None


def find_stream(row):
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(query_host, host, row) for host in PIPED_HOSTS]
        answers = []
        for fut in cf.as_completed(futures):
            try:
                result = fut.result()
            except Exception:
                result = None
            if result:
                answers.append(result)
                if result["score"] >= 9.0:
                    break
    if not answers:
        return None
    answers.sort(key=lambda x: x["score"], reverse=True)
    return answers[0]


def resolve_preview(row):
    if row.get("preview_url"):
        return row["preview_url"]
    try:
        s = requests.Session(); s.headers["User-Agent"] = UA
        d = s.get("https://itunes.apple.com/search", params={"term": f"{row['artist']} {row['title']}", "entity": "song", "limit": 50, "country": "US"}, timeout=30).json()
        na, nt = core.norm_text(row["artist"]), core.norm_text(row["title"])
        exact = [x for x in d.get("results", []) if core.norm_text(x.get("artistName")) == na and core.norm_text(x.get("trackName")) == nt and x.get("previewUrl")]
        if not exact:
            exact = [x for x in d.get("results", []) if na in core.norm_text(x.get("artistName")) and nt in core.norm_text(x.get("trackName")) and x.get("previewUrl")]
        return exact[0]["previewUrl"] if exact else None
    except Exception:
        return None


def validate_audio(path):
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=30)
        duration = float((p.stdout or "0").strip())
        return duration if duration >= 25 else 0
    except Exception:
        return 0


def download_url(url, path):
    s = requests.Session(); s.headers.update({"User-Agent": UA, "Referer": "https://piped.video/"})
    with s.get(url, stream=True, timeout=90, allow_redirects=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 256):
                if chunk:
                    f.write(chunk)
    return path.stat().st_size


def piped_download_one(row, audio_dir):
    prefix = f"{core.slug(row['artist'])}__{row.get('apple_id') or 'x'}__{core.slug(row['title'])}"
    stream = find_stream(row)
    if stream:
        path = audio_dir / (prefix + ".webm")
        try:
            size = download_url(stream["url"], path)
            duration = validate_audio(path)
            if size > 100000 and duration:
                print("PIPED_OK", row["artist"], row["title"], stream["host"], stream["video_title"], duration, flush=True)
                return {**row, "filename": path.name, "youtube_id": stream["video_id"], "youtube_title": stream["video_title"], "duration": duration, "source": "piped_full"}
        except Exception as exc:
            path.unlink(missing_ok=True)
            print("PIPED_DOWNLOAD_FAIL", row["artist"], row["title"], repr(exc), flush=True)
    preview = resolve_preview(row)
    if preview:
        path = audio_dir / (prefix + ".m4a")
        try:
            size = download_url(preview, path)
            duration = validate_audio(path)
            if size > 50000 and duration:
                print("PREVIEW_FALLBACK", row["artist"], row["title"], duration, flush=True)
                return {**row, "filename": path.name, "duration": duration, "source": "apple_preview"}
        except Exception as exc:
            path.unlink(missing_ok=True)
    print("DOWNLOAD_MISS", row["artist"], row["title"], flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artists-json")
    ap.add_argument("--targets-json")
    ap.add_argument("--query", required=True)
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    audio = work / "audio"; sep = work / "separated"; out = work / "results"; top = work / "top_vocals"
    for d in [audio, sep, out, top]: d.mkdir(parents=True, exist_ok=True)
    if args.targets_json:
        rows = json.loads(args.targets_json)
        (out / "catalog.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        rows = core.enumerate_catalog(json.loads(args.artists_json), out)
    core.download_one = piped_download_one
    downloaded = core.download_catalog(rows, audio, out, workers=4)
    core.separate_vocals(downloaded, audio, sep, batch_size=20)
    ranked = core.rank(downloaded, sep, args.query, out, workers=4)
    core.package_top(ranked, out, top, n=30)


if __name__ == "__main__":
    main()
