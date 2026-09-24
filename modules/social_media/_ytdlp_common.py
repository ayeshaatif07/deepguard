"""Shared yt-dlp download options used by the per-platform extractors."""

import os
import time

import yt_dlp


def download_with_ytdlp(url: str, output_dir: str, filename_prefix: str, extra_opts: dict = None):
    """Downloads `url` into output_dir via yt-dlp, merged to mp4.
    Returns (video_path, info); extra_opts are merged over the shared defaults.
    """
    os.makedirs(output_dir, exist_ok=True)
    unique_id = str(int(time.time() * 1000))
    outtmpl = os.path.join(output_dir, f"{filename_prefix}_{unique_id}_%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Cap duration so a pasted full-length video can't trigger a multi-hour download.
        "match_filter": yt_dlp.utils.match_filter_func("duration < 1800"),
        # Force IPv4; a dead IPv6 route to the video CDN otherwise stalls downloads.
        "source_address": "0.0.0.0",
    }
    if extra_opts:
        ydl_opts.update(extra_opts)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)
        # prepare_filename() can report the pre-merge extension, so correct it to .mp4.
        mp4_path = os.path.splitext(video_path)[0] + ".mp4"
        if os.path.exists(mp4_path):
            video_path = mp4_path

    return video_path, info
