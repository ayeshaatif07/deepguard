"""Facebook extractor for the social media pipeline. Public posts only."""

from modules.social_media._ytdlp_common import download_with_ytdlp


def extract(url: str, output_dir: str):
    """Returns {video_path, title, caption, duration}."""
    video_path, info = download_with_ytdlp(url, output_dir, filename_prefix="facebook")
    return {
        "video_path": video_path,
        "title": info.get("title") or "",
        "caption": info.get("description") or "",
        "duration": info.get("duration"),
    }
