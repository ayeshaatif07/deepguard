"""Detects which platform a pasted URL belongs to and hands off to its extractor.
Supports YouTube, Instagram and Facebook; TikTok blocks automated access.
"""

import re

from modules.social_media import youtube_extractor, facebook_extractor, instagram_extractor


class UnsupportedPlatformError(Exception):
    """Raised for a social media URL from a platform this pipeline does not support."""
    pass


class InvalidURLError(Exception):
    """Raised when the input doesn't look like a URL at all."""
    pass


PLATFORM_PATTERNS = {
    "youtube": re.compile(r"(youtube\.com|youtu\.be)", re.IGNORECASE),
    "instagram": re.compile(r"instagram\.com", re.IGNORECASE),
    "facebook": re.compile(r"facebook\.com|fb\.watch", re.IGNORECASE),
}

PLATFORM_EXTRACTORS = {
    "youtube": youtube_extractor.extract,
    "facebook": facebook_extractor.extract,
    "instagram": instagram_extractor.extract,
}


def detect_platform(url: str) -> str:
    """Returns 'youtube', 'instagram', 'facebook', or 'unknown'."""
    if not re.match(r"^https?://", url.strip(), re.IGNORECASE):
        raise InvalidURLError("That doesn't look like a URL - it should start with http:// or https://")
    for platform, pattern in PLATFORM_PATTERNS.items():
        if pattern.search(url):
            return platform
    return "unknown"


def download_video(url: str, output_dir: str):
    """Downloads the video at `url` and returns a dict with video_path, title,
    caption, platform and duration.
    """
    platform = detect_platform(url)
    if platform == "unknown":
        raise UnsupportedPlatformError(
            "This doesn't look like a YouTube, Instagram, or Facebook link."
        )

    extractor_fn = PLATFORM_EXTRACTORS[platform]
    try:
        extracted = extractor_fn(url, output_dir)
    except instagram_extractor.InstagramNotYetSupportedError as e:
        raise UnsupportedPlatformError(str(e))

    extracted["platform"] = platform
    return extracted
