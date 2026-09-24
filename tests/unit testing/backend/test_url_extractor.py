"""Unit tests for modules/social_media/url_extractor.py's detect_platform()
- pure string/regex logic, no network call, no yt-dlp download involved."""
import pytest

from modules.social_media.url_extractor import detect_platform, InvalidURLError


@pytest.mark.parametrize("url,expected", [
    ("https://www.youtube.com/watch?v=abc123", "youtube"),
    ("https://youtu.be/abc123", "youtube"),
    ("https://www.instagram.com/p/abc123/", "instagram"),
    ("https://www.facebook.com/watch/?v=123456", "facebook"),
    ("https://fb.watch/abc123/", "facebook"),
    ("https://www.tiktok.com/@someone/video/123", "unknown"),
    ("https://example.com/some/random/page", "unknown"),
])
def test_detect_platform_classification(url, expected):
    assert detect_platform(url) == expected


def test_detect_platform_rejects_non_url_input():
    with pytest.raises(InvalidURLError):
        detect_platform("not a url at all")


def test_detect_platform_rejects_empty_string():
    with pytest.raises(InvalidURLError):
        detect_platform("")
