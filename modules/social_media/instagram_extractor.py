"""Instagram extractor for the social media pipeline.
Instagram refuses anonymous access, so this reads a Netscape cookies file.
"""

import os

from modules.social_media._ytdlp_common import download_with_ytdlp

# In-project path - safe here specifically because these are throwaway-
# account cookies (see SECURITY note above), not real personal credentials.
COOKIES_PATH = os.path.join(os.path.dirname(__file__), "instagram_cookies.txt")


class InstagramNotYetSupportedError(Exception):
    """Base class; InstagramNotConfiguredError below is what actually gets raised."""
    pass


class InstagramNotConfiguredError(InstagramNotYetSupportedError):
    pass


def extract(url: str, output_dir: str):
    if not os.path.exists(COOKIES_PATH):
        raise InstagramNotConfiguredError(
            "Instagram requires an authenticated session - anonymous access is "
            "refused by Instagram itself, not just unreliable. To enable it: "
            "(1) log into a throwaway/secondary Instagram account (not your "
            "personal one) in a browser, (2) export that session's cookies "
            "using a browser extension (e.g. \"Get cookies.txt LOCALLY\"), "
            f"(3) save the file to {COOKIES_PATH}"
        )

    video_path, info = download_with_ytdlp(
        url, output_dir, filename_prefix="instagram", extra_opts={"cookiefile": COOKIES_PATH}
    )
    return {
        "video_path": video_path,
        "title": info.get("title") or "",
        "caption": info.get("description") or "",
        "duration": info.get("duration"),
    }
