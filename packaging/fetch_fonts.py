"""Save the summary report's fonts (Fraunces, Nunito) as assets/fonts/fonts.css, with the font files embedded.

build_forecast_pdf.py embeds this file in the report when it exists, so the PDF looks the same on a PC with
no internet connection. Only the Latin subsets are kept. Run once at build time (build_app.ps1 does).
"""

import base64
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_forecast_pdf import FONT_CSS, FONTS  # noqa: E402

# Google Fonts serves woff2 only to browsers it recognises.
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
KEEP = {"latin", "latin-ext"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as r:
        return r.read()


def main() -> None:
    css = get(FONTS).decode("utf-8")
    # Each @font-face block is preceded by a comment naming its subset: /* latin */
    blocks = re.findall(r"/\* ([\w-]+) \*/\s*(@font-face \{.*?\})", css, flags=re.S)
    kept = []
    for subset, block in blocks:
        if subset not in KEEP:
            continue
        url = re.search(r"url\((https://[^)]+)\)", block).group(1)
        data = base64.b64encode(get(url)).decode("ascii")
        kept.append(f"/* {subset} */\n" + block.replace(url, f"data:font/woff2;base64,{data}"))
    if not kept:
        raise SystemExit("Google Fonts returned no Latin fonts; the report will fall back to the online link.")
    FONT_CSS.parent.mkdir(parents=True, exist_ok=True)
    FONT_CSS.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"Wrote {FONT_CSS.relative_to(ROOT)} ({len(kept)} font faces, {FONT_CSS.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
