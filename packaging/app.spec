# PyInstaller spec for the desktop apps: a folder holding "<app name>.exe" and _internal\.
#
#   --profile forecast  Antibiotic Resistance Forecast (desktop.py), with the bundled workbook and its seed cache
#   --profile entry     Antibiotic Resistance Forecast Studio (entry_desktop.py), which starts empty
#
# Build with build_app.ps1, which first trains the seed cache (build\seed_cache) and fetches the report fonts.
# Run directly:  pyinstaller packaging\app.spec --noconfirm --distpath dist --workpath build\pyinstaller -- --profile entry

import argparse
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

PROFILES = {
    "forecast": {"name": "Antibiotic Resistance Forecast", "script": "desktop.py", "page": "index.html",
                 "icon": "app.ico", "bundled": True},
    "entry": {"name": "Antibiotic Resistance Forecast Studio", "script": "entry_desktop.py", "page": "studio.html",
              "icon": "entry.ico", "bundled": False},
}
parser = argparse.ArgumentParser()
parser.add_argument("--profile", choices=PROFILES, default="forecast")
PROFILE = PROFILES[parser.parse_args().profile]

ROOT = Path(SPECPATH).parent
SITE = Path(sys.prefix) / "Lib" / "site-packages"
APP_NAME = PROFILE["name"]
SEED_CACHE = ROOT / "build" / "seed_cache"
FONTS = ROOT / "assets" / "fonts" / "fonts.css"

for required in ([SEED_CACHE] if PROFILE["bundled"] else []) + [FONTS]:
    if not required.exists():
        raise SystemExit(f"{required} is missing: run build_app.ps1, which creates it before packaging.")

datas = [
    (str(ROOT / "static" / PROFILE["page"]), "static"),
    (str(FONTS), "assets/fonts"),
    # Prophet's compiled Stan model and the CmdStan folder cmdstanpy checks for, copied as they are. Its
    # tbb.dll must sit beside prophet_model.bin (see use_bundled_tbb in build_forecast_dashboard.py); copy it
    # there at build time, because Program Files is read-only once installed.
    (str(SITE / "prophet" / "stan_model"), "prophet/stan_model"),
    (str(next((SITE / "prophet" / "stan_model").glob("cmdstan-*/stan/lib/stan_math/lib/tbb/tbb.dll"))),
     "prophet/stan_model"),
]
if PROFILE["bundled"]:  # the Studio starts empty: no workbook, no trained models
    datas += [(str(ROOT / "antibiotic trend (2022-2025).xlsx"), "."), (str(SEED_CACHE), "seed_cache")]
datas += collect_data_files("xgboost")  # its VERSION file
binaries = collect_dynamic_libs("xgboost")
hiddenimports = collect_submodules("uvicorn") + ["prophet.models", "cmdstanpy"]

# Never imported when the app runs (Prophet guards its plotting imports); torch._dynamo does need sympy.
excludes = [
    "matplotlib", "plotly", "PIL", "fontTools", "networkx", "tkinter", "_tkinter", "IPython", "jupyter_client",
    "ipykernel", "ipywidgets", "notebook", "pytest", "pip", "setuptools", "pkg_resources", "torch.utils.tensorboard",
    "tensorboard", "torchvision", "torchaudio",
]

a = Analysis(
    [str(ROOT / PROFILE["script"])],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)


def unused(dest: str) -> bool:
    """Files the app never loads: torch's C++ headers and link libraries, and test folders."""
    p = dest.replace("\\", "/").lower()
    return (p.startswith("torch/include/") or (p.startswith("torch/lib/") and p.endswith(".lib"))
            or "/tests/" in p or p.startswith("torch/share/"))


a.datas = [entry for entry in a.datas if not unused(entry[0])]
a.binaries = [entry for entry in a.binaries if not unused(entry[0])]

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    icon=str(ROOT / "assets" / PROFILE["icon"]),
    console=False,
    upx=False,  # UPX corrupts torch's DLLs and makes antivirus suspicious
)
coll = COLLECT(exe, a.binaries, a.datas, name=APP_NAME, upx=False)
