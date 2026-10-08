"""Antibiotic Resistance Forecast Studio - the desktop app that starts empty.

The user uploads a workbook or types the values in, reviews and confirms them, then picks one of 12 models - the
same 10 as the Antibiotic Resistance Forecast app (desktop.py), plus SARIMA and a SARIMA-LSTM hybrid - and sees its
forecasts and can save them, or a PDF report on them; only the picked model is trained. It is that app in the entry
profile (appdata.py), with its own page (static/studio.html), data folder and installer.

Run:  python entry_desktop.py                the app
      python entry_desktop.py --self-test --workbook FILE [--seed-cache DIR]
                                             review FILE as an upload, train on the confirmed data, check
                                             against the seed cache, write every export (used by the build)
"""

import appdata

appdata.use_profile("entry")

import desktop  # noqa: E402  (after the profile is chosen)

if __name__ == "__main__":
    desktop.main()
