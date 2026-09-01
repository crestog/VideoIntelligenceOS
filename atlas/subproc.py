"""Creation flags for every child process this application starts.

WHY THIS FILE EXISTS
════════════════════
The desktop shortcut runs `pythonw.exe -m desktop`, and `pythonw` has no
console — that is the whole reason it is used, and it is stated as such in
`desktop/__main__.py` and `desktop/make_shortcut.py`. What neither of them said
is what it does to children: when a process with no console starts a *console*
executable, Windows allocates one for it. Not a shared one, not a hidden one —
a new window, on screen, for as long as the child runs.

Every external tool this application drives is a console executable. `ffmpeg`,
`ffprobe`, `git`, `pg_dump`, `pg_restore`, `7z`, `wmic`. So a black window
appeared and vanished for each one, which is the reported symptom — windows
opening while searching — and the search path is the worst case rather than an
unlucky one: a page of results renders a poster per video, `media.poster()`
spawns one `ffmpeg` per poster, and they are not serialised. Ten results is ten
windows, and they arrive after the click, over the page the user is reading.

Two flags fix it, and they are separate because the reason for each is
different:

`CREATE_NO_WINDOW` is the fix for the windows. It tells Windows this child does
not get a console, which is what every one of these children wanted — nothing
reads their stdout from a terminal, the callers all pass `capture_output=True`
and read it as bytes.

`BELOW_NORMAL_PRIORITY_CLASS` is a fix for something else that shares the same
argument, so it is spelled separately rather than folded in. A derivation pass
is minutes of ffmpeg at full tilt; at normal priority it competes with the
window the user is typing into for the same cores. Below-normal means the batch
work yields to the interactive work rather than the scheduler splitting the
difference. It is *wrong* for the search path, where the child is the thing the
user is waiting on, which is why `FOREGROUND` exists and is not just an alias.

WHY IT LIVES IN `atlas/`
════════════════════════
`atlas/media.py` and `atlas/ingest.py` both spawn children, and a module inside
a package must not import from the application above it — that inverts the
dependency and breaks the Kaggle engine, which has this package and no desktop
around it. Being here means both forks can hold the same file.

On anything other than Windows both values are 0, which is `subprocess`'s own
default, so passing them changes nothing and no caller needs a platform test.
That is the point of exporting zeroes rather than exporting nothing: the call
sites stay identical across both forks and both platforms, and the Kaggle
engine — which runs on Linux and has no console to speak of — is unaffected by
every one of them.

Stdlib only, and deliberately: this is imported by `sizing/` and `capture/`,
which run before anything heavier is loaded.
"""

from __future__ import annotations

import subprocess
import sys

if sys.platform == "win32":
    # Not from `subprocess`: `BELOW_NORMAL_PRIORITY_CLASS` is not exposed there
    # (only `ABOVE_NORMAL_PRIORITY_CLASS`, `HIGH_PRIORITY_CLASS`, `IDLE_` and
    # `NORMAL_`), so the constant is written out. From winbase.h.
    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

    #: A child the user is waiting on. No console window, normal priority.
    FOREGROUND = subprocess.CREATE_NO_WINDOW

    #: A child doing batch work. No console window, and it yields to whatever
    #: the user is doing rather than competing with it.
    BACKGROUND = subprocess.CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
else:
    BELOW_NORMAL_PRIORITY_CLASS = 0
    FOREGROUND = 0
    BACKGROUND = 0
