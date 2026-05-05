# Manual: Reproducing `frame_images`

This document describes how to reproduce the `frame_images` folder from the original
test videos using the provided script.

---

## Quick start (recommended)

Open a terminal in `SETUP_frame_images_extraction\` and run:

*make sure you have a the videos unpacked to ChannelPositionDetection\ **videos, NOT** ChannelPositionDetection\ **videos_pw** .*

```
python extract_frame_images.py
```

That's it. The script will:
1. Download `ffmpeg.exe` automatically if it is not already present (requires internet).
2. Create `frame_images\1\` through `frame_images\11\` inside `Enz_Positionserkennung\`.
3. Extract every frame from each video as `1.png`, `2.png`, … into the matching subfolder.
4. Delete the leading frames so each subfolder starts at the frame shown in the table below.

The script is **idempotent** — re-running it skips any channel whose frames are already extracted.


---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.8+ | Standard library only — no pip installs needed |
| Internet access | Only needed once, to download `ffmpeg.exe` (~90 MB zip) |
| Original videos | Must be in `D:\EESTEC_hack_2026\Enz_Positionserkennung\videos\` |

---

## Expected directory structure

```
D:\EESTEC_hack_2026\
└── ChannelPositionDetection\
    ├── videos\
    │   ├── 20230810_081214_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test1.mp4
    │   ├── ... (Test2 through Test11)
    ├── SETUP_frame_images_extraction\   ← run the script from here
    │   ├── extract_frame_images.py
    │   ├── extract_frame_images.bat
    │   ├── MANUAL_to_extract_frame_images.md
    │   └── ffmpeg.exe                   ← downloaded automatically on first run
    └── frame_images\                    ← created by the script
        ├── 1\   (180.png, 181.png, ...)
        ├── 2\   (56.png, 57.png, ...)
        └── ...
```

Each subfolder of `frame_images` corresponds to one test video. The first N frames of
each video are discarded; the table below shows which frame number each subfolder starts at.

| Channel | First kept frame |
|---------|-----------------|
| 1       | 180             |
| 2       | 56              |
| 3       | 58              |
| 4       | 30              |
| 5       | 55              |
| 6       | 58              |
| 7       | 330             |
| 8       | 100             |
| 9       | 35              |
| 10      | 42              |
| 11      | 314             |

---

## Offline / no-internet setup

If the machine has no internet access, place `ffmpeg.exe` manually next to the script:

```
...\ChannelPositionDetection\SETUP_frame_images_extraction\ffmpeg.exe
```

Download the **essentials build** from `https://www.gyan.dev/ffmpeg/builds/` on another
machine, extract the zip, and copy only `ffmpeg.exe` from the `bin\` subfolder.
Alternatively, install FFmpeg system-wide and add its `bin\` folder to `PATH` — both
scripts will find it automatically.
