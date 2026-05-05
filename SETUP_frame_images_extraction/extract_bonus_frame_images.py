"""
Reproduces the bonus_frame_images folder from the bonus videos.
See MANUAL_to_extract_frame_images.md for the expected directory structure
and instructions to obtain ffmpeg.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_frame_images import find_ffmpeg  # reuse ffmpeg detection / download

VIDEOS_DIR = Path(__file__).parent.parent / "bonus_videos"
OUT_DIR = Path(__file__).parent.parent / "bonus_frame_images"

VIDEOS = {
    1:  "20230914_142808_NEX2E2023_SENECAMEADOWS_NEX2E2023_NEX2E2023.mp4",
    2:  "20240119_083323_Pieterlen_ASTRA_2_1.mp4",
    3:  "20240315_155426_HillParkRdbrixham_sww_pre_line1and2.mp4",
    4:  "20240606_103810_stanboroughroad_sww_pre_lines7and8.mp4",
    5:  "20240813_133337_3600Munkwitz_engineering_66_3700Munkwitz.mp4",
    6:  "20240826_132001_fulleravenw_CityofMassillon_03_47728thnw.mp4",
    7:  "20240903_034835_LainzertunnelGleis7_OeBB_46_LT330477.mp4",
    8:  "20240906_022339_Lainzertunnel_OeBB_48_LT313377.mp4",
    9:  "20240906_095457_Kampstrasse_Vardeilsen_Regenkakal_Friedhof.mp4",
    10: "20240908_015511_Lainzertunnel_OeBB_48_LT313219.mp4",
    11: "20240926_100453_A2_A2_92624_7470696.mp4",
    12: "20240927_091832_202WLockheed_Jacobgentry_1_E2074.mp4",
    13: "20240930_220035_EASTVILEMULLERROAD_wessex_5667469000277_ST61750245.mp4",
    14: "20241001_143649_9713WoodrockPl_JacobGentry_1_D4013.mp4",
    15: "20241003_133947_SMilmoAveClevelandSt_SantoNioNE_DiazStSMilmoAve_MH9139.mp4",
    16: "20241008_102749_2803DiazSt_SantoNioNE_2715DiazSt_MH11368.mp4",
    17: "20241009_101753_schollsferry_Beaverton_79793_swgm0019061.mp4",
    18: "20241010_100916_2604NTowry_JacobGentry_2_D3210.mp4",
    19: "20241015_123954_mormoironcarboussan_Suez_1_12.mp4",
    20: "20241017_103717_1_DSRSD_1_1.mp4",
}


def extract_frames(ffmpeg: str, channel: int) -> None:
    video = VIDEOS_DIR / VIDEOS[channel]
    out_dir = OUT_DIR / str(channel)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-i", str(video), "-start_number", "0", str(out_dir / "%d.png")],
        check=True,
    )


def main() -> None:
    ffmpeg = find_ffmpeg()
    total = len(VIDEOS)

    for channel, filename in VIDEOS.items():
        out_dir = OUT_DIR / str(channel)
        if out_dir.exists() and any(out_dir.glob("*.png")):
            print(f"[{channel}/{total}] Channel {channel} already extracted, skipping.")
            continue
        print(f"\n[{channel}/{total}] Extracting frames for channel {channel} ...")
        extract_frames(ffmpeg, channel)

    print(f"\nDone. bonus_frame_images is ready at: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
