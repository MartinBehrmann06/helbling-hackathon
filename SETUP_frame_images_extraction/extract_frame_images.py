"""
Reproduces the frame_images folder from the original test videos.
See MANUAL_to_extract_frame_images.md for the expected directory structure
and instructions to obtain ffmpeg.
"""

import platform
import shutil
import ssl
import stat
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

FFMPEG_WINDOWS_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_MACOS_URL   = "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip"
FFMPEG_LINUX_URLS  = {
    "x86_64":  "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
    "aarch64": "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz",
}

VIDEOS_DIR = Path(__file__).parent.parent / "videos"
OUT_DIR = Path(__file__).parent.parent / "frame_images"

VIDEOS = {
    1:  "20230810_081214_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test1.mp4",
    2:  "20230810_083027_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test2.mp4",
    3:  "20230810_084551_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test3.mp4",
    4:  "20230810_090123_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test4neuerKanal.mp4",
    5:  "20230810_091120_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test5gleicherKanalwie4.mp4",
    6:  "20230810_092329_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test6neuerKanal.mp4",
    7:  "20230810_100306_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test7neuerSchacht.mp4",
    8:  "20230810_101624_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test8gleicherKanal.mp4",
    9:  "20230810_103619_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test9gleicherKanal.mp4",
    10: "20230810_104524_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test10gleicherKanal.mp4",
    11: "20230810_105533_ERZZuerich_EnzTechnikAG_TestMeterzaehler_Test11gleicherKanal.mp4",
}

# First frame to keep per channel; all frames before this number are deleted.
FIRST_KEPT_FRAME = {
    1: 180, 2: 56, 3: 58, 4: 30, 5: 55, 6: 58,
    7: 330, 8: 100, 9: 35, 10: 42, 11: 314,
}

# Last frame to keep per channel; frames after this number are deleted (None = keep all).
LAST_KEPT_FRAME = {8: 1600}


def _ssl_context() -> ssl.SSLContext:
    # On macOS python.org installs, the default context has no CA certs until
    # the user runs "Install Certificates.command". Using system certs avoids this.
    try:
        return ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    except Exception:
        return ssl.create_default_context()


def _urlretrieve(url: str, dest: Path) -> None:
    ctx = _ssl_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ctx) as response, open(dest, "wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as exc:
        sys.exit(f"ERROR: Download failed ({url}):\n  {exc}")


def _download_ffmpeg_windows() -> str:
    local = Path(__file__).parent / "ffmpeg.exe"
    local.unlink(missing_ok=True)
    zip_path = Path(__file__).parent / "ffmpeg_dl.zip"
    print("ffmpeg not found -- downloading from gyan.dev ...")
    try:
        _urlretrieve(FFMPEG_WINDOWS_URL, zip_path)
        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith("/ffmpeg.exe"):
                    with zf.open(name) as src, open(local, "wb") as dst:
                        dst.write(src.read())
                    break
    finally:
        if zip_path.exists():
            zip_path.unlink()
    if not local.exists():
        sys.exit("ERROR: Could not find ffmpeg.exe inside the downloaded archive.")
    print("ffmpeg downloaded successfully.")
    return str(local)


def _download_ffmpeg_macos() -> str:
    local = Path(__file__).parent / "ffmpeg"
    local.unlink(missing_ok=True)
    zip_path = Path(__file__).parent / "ffmpeg_dl.zip"
    if platform.machine() == "arm64":
        print("NOTE: No arm64 static build available for macOS. Downloading x86_64 binary (requires Rosetta 2).")
    print("ffmpeg not found -- downloading from evermeet.cx ...")
    try:
        _urlretrieve(FFMPEG_MACOS_URL, zip_path)
        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if Path(name).name == "ffmpeg":
                    with zf.open(name) as src, open(local, "wb") as dst:
                        dst.write(src.read())
                    break
    finally:
        if zip_path.exists():
            zip_path.unlink()
    if not local.exists():
        sys.exit("ERROR: Could not find ffmpeg inside the downloaded archive.")
    local.chmod(local.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    subprocess.run(["xattr", "-d", "com.apple.quarantine", str(local)], check=False, capture_output=True)
    print("ffmpeg downloaded successfully.")
    return str(local)


def _download_ffmpeg_linux() -> str:
    machine = platform.machine()
    url = FFMPEG_LINUX_URLS.get(machine)
    if url is None:
        sys.exit(f"ERROR: No static ffmpeg build available for Linux/{machine}. Install manually:\n  sudo apt install ffmpeg")
    local = Path(__file__).parent / "ffmpeg"
    local.unlink(missing_ok=True)
    tar_path = Path(__file__).parent / "ffmpeg_dl.tar.xz"
    print(f"ffmpeg not found -- downloading static build from johnvansickle.com ({machine}) ...")
    try:
        _urlretrieve(url, tar_path)
        print("Extracting ...")
        with tarfile.open(tar_path, "r:xz") as tf:
            for member in tf.getmembers():
                if member.isfile() and Path(member.name).name == "ffmpeg":
                    src = tf.extractfile(member)
                    if src:
                        with open(local, "wb") as dst:
                            dst.write(src.read())
                    break
    finally:
        if tar_path.exists():
            tar_path.unlink()
    if not local.exists():
        sys.exit("ERROR: Could not find ffmpeg inside the downloaded archive.")
    local.chmod(local.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print("ffmpeg downloaded successfully.")
    return str(local)


def find_ffmpeg() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"

    system = platform.system()
    local_name = "ffmpeg.exe" if system == "Windows" else "ffmpeg"
    local = Path(__file__).parent / local_name
    if local.exists():
        return str(local)

    if system == "Windows":
        return _download_ffmpeg_windows()
    if system == "Darwin":
        return _download_ffmpeg_macos()
    if system == "Linux":
        return _download_ffmpeg_linux()
    sys.exit(f"ERROR: Unsupported OS '{system}'. Install ffmpeg manually.")


def extract_frames(ffmpeg: str, channel: int) -> None:
    video = VIDEOS_DIR / VIDEOS[channel]
    out_dir = OUT_DIR / str(channel)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-i", str(video), "-start_number", "0", str(out_dir / "%d.png")],
        check=True,
    )


def delete_leading_frames(channel: int) -> None:
    out_dir = OUT_DIR / str(channel)
    last_to_delete = FIRST_KEPT_FRAME[channel] - 1
    deleted = 0
    for frame in range(0, last_to_delete + 1):
        f = out_dir / f"{frame}.png"
        if f.exists():
            f.unlink()
            deleted += 1
    print(f"  Channel {channel}: deleted frames 0–{last_to_delete} ({deleted} files)")


def delete_trailing_frames(channel: int) -> None:
    last = LAST_KEPT_FRAME.get(channel)
    if last is None:
        return
    out_dir = OUT_DIR / str(channel)
    deleted = 0
    for f in out_dir.glob("*.png"):
        if f.stem.isdigit() and int(f.stem) > last:
            f.unlink()
            deleted += 1
    print(f"  Channel {channel}: deleted frames after {last} ({deleted} files)")


def main() -> None:
    ffmpeg = find_ffmpeg()

    for channel in range(1, 12):
        out_dir = OUT_DIR / str(channel)
        if (out_dir / f"{FIRST_KEPT_FRAME[channel]}.png").exists():
            print(f"[{channel}/11] Channel {channel} already extracted, skipping.")
            continue
        print(f"\n[{channel}/11] Extracting frames for channel {channel} ...")
        extract_frames(ffmpeg, channel)
        delete_leading_frames(channel)
        delete_trailing_frames(channel)

    print(f"\nDone. frame_images is ready at: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
