# Channel Position Detection — Hackathon

Estimate the position of an object moving through a channel, frame by frame, based on video data.

  Core challenge: Given video footage of an object (likely a probe or sensor) moving through a pipe/channel, estimate its position frame-by-frame using computer vision.
                                                                                                                                                                                 
  What you implement: A single method in mySolution/MovementPathEstimator.py that, for each video, returns:
  - movement_path — position in meters per frame (0 → channel_length and back)
  - turning_point — the frame where the object reverses direction
  - movement_direction — +1 forward / -1 backward / 0 stationary per frame

  The data: 11 videos (already extracted as frame images), each showing an object traveling through a channel of known physical length. Some videos have ground-truth labels
  (distance_labels/), others are withheld for scoring.

  Scoring: Turning point frame error, position MAE (normalized by channel length), and directional accuracy.

  Context clues: The project name is "ChannelPositionDetection". This is about tracking a camera or probe moving through a sewage pipe or water channel infrastructure — a real-world inspection use case.

  Bonus challenges (optional, but can gain points): Gravel, Lime deposit, Roots, drop location detection, channel geometry changes, visibility scoring, etc.

  The starting point is a dummy triangle-wave implementation that you replace with actual computer vision logic.

  **Important: all files you create and need must reside inside your 'mySolution' folder.**
---

## Quick start

### 1. Install Python 3.11+

Download from [python.org](https://www.python.org/downloads/) and verify:
```bash
python --version
```

### 2. Install uv

```bash
pip install uv
```

Or on Windows:
```bash
winget install astral-sh.uv
```

### 3. Set up the workspace

Clone or copy this repository, then navigate into the project folder:

```bash
cd ChannelPositionDetection
```

Install dependencies:

```bash
uv sync
```

### 4. Prepare the frame images

videos_pw.zip: Password-protected ZIP archives can be downloaded here:

https://www.swisstransfer.com/d/4525aee3-72f8-4a58-b2c0-e0d73fa00f8c


You will receive a password-protected ZIP containing the videos. Extract it with the provided password so that the video files land in a `videos/` folder next to this README (not `videos_pw/`):

```
ChannelPositionDetection/
└── videos/
    ├── ...Test1.mp4
    ├── ...Test2.mp4
    └── ...
```

Then run the extraction script from the repo root:

```bash
python SETUP_frame_images_extraction/extract_frame_images.py
```

This downloads `ffmpeg` automatically (requires internet on first run), then extracts every frame from each video into `frame_images/1/`, `frame_images/2/`, … The script is idempotent — re-running it skips channels already extracted.

For the bonus videos, extract the bonus ZIP into a `bonus_videos/` folder alongside `videos/`, then run:

```bash
python SETUP_frame_images_extraction/extract_bonus_frame_images.py
```

---

## Your task

Open `mySolution/MovementPathEstimator.py` and implement the method:

```python
def calculate_movement_path_and_turning_point(self, video_number, channel_length):
```

**Inputs:**
- `video_number` — which video (1-based index)
- `channel_length` — physical length of the channel in metres
- `self.path_to_videos + str(video_number)` — folder containing the frame images (`0.png`, `1.png`, ...)

**Return a tuple of three arrays:**
- `movement_path` — `np.ndarray` shape `(N,)`, position in `[0, channel_length]` per frame
- `turning_point` — `float`, frame index where the object reverses direction
- `movement_direction` — `np.ndarray` shape `(N,)`, `+1` forward / `-1` backward / `0` stationary per frame

The file already contains a dummy implementation (a simple triangle wave) as a starting point — replace it with your solution.

---

## Testing your solution

Run a single video and see your score immediately:

```bash
uv run python estimate_single_movement_path.py
```

Change `video_num_to_test` at the top of the file to test a different video (1–11).

To test all videos at once:

```bash
uv run python compare_estimators.py
```

**Scoring metrics:**
| Metric | Direction | Description |
|---|---|---|
| TP error [frames] | lower is better | Absolute error on the turning point frame |
| Path MAE [m] | lower is better | Mean absolute position error, normalized by channel length |
| Dir accuracy | higher is better | Fraction of frames with correct movement direction |

---

## Using `distance_labels/` as training data

If you decide to create a ML-based solution, you may use  `distance_labels/` as training data. However, only a subset of the label files is provided — the rest are withheld as a **held-out test set** used by the organizers to score your solution. This prevents **overfitting**: if all labels were available, a model could simply memorize the exact output for each video rather than learning to generalize to unseen inputs.

We do not necessary encourage you to create a ML-based solution. The training set might be too small. 

Your solution must therefore work on any video, including ones for which no label file exists in `distance_labels/`.

---

## Repository structure

```
mySolution/
  MovementPathEstimator.py   ← implement your solution here
  results/                   ← output written here automatically
frame_images/                ← copy separately, not in repo
  {1..11}/                   ← video frames (0.png, 1.png, ...)
distance_labels/             ← ground-truth position curves (for scoring only, see below)
channel_lengths.npy          ← physical channel lengths per video
EstimationRater.py           ← scoring framework (do not modify)
MovementPath.py              ← data class (do not modify)
estimate_single_movement_path.py   ← run & score one video
compare_estimators.py              ← run & score all videos
```

## Additional Challenges

Bonus videos are labeled with timestamped obstruction annotations (Gravel, Lime deposit, Sludge, Sand, Roots, Grease, Concrete; severity 1–3) and visibility events (No Vision (Lens covered / Splash Water / Under Water), No Obstruction). 

Also additional detections you could attempt:

  Position & motion (already covered)
  - Movement path (position over time)
  - Movement direction per frame

  Channel events
  - Drop locations — frame where the object is released / enters the channel
  - Retrieval frame — where it exits or stops moving
  - Stall zones — frames where motion pauses mid-channel (not a full reversal)

  Water / flow conditions
  - Flow direction (is water pushing the object forward or against it?)
  - Flow speed estimate (how fast the object drifts when not actively moving)

  Object / debris detection
  - Dirty locations — frame ranges or positions where sediment, blockage, or debris is visible
  - Blockage degree — how much of the channel cross-section is obstructed (0–100%)
  - Object shape change — deformation or tumbling events

  Channel geometry
  - Junction detection — frames where the channel branches or joins another
  - Diameter change — frames where the channel narrows or widens noticeably
  - Inclination change — transition from flat to sloped section

  Practical difficulty
  - Visibility score per frame — how clear/murky the water is
  - Camera shake / stabilization quality

  The most tractable ones for a hackathon (clear ground truth, derivable from video alone) are probably drop location, stall zones, blockage degree, and dirty locations — these have obvious visual signatures and could be labeled from the same videos you already have.