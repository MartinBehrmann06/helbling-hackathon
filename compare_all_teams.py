"""
Hackathon leaderboard runner.

Drop each team's solution folder into the repo root (any name is fine,
as long as it contains a MovementPathEstimator.py).  Then run:

    uv run python compare_all_teams.py

The script discovers all team folders automatically, runs every estimator
against the ground-truth data, and prints a ranked leaderboard.

Caching: if a team's code hasn't changed since the last run, their saved
results are loaded from {team_folder}/results/ instead of recomputing.
Delete a team's results/ folder to force a fresh run.
"""

import hashlib
import importlib.util
import os
import sys

import numpy as np

from EstimationRater import EstimationRater
from MovementPath import MovementPath

# ── config ─────────────────────────────────────────────────────────────

SKIP_FOLDERS = {'mySolution', 'Template', '__pycache__'}
GROUND_TRUTH_DIR = 'distance_labels'
VIDEOS_DIR = 'frame_images'

# ── caching helpers ─────────────────────────────────────────────────────

def _hash_python_files(folder):
    sha256 = hashlib.sha256()
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.py'):
            with open(os.path.join(folder, fname), 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256.update(chunk)
    return sha256.hexdigest()


def _load_cached(results_folder, video_number):
    path_npy = os.path.join(results_folder, f'{video_number}_movement_path.npy')
    path_tp  = os.path.join(results_folder, f'{video_number}_turning_point.txt')
    path_dir = os.path.join(results_folder, f'{video_number}_movement_direction.npy')
    if os.path.exists(path_npy) and os.path.exists(path_tp) and os.path.exists(path_dir):
        movement_path      = np.load(path_npy)
        movement_direction = np.load(path_dir)
        with open(path_tp) as f:
            turning_point = float(f.read())
        return movement_path, turning_point, movement_direction
    return None


def _save_results(results_folder, video_number, movement_path, turning_point, movement_direction):
    os.makedirs(results_folder, exist_ok=True)
    np.save(os.path.join(results_folder, f'{video_number}_movement_path.npy'), movement_path)
    np.save(os.path.join(results_folder, f'{video_number}_movement_direction.npy'), movement_direction)
    with open(os.path.join(results_folder, f'{video_number}_turning_point.txt'), 'w') as f:
        f.write(str(turning_point))


def is_cache_valid(team_folder):
    hash_file = os.path.join(team_folder, 'code_hash.txt')
    current_hash = _hash_python_files(team_folder)
    if os.path.exists(hash_file):
        with open(hash_file) as f:
            if f.read() == current_hash:
                return True
    with open(hash_file, 'w') as f:
        f.write(current_hash)
    return False


# ── discovery & loading ─────────────────────────────────────────────────

def discover_team_folders(root='.'):
    teams = []
    for entry in sorted(os.listdir(root)):
        if entry in SKIP_FOLDERS:
            continue
        folder = os.path.join(root, entry)
        estimator_file = os.path.join(folder, 'MovementPathEstimator.py')
        if os.path.isdir(folder) and os.path.isfile(estimator_file):
            teams.append((entry, folder))
    return teams


def load_estimator_class(team_name, folder):
    spec = importlib.util.spec_from_file_location(
        f'{team_name}.MovementPathEstimator',
        os.path.join(folder, 'MovementPathEstimator.py'),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MovementPathEstimator


def load_ground_truth():
    measured = {}
    for fname in os.listdir(GROUND_TRUTH_DIR):
        if not fname.endswith('.npy'):
            continue
        video_num = int(fname.split('.')[0])
        measured[video_num] = MovementPath(
            video_num, np.load(os.path.join(GROUND_TRUTH_DIR, fname))
        )
    return measured


def video_numbers():
    return [int(e) for e in os.listdir(VIDEOS_DIR) if e.isdigit()]


# ── main ────────────────────────────────────────────────────────────────

def run_team(team_name, folder, videos, channel_lengths, use_cache):
    EstimatorClass = load_estimator_class(team_name, folder)
    estimator = EstimatorClass(-1, test_all_videos=True)
    results_folder = os.path.join(folder, 'results')

    cached = use_cache and is_cache_valid(folder)
    if cached:
        print(f"  code unchanged – loading cached results")

    for video_num in videos:
        if cached:
            hit = _load_cached(results_folder, video_num)
            if hit:
                movement_path, turning_point, movement_direction = hit
                estimator.calculated_movement_paths[video_num] = MovementPath(
                    video_num, movement_path, movement_direction, turning_point
                )
                continue
        # compute
        try:
            channel_length = channel_lengths[video_num - 1]
        except Exception:
            channel_length = 100
        movement_path, turning_point, movement_direction = \
            estimator.calculate_movement_path_and_turning_point(video_num, channel_length)
        estimator.calculated_movement_paths[video_num] = MovementPath(
            video_num, movement_path, movement_direction, turning_point
        )
        _save_results(results_folder, video_num, movement_path, turning_point, movement_direction)

    return estimator


def main():
    teams = discover_team_folders()
    if not teams:
        print("No team folders found. Each team folder must contain MovementPathEstimator.py.")
        sys.exit(1)

    print(f"Found {len(teams)} team(s): {[t for t, _ in teams]}\n")

    measured       = load_ground_truth()
    channel_lengths = np.load('channel_lengths.npy')
    videos         = video_numbers()

    leaderboard = {}

    for team_name, folder in teams:
        print(f"── {team_name} {'─' * 50}")
        try:
            estimator = run_team(team_name, folder, videos, channel_lengths, use_cache=True)

            rater = EstimationRater(
                estimator.calculated_movement_paths,
                measured,
                do_print=True,
                do_plot=False,
                estimator_name=team_name,
                video_num=-1,
            )
            rater.rate()

            leaderboard[team_name] = {
                'tp_error':   np.mean(rater.turning_point_scores),
                'path_error': np.mean(rater.movement_path_scores),
                'dir_score':  np.mean(rater.movement_direction_scores),
            }
        except Exception as exc:
            print(f"  ERROR: {exc}")
            leaderboard[team_name] = None
        print()

    # ── leaderboard ─────────────────────────────────────────────────────
    print("=" * 65)
    print(f"{'LEADERBOARD':^65}")
    print("=" * 65)
    print(f"{'Team':<20} {'TP error [frames]':>17} {'Path MAE [m]':>12} {'Dir accuracy':>13}")
    print("-" * 65)

    valid = [(n, r) for n, r in leaderboard.items() if r is not None]
    valid.sort(key=lambda x: x[1]['path_error'])

    for rank, (name, r) in enumerate(valid, 1):
        print(f"{rank}. {name:<18} {r['tp_error']:>17.1f} {r['path_error']:>12.3f} {r['dir_score']:>13.3f}")
    for name, r in leaderboard.items():
        if r is None:
            print(f"   {name:<18} {'ERROR':>17}")

    print("=" * 65)


if __name__ == '__main__':
    main()
