import numpy as np
import matplotlib.pyplot as plt

from EstimationRater import EstimationRater
from MovementPath import MovementPath
from mySolution.MovementPathEstimator import MovementPathEstimator

# ---- configure which video to run ----
video_num_to_test = 1
# --------------------------------------

estimator = MovementPathEstimator(video_num_to_test, False)
estimator.execute_estimations()

result = estimator.calculated_movement_paths[video_num_to_test]

# EstimationRater expects key 0 in single-video mode
import os
ground_truth_path = f'distance_labels/{video_num_to_test}.npy'
measured = {}
if os.path.exists(ground_truth_path):
    measured = {0: MovementPath(video_num_to_test, np.load(ground_truth_path))}
else:
    print(f"No ground-truth label for video {video_num_to_test} — skipping scoring.")
rater = EstimationRater(
    {0: result},
    measured,
    do_print=bool(measured),
    do_plot=bool(measured),
    estimator_name=type(estimator).__name__,
    video_num=video_num_to_test,
)
rater.rate()
