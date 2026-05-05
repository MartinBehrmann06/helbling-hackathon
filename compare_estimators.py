import numpy as np
import os

from EstimationRater import EstimationRater
from MovementPath import MovementPath
from mySolution.MovementPathEstimator import MovementPathEstimator

test_all_videos = True  # runs all videos found in frame_images/

estimator = MovementPathEstimator(-1, test_all_videos)
estimator.execute_estimations()

# load ground-truth smoothed measurements
measured_movement_paths = {}
for fname in os.listdir('distance_labels/'):
    if not fname.endswith('.npy'):
        continue
    video_num = int(fname.split('.')[0])
    measured_movement_paths[video_num] = MovementPath(
        video_num, np.load(f'distance_labels/{fname}')
    )

rater = EstimationRater(
    estimator.calculated_movement_paths,
    measured_movement_paths,
    do_print=True,
    do_plot=True,
    estimator_name=type(estimator).__name__,
    video_num=-1,
)
rater.rate()
