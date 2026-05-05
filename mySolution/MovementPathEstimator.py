import numpy as np
import os
import cv2
from scipy.signal import savgol_filter

from MovementPath import MovementPath


class MovementPathEstimator:
    """
    Hackathon template: estimate the movement path and turning point
    of an object moving through a channel, based on video frames.

    The framework calls `execute_estimations()` which in turn calls
    `calculate_movement_path_and_turning_point()` for each video.
    Your job is to implement that one method.

    Inputs available inside `calculate_movement_path_and_turning_point`:
      - video_number   : int, which video (1-based)
      - channel_length : float, the physical length of the channel [m]
      - path_to_video  : str, folder containing the frame images (0.png, 1.png, ...)

    Outputs to return (as a tuple):
      - movement_path      : np.ndarray shape (N,)  position in [0, channel_length] per frame
      - turning_point      : float                  frame index where the object reverses
      - movement_direction : np.ndarray shape (N,)  +1 forward, -1 backward, 0 stationary per frame
    """

    def __init__(self, video_num_to_test, test_all_videos):
        self.channel_lengths = np.load('channel_lengths.npy')
        self.test_all_videos = test_all_videos
        self.video_num_to_test = video_num_to_test

        self.path_to_videos = 'frame_images/'
        # Always points to whichever folder this file lives in,
        # so the estimator works regardless of the folder name.
        self.current_folder = os.path.dirname(os.path.abspath(__file__)) + os.sep

        self.calculated_movement_paths = {}

    # ------------------------------------------------------------------ #
    #  TODO: implement your solution here                                  #
    # ------------------------------------------------------------------ #
    def calculate_movement_path_and_turning_point(self, video_number, channel_length):
        path_to_video = os.path.join(self.path_to_videos, str(video_number))
        
        frame_files = [f for f in os.listdir(path_to_video) if f.endswith('.png')]
        frame_files.sort(key=lambda x: int(x.split('.')[0]))
        num_frames = len(frame_files)
        
        if num_frames == 0:
            return np.zeros(0), 0.0, np.zeros(0)
            
        velocities = np.zeros(num_frames)
        
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        feature_params = dict(maxCorners=150, qualityLevel=0.05, minDistance=15, blockSize=7)

        first_frame_path = os.path.join(path_to_video, frame_files[0])
        old_frame_full = cv2.imread(first_frame_path, cv2.IMREAD_GRAYSCALE)
        
        scale = 0.5
        old_frame = cv2.resize(old_frame_full, (0, 0), fx=scale, fy=scale)
        h, w = old_frame.shape
        cx, cy = w / 2.0, h / 2.0 
        
        # ---------------------------------------------------------
        # TWEAK 1: "The Dry Wall" Mask
        # ---------------------------------------------------------
        mask = np.zeros_like(old_frame)
        cv2.circle(mask, (int(cx), int(cy)), int(min(h, w) * 0.45), 255, -1) 
        cv2.circle(mask, (int(cx), int(cy)), int(min(h, w) * 0.20), 0, -1)   
        
        # NEW: Mask out the bottom 40% of the frame entirely.
        # This prevents tracking water, sludge, and dragging cables that 
        # cause the "False Forward" bumps during the backward pull.
        bottom_cutoff = int(h * 0.60)
        cv2.rectangle(mask, (0, bottom_cutoff), (w, h), 0, -1)
        # ---------------------------------------------------------
        
        p0 = cv2.goodFeaturesToTrack(old_frame, mask=mask, **feature_params)

        for i in range(1, num_frames):
            frame_path = os.path.join(path_to_video, frame_files[i])
            frame_full = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            frame = cv2.resize(frame_full, (0, 0), fx=scale, fy=scale)
            
            # Smart Feature Refresh
            if p0 is None or len(p0) < 30:
                p0 = cv2.goodFeaturesToTrack(old_frame, mask=mask, **feature_params)
                
            if p0 is not None and len(p0) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)
                
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                if len(good_new) > 0:
                    rx = good_old[:, 0] - cx
                    ry = good_old[:, 1] - cy
                    radii = np.sqrt(rx**2 + ry**2)
                    
                    valid_mask = radii > (min(h, w) * 0.20)
                    
                    good_new = good_new[valid_mask]
                    good_old = good_old[valid_mask]
                    rx = rx[valid_mask]
                    ry = ry[valid_mask]
                    radii = radii[valid_mask]
                    
                    if len(good_new) >= 5:
                        dx = good_new[:, 0] - good_old[:, 0]
                        dy = good_new[:, 1] - good_old[:, 1]
                        
                        radial_velocities = (dx * rx + dy * ry) / radii
                        
                        raw_velocity = np.median(radial_velocities)
                        
                        # ---------------------------------------------------------
                        # TWEAK 2: Stationary Thresholding (Noise Floor)
                        # ---------------------------------------------------------
                        # Flatten out the peaks and stop drift when the camera stops.
                        noise_floor = 0.05 # Adjust this slightly if it zeros out too much real movement
                        if abs(raw_velocity) < noise_floor:
                            raw_velocity = 0.0
                        # ---------------------------------------------------------
                        
                        velocities[i] = np.clip(raw_velocity * 0.05, -0.3, 0.3)
                        
                        p0 = good_new.reshape(-1, 1, 2)
                    else:
                        p0 = None 
                else:
                    p0 = None

            old_frame = frame
            
        # Smooth velocities to remove micro-jitters
        kernel_size = 90
        if num_frames > kernel_size:
            velocities = np.convolve(velocities, np.ones(kernel_size)/kernel_size, mode='same')
            
        movement_path_raw = np.cumsum(velocities)
        movement_path_raw -= movement_path_raw[0] 
        
        turn_idx = int(np.argmax(movement_path_raw))
        turning_point = float(turn_idx)
        
        movement_path = np.zeros_like(movement_path_raw)
        
        # --- Scale Forward Pass ---
        forward_raw = movement_path_raw[:turn_idx + 1]
        peak_raw = forward_raw[-1]
        if peak_raw > 0.001:
            movement_path[:turn_idx + 1] = forward_raw * (channel_length / peak_raw)
            
        # --- Scale Backward Pass ---
        if turn_idx + 1 < num_frames:
            backward_raw = movement_path_raw[turn_idx:]
            raw_drop = backward_raw[0] - backward_raw[-1] 
            
            if raw_drop > 0.001:
                normalized_backward = channel_length - (backward_raw[0] - backward_raw) * (channel_length / raw_drop)
                movement_path[turn_idx:] = normalized_backward
            else:
                movement_path[turn_idx:] = channel_length
                
        movement_path = np.clip(movement_path, 0, channel_length)
        
        # Smooth final position data
        window_length = 151 
        polyorder = 2 
        if num_frames > window_length:
            movement_path = savgol_filter(movement_path, window_length, polyorder)
            movement_path = np.clip(movement_path, 0, channel_length)
            
        turning_point = float(np.argmax(movement_path))
        
        # Calculate directions based on the finalized path
        movement_direction = np.zeros(num_frames)
        diffs = np.diff(movement_path, prepend=0)
        threshold = 0.001 * channel_length
        movement_direction[diffs > threshold] = 1
        movement_direction[diffs < -threshold] = -1

        return movement_path, turning_point, movement_direction
    # ------------------------------------------------------------------ #
    #  Framework boilerplate – you should not need to change this          #
    # ------------------------------------------------------------------ #

    def execute_estimations(self):
        if self.test_all_videos:
            if not os.path.exists(self.path_to_videos):
                raise FileNotFoundError(f"The folder '{self.path_to_videos}' does not exist.")
            for entry in os.listdir(self.path_to_videos):
                if entry.isdigit():
                    self._run_single(int(entry))
        else:
            self._run_single(self.video_num_to_test)

    def _run_single(self, video_number):
        try:
            channel_length = self.channel_lengths[video_number - 1]
        except Exception:
            print("Cannot load channel length, using 100 m")
            channel_length = 100
        movement_path, turning_point, movement_direction = \
            self.calculate_movement_path_and_turning_point(int(video_number), channel_length)
        self.calculated_movement_paths[int(video_number)] = MovementPath(
            int(video_number), movement_path, movement_direction, turning_point
        )
