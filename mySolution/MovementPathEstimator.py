import os
import cv2
import glob
import numpy as np
import torch
import scipy.signal
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from typing import Tuple
from MovementPath import MovementPath

import scipy.ndimage


class MovementPathEstimator:
    def __init__(self, video_num_to_test: int, test_all_videos: bool):
        self.channel_lengths = np.load('channel_lengths.npy')
        self.test_all_videos = test_all_videos
        self.video_num_to_test = video_num_to_test

        self.path_to_videos = 'frame_images/'
        self.current_folder = os.path.dirname(os.path.abspath(__file__)) + os.sep
        self.calculated_movement_paths = {}

        # --- Deep Learning Setup ---
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Loading RAFT Optical Flow Model on {self.device}...")
        
        # We use raft_small for speed. For maximum accuracy, change to raft_large and Raft_Large_Weights.
        weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=weights, progress=False).to(self.device)
        self.model.eval()
        self.transforms = weights.transforms()

    def preprocess_image(self, img_path: str, target_size=(256, 256)) -> torch.Tensor:
        """Loads and preps image for RAFT (expects RGB, resized, normalized tensor)."""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        # Convert to tensor and scale to [−1.0, 1.0] as expected by RAFT
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor * 2.0 - 1.0
       
       
        
        return img_tensor.unsqueeze(0).to(self.device)

    def calculate_movement_path_and_turning_point(self, video_number: int, channel_length: float) -> Tuple[np.ndarray, float, np.ndarray]:
        # 1. Setup paths and load frame sequence
        video_dir = os.path.join(self.path_to_videos, str(video_number))
        frame_files = sorted(glob.glob(os.path.join(video_dir, "*.png")), key=lambda x: int(os.path.basename(x).split('.')[0]))
        num_frames = len(frame_files)
        
        if num_frames == 0:
            print(f"Warning: No frames found for video {video_number}")
            return np.zeros(0), 0.0, np.zeros(0)

        H, W = 256, 256
        
        # 2. Pre-calculate coordinate grids and masks on the GPU 
        y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        y_grid = y_grid.to(self.device).float()
        x_grid = x_grid.to(self.device).float()
        
        center_x, center_y = W / 2.0, H / 2.0
        
        dx = x_grid - center_x
        dy = y_grid - center_y
        dist = torch.sqrt(dx**2 + dy**2) + 1e-6 
        radial_x = dx / dist
        radial_y = dy / dist
        
        # Environmental Mask ("Annulus")
        mask = torch.ones((H, W), device=self.device, dtype=torch.bool)
        mask[int(H * 0.6):, :] = False  # Water/turbulence
        mask[dist < (W * 0.20)] = False # Center void
        mask[dist > (W * 0.45)] = False # Outer edge distortion
        lk_mask = (mask.cpu().numpy() * 255).astype(np.uint8)

        # ---------------------------------------------------------
        # 3. FAST I/O: Pre-load video into RAM
        # ---------------------------------------------------------
        print(f"Loading {num_frames} frames into RAM for Video {video_number}...")
        frames_gray = []
        tensor_list = []
        
        for f in frame_files:
            img = cv2.imread(f)
            img_resized = cv2.resize(img, (W, H))
            frames_gray.append(cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY))
            
            # PyTorch Prep
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor * 2.0 - 1.0
            tensor_list.append(img_tensor)

        # Stack into one giant tensor in CPU RAM: [N, 3, H, W]
        video_tensor = torch.stack(tensor_list)

        raw_raft_velocities = np.zeros(num_frames)
        raw_lk_velocities = np.zeros(num_frames)
        
        # ---------------------------------------------------------
        # 4. STREAM 1: Batched RAFT (Dense) on GPU
        # ---------------------------------------------------------
        batch_size = 32  # Decrease to 16 if you run out of VRAM
        print(f"  -> Processing RAFT (GPU Batched)...")
        
        with torch.no_grad():
            for start_idx in range(0, num_frames - 1, batch_size):
                end_idx = min(start_idx + batch_size, num_frames - 1)
                
                # Push only current batch to GPU
                img1_batch = video_tensor[start_idx : end_idx].to(self.device)
                img2_batch = video_tensor[start_idx + 1 : end_idx + 1].to(self.device)
                
                flow_predictions = self.model(img1_batch, img2_batch)
                flow_batch = flow_predictions[-1] 
                
                u = flow_batch[:, 0, :, :]
                v = flow_batch[:, 1, :, :]
                
                radial_flow = u * radial_x + v * radial_y
                
                # Vectorized masking and median calculation
                masked_flow = radial_flow[:, mask] 
                batch_medians, _ = torch.median(masked_flow, dim=1) 
                
                raw_raft_velocities[start_idx + 1 : end_idx + 1] = batch_medians.cpu().numpy()

        # ---------------------------------------------------------
        # 5. STREAM 2: Lucas-Kanade (Sparse) on CPU
        # ---------------------------------------------------------
        print("  -> Processing Lucas-Kanade (CPU)...")
        feature_params = dict(maxCorners=200, qualityLevel=0.05, minDistance=7, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        for i in range(1, num_frames):
            old_gray = frames_gray[i-1]
            frame_gray = frames_gray[i]
            
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=lk_mask, **feature_params)
            lk_velocity = 0.0
            
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    
                    if len(good_new) > 0:
                        dx_lk = good_new[:, 0] - good_old[:, 0]
                        dy_lk = good_new[:, 1] - good_old[:, 1]
                        
                        vx = good_old[:, 0] - center_x
                        vy = good_old[:, 1] - center_y
                        pt_dist = np.sqrt(vx**2 + vy**2) + 1e-6
                        rx_lk = vx / pt_dist
                        ry_lk = vy / pt_dist
                        
                        radial_movements = dx_lk * rx_lk + dy_lk * ry_lk
                        lk_velocity = np.median(radial_movements)
            
            raw_lk_velocities[i] = lk_velocity
                
        # ---------------------------------------------------------
        # 6. Filter, Normalize Streams, and Ensemble
        # ---------------------------------------------------------
        window_size = 11
        smoothed_raft = scipy.signal.medfilt(raw_raft_velocities, kernel_size=window_size)
        smoothed_lk = scipy.signal.medfilt(raw_lk_velocities, kernel_size=window_size)
        
        raft_max = np.max(np.abs(smoothed_raft)) or 1.0
        lk_max = np.max(np.abs(smoothed_lk)) or 1.0
        
        raft_norm = smoothed_raft / raft_max
        lk_norm = smoothed_lk / lk_max
        
        combined_velocities = (raft_norm + lk_norm) / 2.0

        # ---------------------------------------------------------
        # 7. Asymmetric Motion State Detection
        # ---------------------------------------------------------
        noise_floor = np.percentile(np.abs(combined_velocities), 20)
        forward_threshold = max(0.04, noise_floor * 2.0) 
        backward_threshold = forward_threshold * 0.6 # More sensitive to backward drag
        
        states = np.zeros(num_frames, dtype=int)
        states[combined_velocities > forward_threshold] = 1   
        states[combined_velocities < -backward_threshold] = -1 
        
        filtered_velocities = combined_velocities.copy()
        filtered_velocities[states == 0] = 0.0

        # Debug Prints
        num_forward = np.sum(states == 1)
        num_backward = np.sum(states == -1)
        num_stat = np.sum(states == 0)
        print(f"  -> States Detected: {num_forward} Forward, {num_backward} Backward, {num_stat} Stationary")

        # ---------------------------------------------------------
        # 8. Path Integration (Constant Velocity + Return Anchor)
        # ---------------------------------------------------------
        # Simulate mechanical acceleration of the crawler
        smoothed_states = scipy.ndimage.gaussian_filter1d(states.astype(float), sigma=15)
        raw_path = np.cumsum(smoothed_states)
        
        turning_point_idx = np.argmax(raw_path)
        turning_point = float(turning_point_idx)
        
        max_dist = raw_path[turning_point_idx]
        
        if max_dist > 0:
            scale_factor = channel_length / max_dist
            movement_path = raw_path * scale_factor
        else:
            movement_path = np.zeros(num_frames)
            
        # The Return Trip Anchor: Force the end of the line down to 0 meters
        ending_error = movement_path[-1]
        if ending_error > 0.5: 
            frames_remaining = num_frames - turning_point_idx
            if frames_remaining > 0:
                correction_slope = np.linspace(0, ending_error, frames_remaining)
                movement_path[turning_point_idx:] -= correction_slope

        movement_path = np.clip(movement_path, 0, channel_length)

        # Hackathon Bonus Call
        try:
            self._calculate_bonus_events(video_number, filtered_velocities, turning_point)
        except AttributeError:
            pass 

        print("-" * 40)
        return movement_path, turning_point, states

    def _calculate_bonus_events(self, video_number, states, turning_point):
        """Calculates and prints bonus challenge data based on the cleaned binary states."""
        
        drop_frame = 0
        for i in range(len(states)):
            if states[i] == 1: 
                if np.sum(states[i:i+15] == 1) >= 10: 
                    drop_frame = i
                    break
                    
        stall_zones = []
        in_stall = False
        stall_start = 0

        for i in range(drop_frame, int(turning_point)):
            if states[i] != 1 and not in_stall:
                in_stall = True
                stall_start = i
            elif states[i] == 1 and in_stall:
                in_stall = False
                stall_length = i - stall_start
                if stall_length > 30: # Only count severe stalls (e.g., > 1 second at 30fps)
                    stall_zones.append((stall_start, i))
                    
        print(f"--- Video {video_number} Bonus Data ---")
        print(f"  Drop Frame: {drop_frame}")
        print(f"  Stall Zones: {stall_zones if stall_zones else 'None detected'}")
        print("-" * 30)

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