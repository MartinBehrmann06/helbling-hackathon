import os
import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

from typing import Tuple

from MovementPath import MovementPath

class MovementPathEstimator:
    def __init__(self, video_num_to_test: int, test_all_videos: bool):
        self.channel_lengths = np.load('channel_lengths.npy')
        self.test_all_videos = test_all_videos
        self.video_num_to_test = video_num_to_test

        self.path_to_videos = 'frame_images/'
        self.current_folder = os.path.dirname(os.path.abspath(__file__)) + os.sep
        self.calculated_movement_paths = {}

        # --- Deep Learning Setup ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        path_to_video = os.path.join(self.path_to_videos, str(video_number))
        
        if not os.path.exists(path_to_video):
            raise FileNotFoundError(f"Video directory not found: {path_to_video}")

        frame_files = [f for f in os.listdir(path_to_video) if f.endswith('.png')]
        frame_files.sort(key=lambda x: int(x.split('.')[0]))
        num_frames = len(frame_files)
        
        if num_frames == 0:
            return np.zeros(0), 0.0, np.zeros(0)
            
        velocities = np.zeros(num_frames, dtype=np.float64)
        
        # RAFT resolution (must be divisible by 8)
        # Lower size = faster, Higher size = more accurate. 256x256 is a good balance.
        H, W = 256, 256 
        cx, cy = W / 2.0, H / 2.0

        # Pre-calculate radial vectors for the dense grid to calculate expansion direction
        x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
        rx = x_grid - cx
        ry = y_grid - cy
        
        # Mask to ignore the dead center (infinite depth) and extreme edges
        r_dist = np.sqrt(rx**2 + ry**2)
        valid_mask = (r_dist > (W * 0.15)) & (r_dist < (W * 0.45))

        # ZVU threshold (Tuned for dense flow)
        DENSE_NOISE_THRESHOLD = 0.15

        print(f"Processing Video {video_number} ({num_frames} frames)...")
        
        img1_batch = self.preprocess_image(os.path.join(path_to_video, frame_files[0]), (W, H))

        with torch.no_grad(): # Crucial for memory management
            for i in range(1, num_frames):
                img2_batch = self.preprocess_image(os.path.join(path_to_video, frame_files[i]), (W, H))
                
                # RAFT outputs a list of flow estimates, we take the final (most accurate) one
                list_of_flows = self.model(img1_batch, img2_batch)
                predicted_flow = list_of_flows[-1][0].cpu().numpy() # Shape: (2, H, W)
                
                dx = predicted_flow[0]
                dy = predicted_flow[1]

                # Dot product of flow vectors with radial vectors to find expansion/contraction
                dot_products = dx * rx + dy * ry
                
                # Calculate median magnitude and direction strictly within our valid annulus mask
                valid_dots = dot_products[valid_mask]
                valid_dx = dx[valid_mask]
                valid_dy = dy[valid_mask]
                
                direction_sign = np.sign(np.median(valid_dots))
                magnitudes = np.sqrt(valid_dx**2 + valid_dy**2)
                median_mag = np.median(magnitudes)
                
                if median_mag < DENSE_NOISE_THRESHOLD:
                    velocities[i] = 0.0
                else:
                    velocities[i] = direction_sign * median_mag
                
                img1_batch = img2_batch
                
                if i % 500 == 0:
                    print(f"  ...processed {i}/{num_frames} frames")

        # --- Pipeline Smoothing & Integration (State Machine Mode) ---
        
        # 1. Use raw integration ONLY to find the highly-accurate turning point
        kernel_size = 15
        if num_frames > kernel_size:
            padded_velocities = np.pad(velocities, (kernel_size//2, kernel_size//2), mode='edge')
            smoothed_vels = np.convolve(padded_velocities, np.ones(kernel_size)/kernel_size, mode='valid')
        else:
            smoothed_vels = velocities
            
        movement_path_raw = np.cumsum(smoothed_vels)
        tp_idx = int(np.argmax(movement_path_raw))
        turning_point = float(tp_idx)

        # 2. Strict State Classification
        # Increase the threshold to completely kill the "water flowing" false starts
        STATE_THRESHOLD = 0.25 
        states = np.zeros(num_frames, dtype=np.int8)
        states[smoothed_vels > STATE_THRESHOLD] = 1
        states[smoothed_vels < -STATE_THRESHOLD] = -1

        # 3. Constant-Velocity Integration (The Winch Heuristic)
        movement_path = np.zeros(num_frames, dtype=np.float64)
        
        # --- Outbound Trip ---
        outbound_states = states[:tp_idx+1]
        forward_frames = np.sum(outbound_states == 1)
        
        # Calculate the exact constant velocity per frame
        if forward_frames > 0:
            v_forward = channel_length / forward_frames
        else:
            v_forward = 0.0
            
        current_pos = 0.0
        for i in range(tp_idx+1):
            if outbound_states[i] == 1:
                current_pos += v_forward
            movement_path[i] = current_pos
            
        # Ensure the peak hits exactly channel_length
        movement_path[tp_idx] = channel_length
            
        # --- Inbound Trip ---
        inbound_states = states[tp_idx+1:]
        backward_frames = np.sum(inbound_states == -1)
        
        if backward_frames > 0:
            v_backward = channel_length / backward_frames
        else:
            v_backward = 0.0
            
        current_pos = channel_length
        for i in range(len(inbound_states)):
            if inbound_states[i] == -1:
                current_pos -= v_backward
            movement_path[tp_idx + 1 + i] = current_pos

        # Safety clip to physical bounds
        movement_path = np.clip(movement_path, 0, channel_length)
        
        # 4. Final Output Direction
        movement_direction = states # We can directly use our cleaned states

        # --- BONUS HUNTING: Drop Locations & Stall Zones ---
        self._calculate_bonus_events(video_number, velocities, turning_point)

        return movement_path, turning_point, movement_direction

    def _calculate_bonus_events(self, video_number, velocities, turning_point):
        """Calculates and prints bonus challenge data. You can redirect this to a file if required by the hackathon."""
        
        # 1. Drop Location (First sustained movement)
        drop_frame = 0
        for i in range(len(velocities)):
            if abs(velocities[i]) > 0.5: 
                if np.all(np.abs(velocities[i:i+15]) > 0.1): # 15 frames of sustained movement
                    drop_frame = i
                    break
                    
        # 2. Stall Zones (Pauses during the forward trip)
        stall_zones = []
        in_stall = False
        stall_start = 0

        for i in range(drop_frame, int(turning_point)):
            if velocities[i] == 0.0 and not in_stall:
                in_stall = True
                stall_start = i
            elif velocities[i] != 0.0 and in_stall:
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
