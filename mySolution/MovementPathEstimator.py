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

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Loading RAFT Optical Flow Model on {self.device}...")
        
        weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=weights, progress=False).to(self.device)
        self.model.eval()
        self.transforms = weights.transforms()

    def preprocess_image(self, img_path: str, target_size=(256, 256)) -> torch.Tensor:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        
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
        
        H, W = 256, 256 
        cx, cy = W / 2.0, H / 2.0

        x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
        rx = x_grid - cx
        ry = y_grid - cy
        
        r_dist = np.sqrt(rx**2 + ry**2)
        valid_mask = (r_dist > (W * 0.15)) & (r_dist < (W * 0.45))

        DENSE_NOISE_THRESHOLD = 0.15

        print(f"Processing Video {video_number} ({num_frames} frames)...")
        
        img1_batch = self.preprocess_image(os.path.join(path_to_video, frame_files[0]), (W, H))

        with torch.no_grad():
            for i in range(1, num_frames):
                img2_batch = self.preprocess_image(os.path.join(path_to_video, frame_files[i]), (W, H))
                
                list_of_flows = self.model(img1_batch, img2_batch)
                predicted_flow = list_of_flows[-1][0].cpu().numpy() # Shape: (2, H, W)
                
                dx = predicted_flow[0]
                dy = predicted_flow[1]

                dot_products = dx * rx + dy * ry
                
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

        kernel_size = 15
        if num_frames > kernel_size:
            padded_velocities = np.pad(velocities, (kernel_size//2, kernel_size//2), mode='edge')
            smoothed_vels = np.convolve(padded_velocities, np.ones(kernel_size)/kernel_size, mode='valid')
        else:
            smoothed_vels = velocities

        STATE_THRESHOLD = 0.15 
        states = np.zeros(num_frames, dtype=np.int8)
        states[smoothed_vels > STATE_THRESHOLD] = 1
        states[smoothed_vels < -STATE_THRESHOLD] = -1
        
        forward_states = (states == 1).astype(int)
        backward_states = (states == -1).astype(int)
        
        cum_forward = np.cumsum(forward_states)
        total_backward = np.sum(backward_states)
        cum_backward = np.cumsum(backward_states)
        backward_after = total_backward - cum_backward
        
        split_score = cum_forward + backward_after
        tp_idx = int(np.argmax(split_score))
        turning_point = float(tp_idx)

        movement_path = np.zeros(num_frames, dtype=np.float64)

        outbound_states = states[:tp_idx+1]
        forward_frames = np.sum(outbound_states == 1)

        if forward_frames > 0:
            v_forward = channel_length / forward_frames
        else:
            v_forward = 0.0

        current_pos = 0.0
        for i in range(tp_idx+1):
            if outbound_states[i] == 1:
                current_pos += v_forward
            movement_path[i] = current_pos

        movement_path[tp_idx] = channel_length

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

        movement_path = np.clip(movement_path, 0, channel_length)
        movement_direction = states

        self._calculate_bonus_events(video_number, states, turning_point)

        return movement_path, turning_point, movement_direction

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