import numpy as np

class MovementPath:

    def __init__(self, video_num, movement_path, movement_direction = np.zeros(1), turning_point = 0):
        self.video_num = int(video_num)
        self.movement_path = np.array(movement_path)
        self.movement_direction = np.array(movement_direction)
        self.threshold = 0.03
        if turning_point == 0:
            self.turning_point = self.calculate_turning_point()
        else:
            self.turning_point = turning_point
        if movement_direction.shape[0] == 1:
            self.movement_direction = self.calculate_movement_direction()
        else:
            self.movement_direction = movement_direction


    def calculate_turning_point(self):
        max_value = np.max(self.movement_path)  # Find the maximum value
        indices = np.argwhere(self.movement_path == max_value)  # Find indices where the maximum value occurs
        first_index = indices[0][0]  # Get the first index
        last_index = indices[-1][0]  # Get the last index
        return first_index + (last_index-first_index)/2

    def calculate_movement_direction(self):
        movement_direction = self.movement_path
        print(movement_direction.shape[0])
        movement_direction = movement_direction[1:] - movement_direction[:-1]
        movement_direction[:180] = 0
        movement_direction[-100:] = 0
        threshold = np.max(movement_direction)/6
        movement_direction[(movement_direction >= -threshold) & (movement_direction <= threshold)] = 0
        movement_direction = np.sign(movement_direction)
        movement_direction = np.append(movement_direction, [-1])
        return movement_direction
