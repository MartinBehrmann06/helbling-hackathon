import math
import numpy as np
import matplotlib.pyplot as plt

from MovementPath import MovementPath

class EstimationRater:

    def __init__(self, estimated_movement_paths, measured_movement_paths, do_print, do_plot, estimator_name, video_num):
        self.estimated_movement_paths = estimated_movement_paths
        self.measured_movement_paths = measured_movement_paths
        self.do_print = do_print
        self.do_plot = do_plot
        self.estimator_name = estimator_name
        self.video_num = video_num
        self.num_videos = 11

        self.channel_length = np.load("channel_lengths.npy")

        # to be calculated
        self.turning_point_scores = []
        self.movement_path_scores = []

    def rate(self):
        if self.video_num > 0 and 0 in self.measured_movement_paths:
            self.turning_point_scores.append(self.calc_turning_point_score(0))
            self.movement_path_scores.append(self.calc_mean_absolute_diff_movement_path(0))
            if self.do_print:
                print('Estimator: ' + self.estimator_name + ', Video: ' + str(self.video_num) + ', Measured turning point: ',
                      self.measured_movement_paths[0].turning_point, ', estimated turning point: ',
                      self.estimated_movement_paths[0].turning_point, ", absolute error [Frames]: ", self.turning_point_scores[0])
                print('Estimator: ' + self.estimator_name + ', Video: ' + str(
                    self.video_num) + " Mean absolute error of the movement path [m]: ", self.movement_path_scores[0])
            if self.do_plot: # plot movement path
                plt.plot(self.estimated_movement_paths[0].movement_path, label='estimated')
                plt.plot(self.measured_movement_paths[0].movement_path, label='measured')
                plt.xlabel('Frames')
                plt.ylabel('Distance')
                plt.title('Estimator: ' + str(self.estimator_name) + ', Video: ' +  str(self.video_num) )
                plt.legend()
                plt.show()
            # if self.do_plot:  # plot directions
            #     plt.plot(self.estimated_movement_paths[0].movement_direction, label='estimated')
            #     plt.plot(self.measured_movement_paths[0].movement_direction, label='measured')
            #     plt.plot(self.measured_movement_paths[0].movement_path/100, label='movement path')
            #     plt.xlabel('Frames')
            #     plt.ylabel('Direction')
            #     plt.title('Estimator: ' + str(self.estimator_name) + ', Video: ' + str(self.video_num))
            #     plt.legend()
            #     plt.show()
        else:
            for index in range(0, self.num_videos):
                if index + 1 in self.estimated_movement_paths and index + 1 in self.measured_movement_paths:
                    self.turning_point_scores.append(self.calc_turning_point_score(index + 1))
                    self.movement_path_scores.append(self.calc_mean_absolute_diff_movement_path(index + 1))
            if self.do_print:
                self.print_scores()
            if self.do_plot:
                self.plot_movement_paths()
            if self.do_plot:
                self.plot_error_movement_paths()
            # if self.do_plot:
            #     self.plot_directions()

    def calc_turning_point_score(self, ind):
        print("ind: ", ind, self.estimated_movement_paths[ind].turning_point, self.measured_movement_paths[ind].turning_point)
        return abs(self.estimated_movement_paths[ind].turning_point - self.measured_movement_paths[ind].turning_point)
    def calc_mean_absolute_diff_movement_path(self, ind):
        return self.mean_absolute_difference(self.estimated_movement_paths[ind].movement_path,
                                    self.measured_movement_paths[ind].movement_path)


    def print_scores(self):
        score_index = 0
        for index in range(0,self.num_videos):
            if index+1 in self.estimated_movement_paths and index+1 in self.measured_movement_paths:
                estimated_movement_path = self.estimated_movement_paths[index+1]
                measured_movement_path = self.measured_movement_paths[index+1]
                movement_path_score = self.movement_path_scores[score_index]
                turning_point_score = self.turning_point_scores[score_index]

                score_index += 1
                video_num = self.video_num
                if len(self.turning_point_scores) > 1:
                    video_num = index + 1
                print('Estimator: ' + self.estimator_name + ', Video: ' + str(video_num) + ', Measured turning point: ',
                      measured_movement_path.turning_point, ', estimated turning point: ',
                      estimated_movement_path.turning_point, ", absolute error [Frames]: ", turning_point_score)
                print('Estimator: ' + self.estimator_name + ', Video: ' + str(
                    video_num) + " Mean absolute error of the movement path [m]: ", movement_path_score)
        #avg_turning_point_score = sum(self.turning_point_scores) / len(self.turning_point_scores)
        avg_turning_point_score = np.average(np.array(self.turning_point_scores))
        avg_movement_path_score = np.average(np.array(self.movement_path_scores))
        print('\n\n::::: Estimator: ', self.estimator_name, ", Absolute turning point error (average over all videos) [Frames]: ",
              avg_turning_point_score, ", Mean absolute error of the movement path (average over all videos) [m]: ", avg_movement_path_score,
     )

    def plot_movement_paths_old(self):
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'brown', 'pink']

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(11,9))

        for index in range(0,self.num_videos):
            if index+1 in self.estimated_movement_paths:
                estimated_movement_path = self.estimated_movement_paths[index + 1]
                measured_movement_path = self.measured_movement_paths[index+1]
                ax.plot(measured_movement_path.movement_path, color=colors[index], linestyle='dashed', label=f'Measurement {index + 1}')
                ax.plot(estimated_movement_path.movement_path, color=colors[index], linestyle='solid',
                        label=f'Estimation {index + 1}')

        # Add labels and legend
        ax.set_xlabel("Frame")
        ax.set_ylabel("Distance")
        ax.set_title("Estimator: " + str(self.estimator_name))
        ax.legend()
        plt.show()

    def plot_movement_paths(self):
        # Create a figure and a grid of 3x4 subplots
        fig, axes = plt.subplots(3, 4, figsize=(12, 8))

        # Generate and plot random numbers in each subplot
        baseline = self.get_baseline()
        for ind, ax in enumerate(axes.flatten()):
            if ind != 11:
                if ind + 1 in self.estimated_movement_paths:
                    estimated_movement_path = self.estimated_movement_paths[ind + 1]
                    ax.plot(estimated_movement_path.movement_path, label=f'Estimation {ind + 1}')
                    if ind + 1 in self.measured_movement_paths:
                        measured_movement_path = self.measured_movement_paths[ind + 1]
                        ax.plot(measured_movement_path.movement_path, label=f'Measurement {ind + 1}')
                    #ax.legend()
                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Distanz [m]')
            if ind == 11:
                ax.plot(np.array([0]), label=f'Schätzung')
                ax.plot(np.array([0]), label=f'Messung')
                ax.set_xticks([])  # Hide x-axis ticks and labels
                ax.set_yticks([])  # Hide y-axis ticks and labels
                ax.legend()

    def get_baseline(self):
        baseline = []

        # video 1
        baseline.append([[0], [0]])
        # video 2
        baseline.append([[0], [0]])
        # video 3
        baseline.append([[0], [0]])
        # video 4
        start_frame = 320-30
        start_dist = self.channel_length[3] #50
        end_frame = 850
        y = np.linspace(start_dist, 0, end_frame - start_frame)
        x = np.linspace(start_frame, end_frame, end_frame - start_frame)
        baseline.append([x, y])
        # video 5
        start_frame = 320-55
        start_dist = self.channel_length[4] #40
        end_frame = 650
        y = np.linspace(start_dist, 0, end_frame - start_frame)
        x = np.linspace(start_frame, end_frame, end_frame - start_frame)
        baseline.append([x, y])
        # video 6
        start_frame = 270-58
        start_dist = self.channel_length[5] #60
        end_frame = 3615
        y = np.linspace(start_dist, 0, end_frame - start_frame)
        x = np.linspace(start_frame, end_frame, end_frame - start_frame)
        baseline.append([x, y])
        # video 7
        start_frame = 510-330
        start_dist = self.channel_length[6] #60
        end_frame = 2615
        y = np.linspace(start_dist, 0, end_frame - start_frame)
        x = np.linspace(start_frame, end_frame, end_frame - start_frame)
        baseline.append([x, y])
        # video 8
        start_frame = 835-100
        start_dist = self.channel_length[7] #60
        end_frame = 1480
        y = np.linspace(start_dist, 0, end_frame - start_frame)
        x = np.linspace(start_frame, end_frame, end_frame - start_frame)
        baseline.append([x, y])
        # video 9
        start_frame = 195-35
        start_dist = self.channel_length[8] #60
        end_frame = 1285
        y = np.linspace(start_dist, 0, end_frame - start_frame)
        x = np.linspace(start_frame, end_frame, end_frame - start_frame)
        baseline.append([x, y])
        # video 10
        start_frame = 225-42
        start_dist = self.channel_length[9] #80
        end_frame = 1065
        y = np.linspace(start_dist, 0, end_frame - start_frame)
        x = np.linspace(start_frame, end_frame, end_frame - start_frame)
        baseline.append([x, y])
        # video 11
        start_frame = 535-314
        start_dist = self.channel_length[10] #40
        end_frame = 1280
        y = np.linspace(start_dist, 0, end_frame - start_frame)
        x = np.linspace(start_frame, end_frame, end_frame - start_frame)
        baseline.append([x, y])

        return baseline

    def plot_error_movement_paths(self):
        # ---- Plot absolut error per video
        # Create a figure and a grid of 3x4 subplots
        fig, axes = plt.subplots(3, 4, figsize=(12, 8))

        # Generate and plot random numbers in each subplot
        for ind, ax in enumerate(axes.flatten()):
            if ind != 11:
                if ind + 1 in self.estimated_movement_paths and ind + 1 in self.measured_movement_paths:
                    estimated_movement_path = self.estimated_movement_paths[ind + 1].movement_path
                    measured_movement_path = self.measured_movement_paths[ind + 1].movement_path
                    max_ind = min(estimated_movement_path.shape[0],measured_movement_path.shape[0])
                    ax.axhline(y=0, color=(0, 0, 0, 1), linestyle='-', linewidth=0.5)
                    ax.plot(measured_movement_path[:max_ind] - estimated_movement_path[:max_ind], label=f'Measurement {ind + 1}')
                    turning_point = self.measured_movement_paths[ind + 1].turning_point
                    ax.axvline(x=turning_point, color='r', linestyle='-', label='Turning Point')
                    #ax.legend()
                    ax.set_ylim(-10,14)
                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Fehler [m]')
            if ind == 11:
                ax.plot(np.array([0]), label=f'Messung - Schätzung')
                ax.plot(np.array([0]), color='r', linestyle='-', label=f'Turning Point')
                ax.set_xticks([])  # Hide x-axis ticks and labels
                ax.set_yticks([])  # Hide y-axis ticks and labels
                ax.legend()
        plt.show()

        # # ---- Plot relative error histogram per video
        # # Create a figure and a grid of 3x4 subplots
        # fig, axes = plt.subplots(3, 4, figsize=(12, 8))

        # # Generate and plot random numbers in each subplot
        # for ind, ax in enumerate(axes.flatten()):
        #     if ind != 11:
        #         if ind + 1 in self.estimated_movement_paths and ind + 1 in self.measured_movement_paths:
        #             estimated_movement_path = self.estimated_movement_paths[ind + 1].movement_path
        #             measured_movement_path = self.measured_movement_paths[ind + 1].movement_path
        #             max_ind = min(estimated_movement_path.shape[0], measured_movement_path.shape[0])
        #             rel_errors = abs(measured_movement_path[:max_ind] - estimated_movement_path[:max_ind]) / self.channel_length[ind]
        #             counts, bins = np.histogram(rel_errors*100, bins=20)
        #             ax.hist(bins[:-1], bins, weights=counts, density=True)
        #             # ax.legend()
        #             ax.set_xlabel('Relativer Fehler [%]')
        #             ax.set_ylabel('Wahrscheinlichkeitsdichte')
        #     if ind == 11:
        #         ax.plot(np.array([0]), label=f'rel abs error')
        #         ax.set_xticks([])  # Hide x-axis ticks and labels
        #         ax.set_yticks([])  # Hide y-axis ticks and labels
        #         #ax.legend()

        # # ---- Plot relative and absolut error histogram for all videos
        # all_abs_errors = np.array([])
        # all_rel_errors = np.array([])
        # for i in range(11):
        #     if i + 1 not in self.estimated_movement_paths or i + 1 not in self.measured_movement_paths:
        #         continue
        #     estimated_movement_path = self.estimated_movement_paths[i + 1].movement_path
        #     measured_movement_path = self.measured_movement_paths[i + 1].movement_path
        #     max_ind = min(estimated_movement_path.shape[0], measured_movement_path.shape[0])
        #     abs_errors = abs(measured_movement_path[:max_ind] - estimated_movement_path[:max_ind])
        #     rel_errors = abs(measured_movement_path[:max_ind] - estimated_movement_path[:max_ind]) / \
        #                  self.channel_length[i]
        #     all_abs_errors = np.append(all_abs_errors, abs_errors)
        #     all_rel_errors = np.append(all_rel_errors, rel_errors)

        # plt.figure()
        # counts, bins = np.histogram(all_abs_errors, bins=100)
        # plt.hist(bins[:-1], bins, weights=counts, density=True)
        # plt.xlabel('Absoluter Fehler [m]')
        # plt.ylabel('Wahrscheinlichkeitsdichte')

        # plt.figure()
        # counts, bins = np.histogram(all_abs_errors, bins=100)
        # counts = counts / (sum(counts))
        # plt.hist(bins[:-1], bins, weights=np.cumsum(counts), density=False)
        # plt.xlabel('Absoluter Fehler [m]')
        # plt.ylabel('Kumulative Wahrscheinlichkeitsdichte')

        # plt.figure()
        # counts, bins = np.histogram(all_rel_errors*100, bins=100)
        # plt.hist(bins[:-1], bins, weights=counts, density=True)
        # plt.xlabel('Relativer Fehler [%]')
        # plt.ylabel('Wahrscheinlichkeitsdichte')

        # plt.figure()
        # counts, bins = np.histogram(all_rel_errors * 100, bins=100)
        # counts = counts / (sum(counts))
        # plt.hist(bins[:-1], bins, weights=np.cumsum(counts), density=False)
        # plt.axhline(y=0.5, color='black', linewidth=0.5)
        # plt.axhline(y=0.8, color='black', linewidth=0.5)
        # plt.axhline(y=0.9, color='black', linewidth=0.5)
        # plt.axhline(y=0.95, color='black', linewidth=0.5)
        # plt.axhline(y=0.99, color='black', linewidth=0.5)
        # #yticks = plt.yticks()[0].tolist()
        # yticks = []
        # yticks.append(0.5)
        # yticks.append(0.8)
        # yticks.append(0.9)
        # yticks.append(0.95)
        # yticks.append(0.99)
        # plt.axvline(x=2.6, color='black', linewidth=0.5)
        # plt.axvline(x=6.3, color='black', linewidth=0.5)
        # plt.axvline(x=8.7, color='black', linewidth=0.5)
        # plt.axvline(x=11, color='black', linewidth=0.5)
        # plt.axvline(x=13.3, color='black', linewidth=0.5)
        # #xticks = plt.xticks()[0].tolist()
        # xticks = []
        # xticks.append(2.6)
        # xticks.append(6.3)
        # xticks.append(8.7)
        # xticks.append(11)
        # xticks.append(13.3)
        # plt.xticks(xticks)
        # plt.yticks(yticks)
        # plt.ylim(0, 1.03)
        # plt.xlim(0, 16.2)
        # plt.xlabel('Relativer Fehler [%]')
        # plt.ylabel('Kumulative Wahrscheinlichkeitsdichte')



    def plot_directions(self):
        # Create a figure and a grid of 3x4 subplots
        fig, axes = plt.subplots(3, 4, figsize=(12, 8))

        # Generate and plot random numbers in each subplot
        for ind, ax in enumerate(axes.flatten()):
            if ind != 11:
                if ind + 1 in self.estimated_movement_paths:
                    estimated_movement_path = self.estimated_movement_paths[ind + 1]
                    ax.plot(estimated_movement_path.movement_direction, label=f'Estimation {ind + 1}')
                    if ind + 1 in self.measured_movement_paths:
                        measured_movement_path = self.measured_movement_paths[ind + 1]
                        ax.plot(measured_movement_path.movement_direction, label=f'Measurement {ind + 1}')
                    #ax.legend()
                    ax.set_xlabel('Frame')
                    ax.set_ylabel('Richtung')
            if ind == 11:
                ax.plot(np.array([0]), label=f'Schätzung')
                ax.plot(np.array([0]), label=f'Messung')
                ax.set_xticks([])  # Hide x-axis ticks and labels
                ax.set_yticks([])  # Hide y-axis ticks and labels
                ax.legend()

        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the plots
        plt.show()

    def mean_absolute_difference(self, estimated, measured):

        # cut estimated if longer than measured
        if len(estimated) > len(measured):
            estimated = estimated[:len(measured)]

        # add zeros if measured longer
        if len(estimated) < len(measured):
            length_diff = len(measured) - len(estimated)
            estimated = np.pad(estimated, (0, length_diff), 'constant')

        # Compute the absolute difference of the common elements
        abs_diff = np.abs(estimated - measured)

        # Calculate the mean  difference
        msd = np.mean(abs_diff)

        return msd



