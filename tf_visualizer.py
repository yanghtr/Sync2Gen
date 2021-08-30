#!/usr/bin/env python
# coding=utf-8

import os
import time
import sys
import tf_logger


class Visualizer():
    def __init__(self, log_dir, name='train'):
        self.logger = tf_logger.Logger(os.path.join(log_dir, name))
        self.log_name = os.path.join(log_dir, 'tf_visualizer_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to save
    def log_images(self, visuals, step):
            for label, image_numpy in visuals.items():
                self.logger.image_summary(
                    label, [image_numpy], step)

    # scalars: dictionary of scalar labels and values
    def log_scalars(self, scalars, step):
        for label, val in scalars.items():
            self.logger.scalar_summary(label, val, step)

    # scatter plots
    def plot_current_points(self, points, disp_offset=10):
        pass

    # scalars: same format as |scalars| of plot_current_scalars
    def print_current_scalars(self, epoch, i, scalars):
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in scalars.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
