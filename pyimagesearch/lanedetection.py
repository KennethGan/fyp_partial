#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import warnings

class lanedetection:
    
    def make_points(self, image, line, k):
        if np.isnan(line).any() or not np.isfinite(line).any():

        # return [[0, 0, 0, 0]]

            return None

        (slope, intercept) = line
        y1 = int(image.shape[0])  # bottom of the image
        y2 = int(y1 * k / 20)  # slightly lower than the middle

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [[x1, y1, x2, y2]]

    def average_slope_intercept(self, image, lines, k):
        left_fit = []
        right_fit = []
        if lines is None:
            return None
        for line in lines:
            for (x1, y1, x2, y2) in line:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:                 
                        fit = np.polyfit((x1, x2), (y1, y2), 1)
                    except ValueError:
                        return None
                    except np.RankWarning:
                        return None
                    slope = fit[0]
                    intercept = fit[1]
                    if slope < 0 and slope > -5:  # y is reversed in image
                        left_fit.append((slope, intercept))
                    elif slope > 0 and slope < 5:
                        right_fit.append((slope, intercept))
                    else:
                        return None

        # add more weight to longer lines

        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = self.make_points(image, left_fit_average, k)
        right_line = self.make_points(image, right_fit_average, k)

        if left_line == None or right_line == None:
            return None

        if left_line[0][0] > 2000 or left_line[0][0] < -1 \
            or left_line[0][1] > 2000 or left_line[0][1] < -1 \
            or left_line[0][2] > 2000 or left_line[0][2] < -1 \
            or left_line[0][3] > 2000 or left_line[0][3] < -1 \
            or right_line[0][0] > 2000 or right_line[0][0] < -1 \
            or right_line[0][1] > 2000 or right_line[0][1] < -1 \
            or right_line[0][2] > 2000 or right_line[0][2] < -1 \
            or right_line[0][3] > 2000 or right_line[0][3] < -1:
            return None

        averaged_lines = [left_line, right_line]
        return averaged_lines

    def canny(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel = 5
        blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
	#The edge pixels above the upper limit are considered in an edge map and edge pixels below the threshold are discarded.So what about the pixels inbetween upper and lower threshold? They are considered only if they are connected to pixels in upper threshold. 
        canny = cv2.Canny(gray, 250, 500)
        return canny

    def display_lines(self, img, lines):
        line_image = np.zeros_like(img)
        if lines is not None:
            cv2.line(line_image, (lines[0][0][2], lines[0][0][3]),
                     (lines[1][0][2], lines[1][0][3]), (255, 0, 0), 10)
            for line in lines:
                for (x1, y1, x2, y2) in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0,
                             0), 10)
        return line_image

    def region_of_interest(self, canny):
        height = canny.shape[0]
        width = canny.shape[1]
        mask = np.zeros_like(canny)

        #triangle = np.array([[(0, height), ( width // 2, height // 2),
                            #(width, height)]], np.int32)

        triangle = np.array([[(0, height), ( 0, height // 2), (width, height // 2),
                            (width, height)]], np.int32)


        cv2.fillPoly(mask, triangle, 255)
        masked_image = cv2.bitwise_and(canny, mask)
        return masked_image



			
