import numpy as np
import cv2

class Visualization:
    def draw_rectangles(self, image, rectangles, color, thickness=1, font_scale=1):
        for rect in rectangles:
            x, y, w, h = map(int, rect)
            cv2.rectangle(image, (x, y), (w, h), color, thickness, font_scale)
