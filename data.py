import os, time, sys

import numpy as np
import cv2

import pandas as pd

class_names = ['forward', 'left', 'notleft', 'notright', 'right', 'stop']


def get_data():
    print('Loading data ...')
    image = []
    shape = []
    labels = []
    bboxes = []

    for i, class_name in enumerate(class_names, 1):
        df = pd.read_csv('data/{}/bounding_box.csv'.format(class_name), header=None)
        num_images = len(df)

        for image_id in range(num_images):
            img_dir = df[0][image_id]
            img = cv2.imread('data/{}'.format(img_dir[img_dir.find(class_name):]))
            x = df[1][image_id]
            y = df[2][image_id]
            w = df[3][image_id]
            h = df[4][image_id]

            image.append(img)
            shape.append(img.shape)
            labels.append([i])
            bboxes.append([(x,y,w,h)])

    print('Done')

    return image, shape, labels, bboxes

if __name__ == "__main__":
    
    image, shape, labels, bboxes = get_data()

    for i, img in enumerate(image):
        x, y, w, h = bboxes[i][0]
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0))
        cv2.imshow('Image', img)
        cv2.waitKey(1)