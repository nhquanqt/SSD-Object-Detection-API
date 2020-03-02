import sys, time, os
import io
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import data

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


class_names = ['none', 'forward', 'left', 'notleft', 'notright', 'right', 'stop']

image, shape, labels, bboxes = data.get_data()

print(len(image))

writer = tf.python_io.TFRecordWriter('training/train.record')

cur_cls = 0
id = 0

path = os.getcwd()

for i in range(len(image)):
    class_id = labels[i][0]
    if class_id != cur_cls:
        id = 0
    filename = 'data/{}/frame_{}.jpg'.format(class_names[class_id], id)

    with tf.gfile.GFile(os.path.join(path, '{}'.format(filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = filename.encode('utf8')
    image_format = b'jpg'
    
    x = bboxes[i][0][0]
    y = bboxes[i][0][1]
    w = bboxes[i][0][2]
    h = bboxes[i][0][3]

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    xmins.append(x / width)
    xmaxs.append((x+w) / width)
    ymins.append(y / height)
    ymaxs.append((y+h) / height)
    classes_text.append(class_names[class_id].encode('utf8'))
    classes.append(class_id)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    
    writer.write(tf_example.SerializeToString())

writer.close()
output_path = os.path.join(os.getcwd(), 'training/train.record')
print('Successfully created the TFRecords: {}'.format(output_path))