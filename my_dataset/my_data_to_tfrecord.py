# -*- coding: utf-8 -*-
# @Time : 2020/3/20
# @File : voc_to_tfrecord.py
# @Software: PyCharm

import os,tqdm,sys
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from glob import glob
import xml.etree.ElementTree as ET
rootPath = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(rootPath)

from components import config


def process_image(image_file):
    # image_string = open(image_file,'rb').read()
    image_string = tf.io.read_file(image_file)
    try:
        image_data = tf.image.decode_jpeg(image_string, channels=3)
        return 0, image_string, image_data
    except tf.errors.InvalidArgumentError:
        logging.info('{}: Invalid JPEG data or crop window'.format(image_file))
        return 1, image_string, None


def xywh_to_voc(file_name, image_data):
    # read xml
    xml = ET.parse(file_name)
    root = xml.getroot()
    root = root.findall('object')

    shape = image_data.shape
    image_info = {}
    image_info['filename'] = file_name
    image_info['width'] = shape[1]
    image_info['height'] = shape[0]
    image_info['depth'] = 3

    difficult = []
    classes = []
    xmin, ymin, xmax, ymax = [], [], [], []
    # print(shape)
    for box in root:
        box = box.find('bndbox')
        classes.append(1)
        difficult.append(0)
        xmin.append(int(box.find('xmin').text))
        ymin.append(int(box.find('ymin').text))
        xmax.append(int(box.find('xmax').text))
        ymax.append(int(box.find('ymax').text))
    image_info['class'] = classes
    image_info['xmin'] = xmin
    image_info['ymin'] = ymin
    image_info['xmax'] = xmax
    image_info['ymax'] = ymax
    image_info['difficult'] = difficult
    # print(image_info)
    return image_info


def make_example(image_string, image_info_list):

    for info in image_info_list:
        filename = info['filename']
        width = info['width']
        height = info['height']
        depth = info['depth']
        classes = info['class']
        xmin = info['xmin']
        ymin = info['ymin']
        xmax = info['xmax']
        ymax = info['ymax']
        # difficult = info['difficult']

    if isinstance(image_string, type(tf.constant(0))):
        encoded_image = [image_string.numpy()]
    else:
        encoded_image = [image_string]

    base_name = [tf.compat.as_bytes(os.path.basename(filename))]

    example = tf.train.Example(features=tf.train.Features(feature={
        'filename':tf.train.Feature(bytes_list=tf.train.BytesList(value=base_name)),
        'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'classes':tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'x_mins':tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'y_mins':tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'x_maxes':tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'y_maxes':tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_image))
    }))
    return example


def main(argv):
    output_file = 'dataset/my_dataset.tfrecord'

    with tf.io.TFRecordWriter(output_file) as writer:
        annotations = glob('my_dataset/*.xml')
        for anno in tqdm.tqdm(annotations):
            image_file = anno.replace('xml', 'jpg')
            assert os.path.exists(image_file), f'{image_file} file doesn\'t exist'
            # image_file = os.path.join(dataset_path, file_path, 'images', info[0])

            error, image_string, image_data = process_image(image_file)
            boxes = xywh_to_voc(anno, image_data)

            if not error:
                tf_example = make_example(image_string, [boxes])

                writer.write(tf_example.SerializeToString())
                # counter += 1

            # else:
            #     # skipped += 1
            #     logging.info('Skipped {:d} of {:d} images.'.format(skipped, len(img_list)))

    # logging.info('Wrote {} images to {}'.format(counter, output_file))

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    try:
        app.run(main)
    except SystemExit:
        pass
