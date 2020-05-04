# -*- coding: utf-8 -*-
# @Time : 2020/3/20
# @File : voc_to_tfrecord.py
# @Software: PyCharm

import os,tqdm,sys
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import xml.etree.ElementTree as ET
rootPath = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(rootPath)

from components import config

flags.DEFINE_string('dataset_path', 'widerface', 'VOC format dataset')
flags.DEFINE_string('output_file', rootPath+'/dataset/train_mask.tfrecord', 'TFRecord file:output dataset')
flags.DEFINE_enum('split', 'train', ['train', 'val'], 'train or val dataset')



def process_image(image_file):
    # image_string = open(image_file,'rb').read()
    image_string = tf.io.read_file(image_file)
    try:
        image_data = tf.image.decode_jpeg(image_string, channels=3)
        return 0, image_string, image_data
    except tf.errors.InvalidArgumentError:
        logging.info('{}: Invalid JPEG data or crop window'.format(image_file))
        return 1, image_string, None


def parse_widerface(config_path):
    boxes_per_img = []
    with open(config_path) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            num_of_obj = int(fp.readline())
            boxes = []
            for i in range(num_of_obj):
                obj_box = fp.readline().split(' ')
                x0, y0, w, h = get_box(obj_box)
                if w == 0:
                    # remove boxes with no width
                    continue
                if h == 0:
                    # remove boxes with no height
                    continue
                # Because our network is outputting 7x7 grid then it's not worth processing images with more than
                # 5 faces because it's highly probable they are close to each other.
                # You could remove this filter if you decide to switch to larger grid (like 14x14)
                # Don't worry about number of train data because even with this filter we have around 16k samples
                # if num_of_obj > 5:
                #     continue
                boxes.append([x0, y0, w, h])
                # boxes.append((line.strip(), x0, y0, w, h))
            if num_of_obj == 0:
                obj_box = fp.readline().split(' ')
                x0, y0, w, h = get_box(obj_box)
                boxes.append([x0, y0, w, h])
            boxes_per_img.append((line.strip(), boxes))
            line = fp.readline()
            cnt += 1
    # print(boxes_per_img)
    return boxes_per_img


def xywh_to_voc(file_name, boxes, image_data):
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
    for box in boxes:
        classes.append(1)
        difficult.append(0)
        xmin.append(box[0])
        ymin.append(box[1])
        xmax.append(box[0] + box[2])
        ymax.append(box[1] + box[3])
    image_info['class'] = classes
    image_info['xmin'] = xmin
    image_info['ymin'] = ymin
    image_info['xmax'] = xmax
    image_info['ymax'] = ymax
    image_info['difficult'] = difficult
    # print(image_info)
    return image_info


def parse_annot(annot_file, CLASSES):
    """Parse Pascal VOC annotations."""
    tree = ET.parse(annot_file)
    root = tree.getroot()

    image_info = {}
    image_info_list = []

    file_name = root.find('filename').text

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text)

    xmin, ymin, xmax, ymax = [], [], [], []
    classes = []
    difficult = []

    for obj in root.iter('object'):
        label = obj.find('name').text

        if len(CLASSES) > 0 and label not in CLASSES:
            continue
        else:
            classes.append(CLASSES.index(label))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)

        for box in obj.findall('bndbox'):
            xmin.append(float(box.find('xmin').text))
            ymin.append(float(box.find('ymin').text))
            xmax.append(float(box.find('xmax').text))
            ymax.append(float(box.find('ymax').text))
            # xmin.append(float(box.find('xmin').text) / width)
            # ymin.append(float(box.find('ymin').text) / height)
            # xmax.append(float(box.find('xmax').text) / width)
            # ymax.append(float(box.find('ymax').text) / height)

    image_info['filename'] = file_name
    image_info['width'] = width
    image_info['height'] = height
    image_info['depth'] = depth
    image_info['class'] = classes
    image_info['xmin'] = xmin
    image_info['ymin'] = ymin
    image_info['xmax'] = xmax
    image_info['ymax'] = ymax
    image_info['difficult'] = difficult

    image_info_list.append(image_info)

    return image_info_list


# Input: [x0, y0, w, h, blur, expression, illumination, invalid, occlusion, pose]
# Output: x0, y0, w, h
def get_box(data):
    x0 = int(data[0])
    y0 = int(data[1])
    w = int(data[2])
    h = int(data[3])
    return x0, y0, w, h


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
    dataset_path = FLAGS.dataset_path

    if not os.path.isdir(dataset_path):
        logging.info('Please define valid dataset path.')
    else:
        logging.info('Loading {}'.format(dataset_path))

    logging.info('Reading configuration...')



    # class_list = config.cfg['labels_list']

    # logging.info("Class dictionary loaded: %s", class_list)

    if os.path.exists(FLAGS.output_file):
        logging.info('{:s} already exists. Exit...'.format(
            FLAGS.output_file))
        exit()

    with tf.io.TFRecordWriter(FLAGS.output_file) as writer:
        # img_list = open(
        #     os.path.join(FLAGS.dataset_path, 'ImageSets', 'Main', '%s.txt' % FLAGS.split)).read().splitlines()
        # logging.info("Image list loaded: %d", len(img_list))
        counter = 0
        skipped = 0
        split = 'wider_face_train_bbx_gt.txt' if FLAGS.split == 'train' else 'wider_face_val_bbx_gt.txt'
        file_path = 'WIDER_train' if FLAGS.split == 'train' else 'WIDER_val'
        for info in tqdm.tqdm(parse_widerface(os.path.join(FLAGS.dataset_path, 'wider_face_split', split))):
            image_file = os.path.join(FLAGS.dataset_path, file_path, 'images', info[0])
            # print(image_file)
            # print(os.path.exists(image_file))

            error, image_string, image_data = process_image(image_file)
            boxes = xywh_to_voc(image_file, info[1], image_data)
            # image_info_list = parse_annot(annot_file, class_list)
            if not error:
                tf_example = make_example(image_string, [boxes])

                writer.write(tf_example.SerializeToString())
                counter += 1

            else:
                skipped += 1
                logging.info('Skipped {:d} of {:d} images.'.format(skipped, len(img_list)))

        # for image in tqdm.tqdm(img_list):
        #     image_file = os.path.join(FLAGS.dataset_path, 'JPEGImages', '%s.jpg' % image)
        #     annot_file = os.path.join(FLAGS.dataset_path, 'Annotations', '%s.xml' % image)

        #     # processes the image and parse the annotation
        #     error, image_string, image_data = process_image(image_file)
        #     image_info_list = parse_annot(annot_file, class_list)
        #     if not error:
        #         tf_example = make_example(image_string, image_info_list)

        #         writer.write(tf_example.SerializeToString())
        #         counter += 1

        #     else:
        #         skipped += 1
        #         logging.info('Skipped {:d} of {:d} images.'.format(skipped, len(img_list)))

    logging.info('Wrote {} images to {}'.format(counter, FLAGS.output_file))

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    try:
        app.run(main)
    except SystemExit:
        pass
