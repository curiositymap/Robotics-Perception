#!/usr/bin/env python3

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Take a folder, write a TFRecord into it
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md

import math
import functools
import time
import random
import sys
import collections
import multiprocessing
import multiprocessing.pool
import io
import PIL
from PIL import Image
import numpy as np
import os
import argparse
import bbox_writer
import tensorflow as tf
import yaml
from object_detection.utils import dataset_util

description_text = """\
Use this script to convert labeled images into TFRecord files for training.

Both scripts in the labeling pipeline (labeler.py, tracking.py) store annotated
images as an image with a corresponding .txt file containing labels. While that
format is convenient for human readability, it is not the format TensorFlow is
expecting. This script converts the paired image format to the TFRecord format
that TensorFlow is expecting. Furthermore, this script can automatically
generate an evaluation split, or create one from a separate folder. Finally, the
label map required for the Object Detection API is generated and placed into the
specified folder.
"""

epilog_text = """\
example:
    ./convert_labels_to_records.py [folder]               convert without eval
    ./convert_labels_to_records.py [folder] --eval        generate an eval split
"""

def parse_tuple(s):
    try:
        out = tuple([int(val.strip()) for val in s.split(",")])
        if len(out) != 2:
            raise argparse.ArgumentTypeError("Must specify width and height")
        return out
    except Exception as e:
        raise argparse.ArgumentTypeError(e)


parser = argparse.ArgumentParser(
        description=description_text,
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--train_folder", type=str, required=True,
        help="Folder containing training data. All converted data output here.")
parser.add_argument("--output_folder", type=str, default="",
        help="Folder for output data. Same as train folder if not specified")
parser.add_argument("-n", "--number", type=int, default=10,
        help="Number of records")
parser.add_argument("-s", "--split", type=float, default=0.10,
        help="Percentage of training data to use for eval. Ignored if"
        " --eval_folder is specified.")
parser.add_argument("-p", "--positives_only", action="store_true",
        default=False,
        help="Only create records from positive examples, rather than all.")
parser.add_argument("--negatives_ratio", type=float, default=1.0,
        help="Ratio of negative examples to write (compared to positive "
        "examples). Result will be approximate.")
parser.add_argument("--resize", type=parse_tuple, default=None,
        help="Resize all images to a specified size w,h, e.g. 640,360")
parser.add_argument("--ignore_classes", type=str, default="",
        help="Comma separated class names to ignore")

eval_options = parser.add_mutually_exclusive_group()
eval_options.add_argument("-e", "--eval", action="store_true", default=False,
        help="Automatically generate eval split from train folder")
eval_options.add_argument("--eval_folder", type=str, default=None,
        help="Folder containing eval data")

args = parser.parse_args()
# Set output folder to train folder by default
args.output_folder = args.output_folder or args.train_folder

# NamedTuple to store results from example writing workers
Result = collections.namedtuple("RecordWritingResult",
        ["id", "record_count", "class_count", "negative_count"])


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def create_tf_example(labels, ignore_classes, txt_full_path):

    # Check to see whether the first line is the path or not.
    with open(txt_full_path, "r") as f:
        first = f.readline().strip()
        if first.startswith("#") and os.path.isfile(first[1:]): # On disk
            eprint("Found filename %s in the first line!" % first[1:])
            image_full_path = first[1:]
        else:
            txt_name = os.path.basename(txt_full_path)
            image_name = os.path.splitext(txt_name)[0] + ".png"
            image_full_path = os.path.join(os.path.dirname(txt_full_path),
                                           image_name)

    try:
        im = Image.open(image_full_path)
        # Store the original size because that's what the labels are based on
        original_height = im.height
        original_width = im.width
        if args.resize is not None:
            im = im.resize(args.resize, PIL.Image.BILINEAR)
    except FileNotFoundError as e:
        eprint("Unable to find image", image_full_path)
        return None

    arr = io.BytesIO()
    im.save(arr, format='PNG')

    height = im.height # (Saved, potentially resized) image height
    width = im.width # (Saved, potentially resized) image width
    channels = np.array(im).shape[2]
    filename = image_full_path # Filename of the image.
    encoded_image_data = arr.getvalue() # Encoded image bytes in png
    image_fmt = 'png' # Compression for the image bytes (encoded_image_data)

    rects, classes = bbox_writer.read_rects(txt_full_path)

    # Go through the classes and remove the ones we're told to ignore
    for i in range(len(classes) - 1, -1, -1):
        cls = classes[i]
        if cls in ignore_classes:
            rects.pop(i)
            classes.pop(i)

    # List of normalized coordinates, 1 per box, capped to [0, 1]
    xmins = [max(min(rect[0] / original_width, 1), 0) for rect in rects]
    xmaxs = [max(min(rect[2] / original_width, 1), 0) for rect in rects]
    ymins = [max(min(rect[1] / original_height, 1), 0) for rect in rects]
    ymaxs = [max(min(rect[3] / original_height, 1), 0) for rect in rects]

    classes_txt = [cls.encode('utf-8') for cls in classes] # String names
    class_ids = [labels[cls] for cls in classes]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/channels': dataset_util.int64_feature(channels),
        'image/colorspace': dataset_util.bytes_feature("RGB".encode('utf-8')),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/image_key': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_fmt.encode('utf-8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_txt),
        'image/object/class/label': dataset_util.int64_list_feature(class_ids),
    }))

    is_negative = len(rects) == 0

    return tf_example, collections.Counter(classes), is_negative


def write_record_from_list(id, labels, ignore_classes, data, out_path):

    class_count = collections.Counter()
    record_count = 0
    negative_count = 0
    with tf.python_io.TFRecordWriter(out_path) as writer:
        for filename in data:
            print("[%s] Writing record for" % id, filename)
            tf_example = create_tf_example(labels, ignore_classes, filename)
            if tf_example is None:
                continue

            example, count, is_negative = tf_example
            if is_negative and args.positives_only:
                continue

            # Write a specific ratio of negatives
            if (is_negative and record_count > 0 and 
                    (negative_count / record_count) > args.negatives_ratio):
                continue

            writer.write(example.SerializeToString())
            class_count += count
            record_count += 1
            negative_count += is_negative

    return Result(id, record_count, class_count, negative_count)


def get_shuffled_filenames(folder):
    # Grab the names of all of the pieces of train data
    train_txts = []
    for root, dirs, files in os.walk(folder, followlinks=True):
        for name in files:
            if not name.endswith(".txt"): continue
            if name.endswith("rects.txt"): continue # ignore init files

            txt_name = name
            txt_full_path = os.path.join(root, txt_name)
            train_txts.append(txt_full_path)

    train_txts.sort() # Allow there to be a random seed for consistency
    random.shuffle(train_txts) # So that we get a spread of eval data
    return train_txts


def get_labels_from_filenames(filenames, ignore_classes):
   
    core_count = multiprocessing.cpu_count()
    step = math.ceil(len(filenames) / core_count)
    splits = [filenames[i*step:(i+1)*step] for i in range(core_count)]

    def get_labels(split):
        labels = []
        for s in split:
            _, classes = bbox_writer.read_rects(s)
            labels.extend(classes)

        return set(labels)

    pool = multiprocessing.pool.ThreadPool()
    all_labels = pool.map(get_labels, splits)

    labels = set()
    for l in all_labels:
        labels.update(l)

    # Should be stable now
    labels = sorted(labels)
    all_labels = {label:i+1 for i,label in enumerate(labels)}

    for cls in ignore_classes:
        eprint("ignoring", repr(cls))
        labels.remove(cls)
    ignored_labels = {label:i+1 for i,label in enumerate(labels)}
    return all_labels, ignored_labels


def get_train_eval_split_filenames(filenames, split):

    eval_size = int(len(filenames) * split)
    eprint("Generating eval split of %d elements" % eval_size)

    eval_filenames = filenames[:eval_size]
    train_filenames = filenames[eval_size:]

    return train_filenames, eval_filenames


def write_labels(folder, labels, filename="label.pbtxt"):

    # Write out the label map file to disk
    label_name = os.path.join(folder, filename)
    label_txt = []
    for cls, i in labels.items():
        label_txt.append("item {\n  id: %d\n  name:'%s'\n}\n" % (i, cls))

    with open(label_name, "w") as f:
        f.write("\n".join(label_txt))

    eprint("Wrote %d labels: %s" % (len(labels), labels))


def write_convert_yaml(folder, all_labels, ignored_labels, 
        filename="remap.yaml"):
   
    conversions = []
    for cls in ignored_labels:
        from_id = ignored_labels[cls]
        to_id = all_labels[cls]
        conversions.append("%d:%d" % (from_id, to_id))

    convert_yaml_name = os.path.join(folder, filename)
    with open(convert_yaml_name, 'w') as f:
        f.write("\n".join(conversions))


def print_results(results):

    fmt = "[%s] has %d example images, %d positive images, " \
          "%d negative images, %s positive example distribution"

    class_counts = collections.Counter()
    record_counts = 0
    negative_counts = 0
    for result in results:
        eprint(fmt % (result.id, result.record_count, result.record_count -
            result.negative_count, result.negative_count, result.class_count))

        record_counts += result.record_count
        class_counts += result.class_count
        negative_counts += result.negative_count

    eprint(fmt % ("Overall", record_counts, record_counts - negative_counts,
        negative_counts, class_counts))


def get_record_writing_tasks():
    """Assign all record writing work into different lists.

    This function parses flags to determine training and eval splits. Eval data
    will either not be generated at all (neither --eval nor --eval_folder
    specified), generated automatically from train data (--eval), or sourced
    from a separate location (--eval_folder).
    """

    train_filenames = get_shuffled_filenames(args.train_folder)
    eprint("Got %d train filenames" % len(train_filenames))

    if args.eval_folder is not None:
        eprint("Getting filenames from eval folder")
        eval_filenames = get_shuffled_filenames(args.eval_folder)
    elif args.eval:
        eprint("Splitting train to get eval")
        train_filenames, eval_filenames = get_train_eval_split_filenames(
                train_filenames, args.split)
        eprint("Got %d new train filenames" % len(train_filenames))
    else: # No eval at all
        eprint("Not generating an eval set")
        eval_filenames = []

    eprint("Got eval filenames", len(eval_filenames))
   
    if args.ignore_classes:
        ignore_classes = set(map(str.strip, args.ignore_classes.split(",")))
    else:
        ignore_classes = set()
    all_labels, ignored_labels = get_labels_from_filenames(
            train_filenames + eval_filenames, ignore_classes)

    eprint("Generated full labels:", all_labels)
    write_labels(args.output_folder, all_labels)

    if ignore_classes:
        eprint("Generated ignored labels:", ignored_labels)
        write_labels(args.output_folder, ignored_labels, "label_ignored.pbtxt")
        write_convert_yaml(args.output_folder, all_labels, ignored_labels)

    tasks = []

    # Generate the training tasks by splitting up the train filenames
    train_record_number = min(max(args.number, 1), len(train_filenames))
    train_lists = [[] for i in range(train_record_number)]
    for i, filename in enumerate(train_filenames):
        train_lists[i % train_record_number].append(filename)

    for i, train_list in enumerate(train_lists):
        out_path = os.path.join(args.output_folder, "train-%05d.record" % i)
        tasks.append(("train-%05d" % i, ignored_labels, ignore_classes, 
            train_list, out_path))

    # Generate eval tasks by splitting up the eval filenames
    eval_record_number = min(max(int(args.number * args.split), 1), 
            len(eval_filenames))
    eval_lists = [[] for i in range(eval_record_number)]
    for i, filename in enumerate(eval_filenames):
        eval_lists[i % eval_record_number].append(filename)

    for i, eval_list in enumerate(eval_lists):
        out_path = os.path.join(args.output_folder, "eval-%05d.record" % i)
        tasks.append(("eval-%05d" % i, ignored_labels, ignore_classes, 
            eval_list, out_path))

    return tasks 


if __name__ == "__main__":
    t_start = time.time()

    random.seed(42) # Make sure the shuffle order is the same

    # Make the tasks for the workers to process
    tasks = get_record_writing_tasks()

    eprint("Starting %d record writing tasks" % len(tasks))

    # Actually have the workers generate the records
    pool = multiprocessing.pool.ThreadPool()
    results = pool.starmap(write_record_from_list, tasks)

    print_results(results)

    t_end = time.time()
    eprint("Took %5.2fs to write records" % (t_end - t_start))
