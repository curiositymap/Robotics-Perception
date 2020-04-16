#!/usr/bin/env python3

import bbox_writer
import os
import argparse
import pathlib
import yaml

description_text = """\
Use this script to convert short form (single character) labels to long ones.

When using the labeler.py or tracking.py scripts, single character class names
are used for efficiency. This makes the label map inconvenient for future
consumption (mostly by humans). This script takes the provided class map and
converts short class names into long ones. Additionally, invalid class names
will be identified (e.g. None), and all labels will be converted to int (instead
of float).

The supplied YAML file should have one class mapping per file, as follows:

c: cat
d: dog
p: person

See class_map.yaml for an example of the file format.
"""

parser = argparse.ArgumentParser(
        description=description_text,
        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--class_map", type=pathlib.Path, required=True,
        help="YAML file defining class mapping")
parser.add_argument("--folder", type=pathlib.Path, required=True,
        help="Folder to walk and convert labels in")
args = parser.parse_args()


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def main():
    if not args.class_map.exists() or not args.class_map.is_file():
        eprint("File %s is invalid!" % args.class_map)

    with open(args.class_map, 'r') as f:
        # Note that we're not using the CLoader since it may not exist on all
        # installations, and isn't really necessary for the small file sizes we
        # have here.
        class_map = yaml.load(f.read(), yaml.Loader)
        print("Loaded class mapping:", class_map)

    short_classes = set(class_map.keys())
    long_classes = set(class_map.values())

    if not args.folder.exists() or not args.folder.is_dir():
        eprint("Specified folder %s is invalid!" % args.folder)

    for root, dirs, files in os.walk(args.folder, followlinks=True):
        for name in files:
            if not name.endswith(".txt"): continue
            if name.endswith("rects.txt"): continue

            txt_name = name
            txt_full_path = os.path.join(root, txt_name)

            bboxes, classes = bbox_writer.read_bboxes(txt_full_path)
            for i, c in enumerate(classes):
                if c in short_classes:
                    classes[i] = class_map[c]

                if not classes[i] in long_classes:
                    print("Class name %s in file %s on line %d is invalid!" % (
                            classes[i], txt_full_path, i + 1))

            bbox_writer.write_bboxes(bboxes, classes, txt_full_path)

if __name__ == "__main__":
    main()
