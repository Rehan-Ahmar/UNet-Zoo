import argparse
import os
import cv2
import numpy as np


def dilate_text(args):
    for file_name in os.listdir(args.input_dir):
        input_file_path = os.path.join(args.input_dir, file_name)
        output_file_path = os.path.join(args.output_dir, file_name)
        dilate_text_in_image(input_file_path, output_file_path)


def dilate_text_in_image(input_file_path, output_file_path):
    image = cv2.imread(input_file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
    _, thresh = cv2.threshold(
        gray, 150, 255, cv2.THRESH_BINARY_INV)  # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=5)  # dilate

    cv2.imwrite(output_file_path, dilated)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Detects text in images')
    parser.add_argument('--input_dir', help='Input directory', required=True)
    parser.add_argument(
        '--output_dir', help='Output data directory', required=False)
    FLAGS, unparsed = parser.parse_known_args()

    dilate_text(FLAGS)
