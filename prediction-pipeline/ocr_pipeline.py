import os
import json
import glob
import concurrent.futures
import time
import cv2
import pytesseract

def ocr(img):
    text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
    return text

def main(root_dir):
    with open(os.path.join(root_dir, "out.json")) as json_file:
        data = json.load(json_file)
    images_dir = os.path.join(root_dir, "images")

    for file in data:
        img_path = os.path.join(images_dir, file["filename"])
        img = cv2.imread(img_path)
        print(img_path)
        for object in file["objects"]:
            print(object["bbox"], img.shape)
            ymin, ymax, xmin, xmax = object["bbox"]['ymin'], object["bbox"]['ymax'], object["bbox"]['xmin'], object["bbox"]['xmax']
            if ymin == ymax or xmin == xmax:
                continue
            cropped_image = img[ymin:ymax, xmin:xmax]
            object["text"] = ocr(cropped_image)
    with open(os.path.join(root_dir, "out2.json"), "w") as write_file:
        json.dump(data, write_file, indent=4)
 
if __name__ == '__main__':
    start = time.time()
    main(root_dir = './Data/')
    end = time.time()
    print("OCR completed in {:0.2f}s".format(end - start))