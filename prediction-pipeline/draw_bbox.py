import os
import json
import cv2
import skimage
from detect_text import detect_text_in_image

def generate_bbox(images_path, masks_path, outputs_path):
    files = []
    for image_name in os.listdir(masks_path):
        try:
            image = cv2.imread(os.path.join(images_path, image_name))
            file = { "filename": image_name, "size": {"height": image.shape[0], "width": image.shape[1]} }
            mask = skimage.io.imread(os.path.join(masks_path, image_name))
            binary = mask <= skimage.filters.threshold_otsu(mask)
            labels = skimage.measure.label(binary)

            props = skimage.measure.regionprops(labels)
            objects = []
            for prop in props:
                bbox = prop.bbox

                crop_xmin = bbox[1]
                crop_ymin = bbox[0]
                crop_xmax = bbox[3]
                crop_ymax = bbox[2]
                cropped_image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
                inside_bboxes = detect_text_in_image(cropped_image)
            
                cv2.rectangle(image, (crop_xmin, crop_ymin), (crop_xmax, crop_ymax), (0, 255, 0), 2)
                for ibbox in inside_bboxes:
                    cv2.rectangle(image, (crop_xmin + ibbox["xmin"], crop_ymin + ibbox["ymin"]), (crop_xmin + ibbox["xmax"], crop_ymin + ibbox["ymax"]), (0, 0, 255), 1)
                    object = { "label": "text", "bbox": { "xmin": crop_xmin + ibbox["xmin"], "xmax": crop_xmin + ibbox["xmax"], "ymin": crop_ymin + ibbox["ymin"], "ymax": crop_ymin + ibbox["ymax"] }}
                    objects.append(object)
                    
            file["objects"] = objects
            files.append(file)
            final_image_name = image_name[:-4] + ".jpg"
            cv2.imwrite(os.path.join(outputs_path, final_image_name), image)
            print (image_name, "done")
        except:
            print ("**", image_name, "error")
    with open("out.json", "w") as write_file:
        json.dump(files, write_file, indent=4)
    return files