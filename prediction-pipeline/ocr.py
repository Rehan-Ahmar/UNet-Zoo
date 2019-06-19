import pytesseract
import cv2
import re
 
def ocr(img_path, out_dir):
    img = cv2.imread(img_path)
    text = pytesseract.image_to_string(img)
    out_file = re.sub(".png",".txt",img_path.split("/")[-1])
    out_path = out_dir + out_file
    fd = open(out_path,"w")
    fd.write("%s" %text)
    return out_file
	
ocr(r'/home/vmadmin/rehan/unet-demo/pipeline/Data/images/1538681118144_4760218.Termination_Agreement._.PDF-1.png', r'ocr_result/')