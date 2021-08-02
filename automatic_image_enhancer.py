from enancement_model import Image_enhancer
import cv2
import numpy 
import os
import sys
import tensorflow as tf

class Image_loader ():
    def __init__(self, image):
        try:
            self.image=image
        except Exception as e:
            print (e)
        try:
            os.makedirs('images_enhanced')
        except Exception as e:
            print (f"{e} \n Truncating files")
            for _, _, files in os.walk('images_enhanced'):
                for m in range (len(files)):
                    os.remove(f'images_enhanced\\{files[m]}')
                os.removedirs('images_enhanced')
        os.makedirs('images_enhanced')
    def process_images (self, crop_only_images=False):
        if crop_only_images==False:
            Image_enhancer(image=self.image, number=0)

            for i in range (5):
                made_path=f"images_enhanced\\Super_Resolution{i}.jpg"
                Image_enhancer(made_path, number=i+1)
                print ('Image {} saved'.format(i))
        else:
            main_list=[]
            cas=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            img=cv2.imread(self.image)
            faces = cas.detectMultiScale(img, 1.3, 5)
            for x, y, w, h in faces:
                main_list.append([x, y, w, h])

            for i in range (len(main_list)):
                main_list[i][1]
                crop_image=img[main_list[i][1]:main_list[i][1]+main_list[i][3], main_list[i][0]: main_list[i][0]+main_list[i][2]]
                cv2.imwrite(filename=f'images_enhanced\\image_{i}.jpg', img=crop_image)
            if len("images_enhanced")==0:
              print ("The image doesn't contain any faces please set the `crop_only_images` param to False")
              sys.exit()
            else:
              pass
            for _, _, files in os.walk('images_enhanced'):
                for i in range (len(files)):
                    Image_enhancer(f"images_enhanced\\{files[i]}", i)


a=Image_loader(image=#Your Image_path)
a.process_images(crop_only_images=True)

