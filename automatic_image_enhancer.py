from enancement_model import ImageEnhancer
import cv2
import numpy 
import os
import sys
import tensorflow as tf

class ImageLoader:
    def __init__(self, image):
        self.image = image
        
        try:
            os.makedirs('images_enhanced')
        except Exception as e:
            print(e, "Truncating files", sep=" \n ")
            for _, _, files in os.walk('images_enhanced'):
                for m in range (len(files)):
                    os.remove(f'images_enhanced\\{files[m]}')
                os.removedirs('images_enhanced')
        os.makedirs('images_enhanced')
        
    def process_images(self, crop_only_images=False):
        if crop_only_images:
            img = cv2.imread(self.image)
            faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(img, 1.3, 5)

            for i, face in enumerate(faces):
                cv2.imwrite(filename=f'images_enhanced\\image_{i}.jpg', img=img[face[1]:sum(face[1::2]), face[0]:sum(face[::2])])
        
            for _, _, files in os.walk('images_enhanced'):
                for i, file in enumerate(files):
                    ImageEnhancer(f"images_enhanced\\{file}", i)
        
        else:
            ImageEnhancer(image=self.image, number=0)
            for i in range(5):
                ImageEnhancer(f"images_enhanced\\Super_Resolution{i}.jpg", number=i+1)
                print(f'Image {i} saved')


a = ImageLoader(image="path/to/your/image")
a.process_images(crop_only_images=True)

