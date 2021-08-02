"""
This is the model for enhancing the images
Usig the SRGAN model over here

"""

import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt


class Image_enhancer ():
    def __init__(self, image, number):
        self.image_path=image
        self.model_path="https://tfhub.dev/captain-pool/esrgan-tf2/1"
        
        model = hub.load(self.model_path)
        
        hr_image = self.process_image()
        
        fake_image = model(hr_image)
        
        fake_image = tf.squeeze(fake_image)
        self.save_image(tf.squeeze(fake_image), filename=f"images_enhanced\\Super_Resolution{number}")
        
    def process_image (self):
        hr_image = tf.image.decode_image(tf.io.read_file(self.image_path))
        
        if hr_image.shape[-1] == 4:
          hr_image = hr_image[...,:-1]
        
        hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
        hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
        hr_image = tf.cast(hr_image, tf.float32)
        
        return tf.expand_dims(hr_image, 0)

    def save_image(self,image, filename):
        
        if not isinstance(image, Image.Image):
            image = tf.clip_by_value(image, 0, 255)
            image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
            
        image.save("%s.jpg" % filename)
        
        print("Saved as %s.jpg" % filename)  
