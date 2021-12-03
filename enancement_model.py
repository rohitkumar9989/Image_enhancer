"""
This is the model for enhancing the images
using the SRGAN model over here
"""

from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

class ImageEnhancer:
    def __init__(self, image, number):
        model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
        hr_image = self.process_image(image)
        fake_image = tf.squeeze(tf.squeeze(model(hr_image)))
        self.save_image(fake_image, filename=f"images_enhanced\\Super_Resolution{number}")
        
    def process_image(self, image_path):
        hr_image = tf.image.decode_image(tf.io.read_file(image_path))
        
        if hr_image.shape[-1] == 4:
          hr_image = hr_image[...,:-1]
        
        hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
        return tf.expand_dims(tf.cast(tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1]), tf.float32), 0)

    def save_image(self,image, filename):
        image = image if isinstance(image, Image.Image) else Image.fromarray(tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8).numpy())
        image.save("%s.jpg" % filename)
        print("Saved as %s.jpg" % filename)  
