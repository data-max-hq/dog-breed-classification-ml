import io
import logging
import numpy as np
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

IMG_SIZE = 224

logger = logging.getLogger('__mymodel__')


class DogBreed():
  dog_classifier = tf.keras.applications.ResNet50V2(
      weights="imagenet",
      input_shape=(int(IMG_SIZE), int(IMG_SIZE), 3)
  )

  def __init__(self):
      logger.info("initializing...")
      logger.info("load model here...")
      self._dog_model = tf.keras.models.load_model('/models/dog_model.h5')

  def is_dog(self,data):
    probs = self.dog_classifier.predict(data)
    preds = tf.argmax(probs, axis=1)
    return ((preds >= 151) & (preds <= 268))

  def predict_breed(self,image):
      probs = self.dog_model.predict(image)
      pred = tf.argmax(probs, axis=1)
      return pred

  def predict_dog(self,image):
      image = image[None,...]
      if self.is_dog(image):
          pred =  self.predict_breed(image)
          print(f"This photo looks like a(n) {pred}.")
          return
      print("No dog detected")

  # def predict(self,):

  #   logger.info("predict called...")
  #   logger.info('converting tensor to image')

    # img = Image.open(io.BytesIO(X)).convert('RGB')
    # img = np.array(img)
    # img = img[:,:,::-1]
    # logger.info("image size = {}".format(img.shape))

    
    #idx_to_class = {value: key for key, value in train_generator.class_indices.items()}
