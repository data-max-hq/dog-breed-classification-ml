import logging
from telnetlib import X3PAD
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image 

class DogBreed(object):

  def __init__(self, models_dir="/models"):
    logging.info("load model here...")
    self._models_dir = models_dir
    logging.info("model has been loaded and initialized...")
    self._dog_model = tf.keras.models.load_model(f"{self._models_dir}/dog_model.h5")

    
  def predict(self, X, features_names):
    """ Seldon Core Prediction API """
    logging.info("Got request.")
    logging.info(f"X={X}.")
    image = X.next()[0][0]
    plt.imshow(image)
    plt.show()
    image = image[None,...]
    probs=self._dog_model.predict(image)
    pred = tf.argmax(probs, axis=1)
    idx_to_class = {value: key for key, value in X.class_indices.items()}
    label = idx_to_class[pred.numpy()[0]]    

    print(label.split(".")[-1])
    
    return label.split(".")[-1]



