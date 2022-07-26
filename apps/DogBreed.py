import logging
import tensorflow as tf
import pickle

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class DogBreed(object):
  """Class DogBreed"""
  def __init__(self, models_dir="/models"):
    logging.info("load model here...")
    self._models_dir = models_dir
    logging.info("model has been loaded and initialized...")
    self._dog_model = tf.keras.models.load_model(f"{self._models_dir}/dog_model.h5")
    with open(f"{self._models_dir}/labels.pickle", 'rb') as handle:
        self._idx_to_class  = pickle.load(handle)
    
    
  def predict(self, X):
    """Predict Method"""
    logging.info("Got request.")
    probs=self._dog_model.predict(X)
    pred = tf.argmax(probs, axis=1)
    idx_to_class = {value: key for key, value in self._idx_to_class.items()}
    label = idx_to_class[pred.numpy()[0]]
    return label.split(".")[-1].replace('_', ' ')
