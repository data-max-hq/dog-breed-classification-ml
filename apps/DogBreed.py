import logging
import tensorflow as tf
import pickle
import numpy as np

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class DogBreed(object):
    """Class DogBreed"""

    def __init__(self, models_dir="dog_model.h5"):
        self.loaded = False
        logging.info("load model here...")
        self._models_dir = models_dir

    def load(self):

        self._dog_model = tf.keras.models.load_model(f"{self._models_dir}")
        with open("labels.pickle", "rb") as handle:
            self._idx_to_class = pickle.load(handle)
        self.loaded = True
        logging.info("model has been loaded and initialized...")

    def predict(self, X, feature_names):
        """Predict Method"""
        if not self.loaded:
            logging.info("Not loaded yet.")
            self.load()
        logging.info("Model loaded.")
        logging.info(X)
        int_image = X.astype(np.uint8)

        logging.info("Got request.")
        probs = self._dog_model.predict(int_image)
        logging.info(f"probs: {probs}")
        pred = tf.argmax(probs, axis=1)
        logging.info(pred)
        idx_to_class = {value: key for key, value in self._idx_to_class.items()}
        logging.info(idx_to_class)
        label = idx_to_class[pred.numpy()[0]]
        logging.info("Return prediction.")
        return label.split(".")[-1].replace("_", " ")
