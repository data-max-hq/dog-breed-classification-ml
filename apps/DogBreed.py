import logging
import tensorflow as tf
import pickle

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class DogBreed(object):
    """Class DogBreed"""

    def __init__(self):
        logging.info("load model here...")
        logging.info("model has been loaded and initialized...")
        self._dog_model = tf.keras.models.load_model("dog_model.h5")
        with open("labels.pickle", "rb") as handle:
            self._idx_to_class = pickle.load(handle)

    def predict(self, X, feature_names):
        """Predict Method"""
        logging.info(X)
        logging.info("Got request.")
        # probs = self._dog_model.predict(X)
        # logging.info(probs)
        logging.info(self._dog_model)
        # pred = tf.argmax(probs, axis=1)
        # logging.info(pred)
        idx_to_class = {value: key for key, value in self._idx_to_class.items()}
        logging.info(idx_to_class)
        # label = idx_to_class[pred.numpy()[0]]
        label = "hello_world"
        logging.info("Return Predicitions.")
        return label
