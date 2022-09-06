from PIL import Image
from io import BytesIO
from kafka import KafkaConsumer
import logging
import json
import requests
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
from seldon_core.seldon_client import SeldonClient
from slack_sdk import WebClient

slack_token = os.environ["SLACK_BOT_TOKEN"]
client = WebClient(token=slack_token)

logging.basicConfig(level=logging.INFO)
consumer = KafkaConsumer("dogtopic",bootstrap_servers=['localhost:9092'])

sc = SeldonClient(
        gateway="seldon",
        transport="rest",
        gateway_endpoint="localhost:9000"
    )


def send_client_request(seldon_client, image):
    client_prediction = seldon_client.predict(
        data=image,
        deployment_name="seldon-dogbreed",
        payload_type="ndarray",
    )
    return client_prediction


def get_test_generator():
    data_datagen = ImageDataGenerator(rescale=1.0 / 255)
    return data_datagen.flow_from_directory(
        "savedimage", target_size=(int(224), int(224)), batch_size=int(1)
    )


def send_slack_message(payload, webhook):
    return requests.post(webhook, json.dumps(payload))


for message in consumer:
    stream = BytesIO(message.value)
    image = Image.open(stream).convert("RGBA")
    logging.info('image recived')
    if image is not None:
        test_generator = get_test_generator()
        image = test_generator.next()[0][0]
        image = image[None, ...]
        prediction = send_client_request(sc, image)
        response = prediction.response.get("data").get("ndarray")
        pred = tf.argmax(response, axis=1)
        ## code for tf-serving
        # url = "http://localhost:8501/v1/models/dog_model:predict"
        # data = json.dumps(
        #     {
        #         "signature_name": "serving_default",
        #         "instances": image.tolist(),
        #     }
        # )
        # headers = {"Content-Type": "application/json"}
        # response = requests.post(url, data=data, headers=headers)
        # prediction = json.loads(response.text)["predictions"]
        # pred = tf.argmax(prediction, axis=1)
        with open("./models/labels.pickle", "rb") as handle:
            idx_to_class1 = pickle.load(handle)
        idx_to_class = {value: key for key, value in idx_to_class1.items()}
        label = idx_to_class[pred.numpy()[0]]
        result = label.split(".")[-1].replace("_", " ")
        logging.info(f'The dog is {result}')
        channel_id = os.environ['SLACK_CHANNEL_ID']
        response = client.files_upload(
            channels=channel_id,
            file="./dog.jpg",
            title=f"seldon : {result}"
        )
        stream.close()