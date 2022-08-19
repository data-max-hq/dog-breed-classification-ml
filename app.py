import json
import requests
from wsgiref import headers
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time

# from seldon_core.seldon_client import SeldonClient
import logging
import json
import requests
import pickle

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# def send_client_request(seldon_client, image):
#     client_prediction = seldon_client.predict(
#         data=image,
#         deployment_name="seldon-dogbreed",
#         payload_type="ndarray",
#     )
#     return client_prediction


# sc = SeldonClient(
#     gateway="seldon",
#     transport="rest",
#     gateway_endpoint="seldon:9000",
#     microservice_endpoint="seldon:9000",
# )

# Function that transforms the image in the required format for the model
def get_test_generator():
    data_datagen = ImageDataGenerator(rescale=1.0 / 255)
    return data_datagen.flow_from_directory(
        "savedimage", target_size=(int(224), int(224)), batch_size=int(1)
    )


dog_classifier = tf.keras.applications.ResNet50V2(
    weights="imagenet", input_shape=(int(224), int(224), 3)
)


def is_dog(data):
    probs = dog_classifier.predict(data)
    preds = tf.argmax(probs, axis=1)
    return (preds >= 151) & (preds <= 268)


components.html(
    """
    <style>svg{
        width: 95vw;
        height: 40vh;
        margin-top: 20vh;
        }
        </style>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 90 13.06"><defs><style>.cls-1{fill:#333;}.cls-2{fill:#f15a24;}</style></defs><g id="Ebene_2" data-name="Ebene 2"><g id="Ebene_1-2" data-name="Ebene 1"><path class="cls-1" d="M11.68,3.33A5.83,5.83,0,0,0,9.22,1.08,8,8,0,0,0,5.59.29H0V13.06H5.59a8,8,0,0,0,3.63-.79A5.83,5.83,0,0,0,11.68,10a6.26,6.26,0,0,0,.88-3.34A6.27,6.27,0,0,0,11.68,3.33ZM9.59,9A3.82,3.82,0,0,1,8,10.52a5.41,5.41,0,0,1-2.47.54H2.37V2.29H5.48A5.41,5.41,0,0,1,8,2.83,3.82,3.82,0,0,1,9.59,4.37a4.45,4.45,0,0,1,.58,2.31A4.44,4.44,0,0,1,9.59,9Z"/><polygon class="cls-2" points="61.42 0.29 56.44 8.76 51.37 0.29 49.41 0.29 49.41 13.06 51.68 13.06 51.68 4.7 55.88 11.6 56.93 11.6 61.13 4.59 61.15 13.06 63.4 13.06 63.38 0.29 61.42 0.29"/><polygon class="cls-2" points="85.2 6.46 89.67 0.29 87.1 0.29 83.87 4.83 80.6 0.29 77.92 0.29 82.41 6.55 78.44 12 71.29 0 64.25 13.06 66.95 13.06 71.39 4.66 73.63 8.43 71.27 8.43 70.19 10.48 74.85 10.48 73.98 9.01 75.03 10.77 75.02 10.77 76.46 13.06 79.07 13.06 79.07 13.06 80.36 13.06 83.79 8.21 87.26 13.06 90 13.06 85.2 6.46"/><polygon class="cls-1" points="36.7 0.29 24.59 0.29 24.59 2.29 28.82 2.29 28.82 13.06 31.2 13.06 31.2 2.29 35.43 2.29 36.7 0.29"/><polygon class="cls-1" points="12.08 13.06 14.79 13.06 19.22 4.66 21.46 8.43 19.1 8.43 18.02 10.48 22.69 10.48 22.86 10.77 22.86 10.77 24.3 13.06 26.9 13.06 19.12 0 12.08 13.06"/><polygon class="cls-1" points="33.17 13.06 35.88 13.06 40.32 4.66 42.56 8.43 40.2 8.43 39.11 10.48 43.78 10.48 43.95 10.77 43.95 10.77 45.39 13.06 48 13.06 40.22 0 33.17 13.06"/></g></g></svg>
"""
)

tab1, tab2 = st.tabs(["📰 Intro", "🐶 Predict"])


with tab1:
    st.write("# Dog Breed Classification")
    st.write(
        "#### A demo that uses a ResNet model to predict the breed of a dog in a photo."
    )
    st.write("## *Prerequisites:*")
    st.write("- **Python 3.8.13**")
    st.write("- **Docker and Docker Compose**")
    st.write("- **k3d Cluster**")
    st.write("- **ZenML**")
    st.write("## *How to run the project with docker:*")
    st.write("1. Start by cloning the repository.", "bash")
    st.code("git clone https://github.com/data-max-hq/dog-breed-classification-ml.git")
    st.write("2. Open the project directory in terminal and type:")
    st.code("pip install -r requirements.txt", "bash")
    st.write("3. Build the docker images and run the containers.")
    st.code("docker compose up --build", "bash")

# Prediction Tab
with tab2:
    st.header("Upload a dog photo and press the Predict button to get a prediction!")

    image = st.file_uploader("Dog Photo: ", type=["jpg", "png", "jpeg"], key=1)

    if image != None:
        with open(os.path.join("savedimage/dog", "dog.png"), "wb") as f:
            f.write((image).getbuffer())
        with st.spinner("Loading image..."):
            time.sleep(0.2)
            st.image(image, use_column_width=True)
            predict_button = st.button("Predict", 2)
        # If predict button is clicked, transform the image, test if it is a dog image, serve it to the model and output the prediction.
        if predict_button != False:
            test_generator = get_test_generator()
            image = test_generator.next()[0][0]
            image = image[None, ...]

            if not is_dog(image):
                with st.spinner("Checking if the image contains a dog..."):
                    time.sleep(0.5)
                    st.error("Please enter a dog photo!")
            else:
                with st.spinner("Predicting the breed..."):
                    # prediction = send_client_request(sc, image)
                    # result = prediction.response["strData"]
                    
                    
                    
                    url = "http://tfserve:8501/v1/models/dog_model:predict"
                    data = json.dumps({"signature_name":"serving_default", "instances":image.tolist()})
                    headers = {"Content-Type": "application/json"}
                    response = requests.post(url, data=data, headers=headers)
                    prediction = json.loads(response.text)["predictions"]
                    pred = tf.argmax(prediction, axis=1)
                    
                    
                    with open("./models/labels.pickle", "rb") as handle:
                        idx_to_class1 = pickle.load(handle)

                    idx_to_class = {value: key for key, value in idx_to_class1.items()}
                    label = idx_to_class[pred.numpy()[0]]
                    result=label.split(".")[-1].replace("_", " ")

                    st.warning(f"The dog in the photo is: **{result}** :sunglasses:")
