from apps.DogBreed import DogBreed
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt 
import os
def get_test_generator():
    data_datagen = ImageDataGenerator(rescale=1./255)
    return data_datagen.flow_from_directory(
        "savedimage",
        target_size=(int(224), int(224)),
        batch_size=int(32)
    )
classifier = DogBreed(models_dir="models")

st.title('Dog Breed Classification')
st.write('This is a project that uses a trained CNN model to accurately predict the breed of a dog based on the picture given as input.')

image = st.file_uploader('Upload a dog photo: ', type=['jpg', 'png'], key=1)
if image is not None:
	#Saving upload
	with open(os.path.join("savedimage/001.dog",'dog.png'),"wb") as f:
		f.write((image).getbuffer())		  
	st.success("File Saved")

if image != None:
    st.image(image, use_column_width=True)

predict_button = st.button('Predict', 2)

if predict_button != False and image != None:
    test_generator = get_test_generator()
    image = test_generator.next()[0][0]
    image = image[None,...]
    st.write(classifier.predict(image))