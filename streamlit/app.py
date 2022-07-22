import streamlit as st

st.title('Dog Breed Classification')
st.write('This is a project that uses a trained CNN model to accurately predict the breed of a dog based on the picture given as input.')

image = st.file_uploader('Upload a dog photo: ', type=['jpg', 'png'], key=1)

if image != None:
    st.image(image, use_column_width=True)