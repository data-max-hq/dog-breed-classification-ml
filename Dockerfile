FROM python:3.8.2
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./models/dog_model.h5 /models/dog_model.h5

COPY ./apps .
COPY train_model.py train_model.py

EXPOSE 9000

ENV MODEL_NAME DogBreed
ENV SERVICE_TYPE MODEL
