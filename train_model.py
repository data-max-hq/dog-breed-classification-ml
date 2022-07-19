from apps.DogBreed import DogBreed
import logging
import dill
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

LR = 6e-4
BATCH_SIZE = 32
NUMBER_OF_NODES = 256
EPOCHS = 4
IMG_SIZE = 224

def get_train_generator():
    data_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=.2,
        height_shift_range=.2,
        brightness_range=[0.5,1.5],
        horizontal_flip=True
    )
    return data_datagen.flow_from_directory(
        "dogImages/train/",
        target_size=(int(IMG_SIZE), int(IMG_SIZE)),
        batch_size=int(BATCH_SIZE),
        
    )

def get_valid_generator():
    data_datagen = ImageDataGenerator(rescale=1./255)
    return data_datagen.flow_from_directory(
        "dogImages/valid/",
        target_size=(int(IMG_SIZE), int(IMG_SIZE)),
        batch_size=int(BATCH_SIZE)
    )


def train():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 3, activation="relu", input_shape=(int(IMG_SIZE), int(IMG_SIZE), 3)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(int(NUMBER_OF_NODES), activation="relu"),
        tf.keras.layers.Dense(133, activation="softmax")
    ])

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=float(LR)),
        loss=tf.losses.categorical_crossentropy,
        metrics=["accuracy"]
    )
    
    train_generator = get_train_generator()
    valid_generator = get_valid_generator()

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="custom_classifier_logs")

    model.fit(train_generator, epochs=2,
        validation_data=valid_generator,
        callbacks=[tb_callback]
    )
    
    logging.info("Dump models.")
    with open("models/model.model", "wb") as model_file:
        dill.dump(model, model_file)

    logging.info("Finished training.")

train()

logging.info("Test model prediction.")
classifier = DogBreed(models_dir="models")