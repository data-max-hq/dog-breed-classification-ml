from apps.DogBreed import DogBreed
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
import matplotlib.pyplot as plt
import pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
LR = 6e-4
BATCH_SIZE = 32
NUMBER_OF_NODES = 256
EPOCHS = 1
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
    logging.info("Training Model.")

    resnet_body = tf.keras.applications.ResNet50V2(
        weights="imagenet",
        include_top=False,
        input_shape=(int(IMG_SIZE), int(IMG_SIZE), 3)
    )
    resnet_body.trainable = False
    inputs = tf.keras.layers.Input(shape=(int(IMG_SIZE), int(IMG_SIZE), 3))
    x = resnet_body(inputs, training=False)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(133, activation="softmax")(x)
    resnet_model = tf.keras.Model(inputs, outputs)
    resnet_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=float(LR)),
        loss=tf.losses.categorical_crossentropy,
        metrics=["accuracy"]
    )

    train_generator = get_train_generator()
    valid_generator = get_valid_generator()

    resnet_model.fit(train_generator, epochs=int(EPOCHS),
                     validation_data=valid_generator
                     )

    logging.info("Dump models.")
    resnet_model.save('models/dog_model.h5')

    logging.info("Finished training.")

    labels = train_generator.class_indices
    with open('models/labels.pickle', 'wb') as handle:
        pickle.dump(labels, handle)
    
#train()

logging.info("Test model prediction.")
classifier = DogBreed(models_dir="models")

train_generator = get_train_generator()
image = train_generator.next()[0][0]
plt.imshow(image)
plt.show()
image = image[None,...]

logging.info(classifier.predict(image))


