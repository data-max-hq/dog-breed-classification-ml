import kfp as kfp
import kfp.dsl as dsl


def train():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import pickle
    from PIL import ImageFile
    import os

    os.system(
        "wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip"
    )
    os.system("unzip -qo dogImages.zip")
    os.system("rm dogImages.zip")
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    LR = 6e-4
    BATCH_SIZE = 32
    NUMBER_OF_NODES = 256
    EPOCHS = 1
    IMG_SIZE = 224

    resnet_body = tf.keras.applications.ResNet50V2(
        weights="imagenet",
        include_top=False,
        input_shape=(int(IMG_SIZE), int(IMG_SIZE), 3),
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
        metrics=["accuracy"],
    )

    data_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
    )

    train_generator = data_datagen.flow_from_directory(
        "dogImages/train/",
        target_size=(int(IMG_SIZE), int(IMG_SIZE)),
        batch_size=int(BATCH_SIZE),
    )

    data_datagen1 = ImageDataGenerator(rescale=1.0 / 255)

    valid_generator = data_datagen1.flow_from_directory(
        "dogImages/valid/",
        target_size=(int(IMG_SIZE), int(IMG_SIZE)),
        batch_size=int(BATCH_SIZE),
    )

    resnet_model.fit(
        train_generator, epochs=int(EPOCHS), validation_data=valid_generator
    )

    resnet_model.save("models/dog_model.h5")

    labels = train_generator.class_indices
    with open("models/labels.pickle", "wb") as handle:
        pickle.dump(labels, handle)


vop1 = dsl.VolumeOp(
    name="create_volume_1", resource_name="vol1", size="1Gi", modes=dsl.VOLUME_MODE_RWM
)


train_model = kfp.components.func_to_container_op(
    func=train,
    base_image="python:3.8",
    packages_to_install=["tensorflow==2.3.0", "pillow==7.2.0"],
)


@dsl.pipeline(
    name="DogBreed Pipeline", description="This is a Tes for DogBreed Pipeline"
)
def train_model_pipeline():
    """Train model Pipeline"""

    train_model()


if __name__ == "__main__":
    # Compile the pipeline
    import kfp.compiler as compiler
    import logging

    logging.basicConfig(level=logging.INFO)
    pipeline_func = train_model_pipeline
    pipeline_filename = pipeline_func.__name__ + ".yaml"
    compiler.Compiler().compile(pipeline_func, pipeline_filename)
    logging.info(f"Generated pipeline file: {pipeline_filename}.")
