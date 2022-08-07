from zenml.pipelines import pipeline
from zenml.steps import step,Output
from zenml.integrations.constants import KUBEFLOW, SELDON, TENSORFLOW
# from zenml.integrations.seldon.model_deployers import SeldonModelDeployer
# from zenml.integrations.seldon.services import SeldonDeploymentService
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
import tensorflow as tf
import pickle, os, logging
# import mlflow
from typing import Type
from zenml.artifacts import DataArtifact
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer


# class MyObj(tf.keras.Model):
#     def __init__(self, name: str):
#         self.name = name


class MyMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (tf.keras.Model,)
    ASSOCIATED_ARTIFACT_TYPES = (DataArtifact,)

    def handle_input(self, data_type: Type[tf.keras.Model]) -> tf.keras.Model:
        """Read from artifact store"""
        super().handle_input(data_type)
        dog_model = tf.keras.models.load_model(f"{self.artifact.uri}/dog_model.h5")
        name = dog_model.read()

        # with open(f"{self.artifact.uri}/labels.pickle", "rb") as handle:
        #     idx_to_class = pickle.load(handle)

        # with fileio.open(os.path.join(self.artifact.uri, 'dog_model.h5'), 'r') as f:
        #     name = f.read()
        return dog_model

    def handle_return(self, my_obj: tf.keras.Model) -> None:
        """Write to artifact store"""
        super().handle_return(my_obj)
        with fileio.open(os.path.join(self.artifact.uri, 'dog_model.h5'), 'w') as f:
            f.write(my_obj.name)
        # with open(f"{self.artifact.uri}/labels.pickle", "wb") as handle:
        #     pickle.dump(labels, handle)

        # Problem is that we want to save at artifact store dog_model and also lables.pickle
        # But there is an error when we dump labels as pickle file,bcs labels are not defined


@step()
def train() -> tf.keras.Model:
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    LR = 6e-4
    BATCH_SIZE = 32
    NUMBER_OF_NODES = 256
    EPOCHS = 1
    IMG_SIZE = 224

    logging.info("Training...")

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

    logging.info("Got here ...")

    data_datagen_2 = ImageDataGenerator(rescale=1.0 / 255)
    valid_generator = data_datagen_2.flow_from_directory(
        "dogImages/valid/",
        target_size=(int(IMG_SIZE), int(IMG_SIZE)),
        batch_size=int(BATCH_SIZE),
    )

    # mlflow.tensorflow.autolog()

    resnet_model.fit(
        train_generator, epochs=int(EPOCHS), validation_data=valid_generator
    )

    #os.system("mkdir models_2")

    resnet_model.save("models_2/dog_model.h5")

    logging.info("Got here 222...")

    labels = train_generator.class_indices
    with open("models_2/labels.pickle", "wb") as handle:
        pickle.dump(labels, handle)

    logging.info("Finished...")

    return resnet_model


@pipeline(enable_cache=False, required_integrations=[KUBEFLOW],requirements="zenml_requirements.txt")
def first_pipeline(
    train
):
    trained_model = train()


if __name__ == "__main__":
    # os.system(
    #     "wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip"
    # )
    # os.system("unzip -qo dogImages.zip")
    # os.system("rm dogImages.zip")

    my_pipeline = first_pipeline(
        train = train().with_return_materializers(MyMaterializer)
    )
    logging.info("Got here 3333..")

    my_pipeline.run()