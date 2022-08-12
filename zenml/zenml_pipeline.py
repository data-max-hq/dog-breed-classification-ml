from zenml.pipelines import pipeline
from zenml.steps import step,Output,BaseStepConfig,STEP_ENVIRONMENT_NAME,StepContext
from zenml.integrations.constants import KUBEFLOW, SELDON
from zenml.integrations.seldon.model_deployers import SeldonModelDeployer
from zenml.integrations.seldon.services import SeldonDeploymentService
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from zenml.integrations.seldon.services.seldon_deployment import (
  SeldonDeploymentConfig,
  SeldonDeploymentService,
)
from PIL import ImageFile
import tensorflow as tf
import pickle, os, logging
import numpy as np
# import mlflow
from typing import Type,cast
from zenml.artifacts import DataArtifact,ModelArtifact
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.environment import Environment

# class MyObj(tf.keras.Model):
#     def __init__(self, name: str):
#         self.name = name


class ModelMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (tf.keras.Model,)
    ASSOCIATED_ARTIFACT_TYPES = (ModelArtifact,)

    def handle_input(self, data_type:Type[tf.keras.Model]) -> tf.keras.Model:
        """Read from artifact store"""
        super().handle_input(data_type)
        dog_model = tf.keras.models.load_model(f"{self.artifact.uri}/dog-model")

        return dog_model

    def handle_return(self, my_obj: tf.keras.Model) -> None:
        """Write to artifact store"""
        super().handle_return(my_obj)
        with fileio.open(os.path.join(self.artifact.uri, 'dog-model'), 'w') as f:
            f.write(my_obj.name)


LR = 6e-4
BATCH_SIZE = 32
NUMBER_OF_NODES = 256
EPOCHS = 1
IMG_SIZE = 224

def train_generator():
    data_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
    )
    return data_datagen.flow_from_directory(
        "dogImages/train/",
        target_size=(int(IMG_SIZE), int(IMG_SIZE)),
        batch_size=int(BATCH_SIZE),
    )

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TensorflowTrainerConfig(BaseStepConfig):
    """Trainer params"""

    epochs: int = 1
    lr: float = 0.001

@step(enable_cache=True)
def train(config: TensorflowTrainerConfig) -> tf.keras.Model:

    ImageFile.LOAD_TRUNCATED_IMAGES = True
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

    data_datagen_2 = ImageDataGenerator(rescale=1.0 / 255)
    valid_generator = data_datagen_2.flow_from_directory(
        "dogImages/valid/",
        target_size=(int(IMG_SIZE), int(IMG_SIZE)),
        batch_size=int(BATCH_SIZE),
    )

    # mlflow.tensorflow.autolog()

    t_generator = train_generator()
    resnet_model.fit(
        t_generator, epochs=int(EPOCHS), validation_data=valid_generator
    )

    # This save might not needed when using
    #  zenml bcs zenml saves model when doing return in step
    resnet_model.save("models_2/dog-model") 

    return resnet_model

@step
def save_labels() -> dict:
    t_generator = train_generator()
    labels = t_generator.class_indices
    
    # This save might also not be needed when
    #  using zenml bcs zenml saves labels when doing return in step
    with open("models_2/labels.pickle", "wb") as handle:
        pickle.dump(labels, handle)
        
    return labels


# Get image data from streamlit 
@step
def dynamic_data_importer() -> Output(data=np.ndarray):
    return np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    
@step
def evaluator() -> bool:
    return True

class SeldonDeploymentLoaderStepConfig(BaseStepConfig):
    """Seldon deployment loader configuration
    Attributes:
        pipeline_name: name of the pipeline that deployed the Seldon prediction
            server
        step_name: the name of the step that deployed the Seldon prediction
            server
        model_name: the name of the model that was deployed
    """

    pipeline_name: str
    step_name: str
    model_name: str

@step(enable_cache=False)
def prediction_service_loader(
    config: SeldonDeploymentLoaderStepConfig,
) -> SeldonDeploymentService:
    """Get the prediction service started by the deployment pipeline"""

    model_deployer = SeldonModelDeployer.get_active_model_deployer()
    logging.info(model_deployer)

    services = model_deployer.find_model_server(
        pipeline_name=config.pipeline_name,
        pipeline_step_name=config.step_name,
        model_name=config.model_name,
    )
    if not services:
        logging.info("--->>>>> Seldon not available")
        raise RuntimeError(
            f"No Seldon Core prediction server deployed by the "
            f"'{config.step_name}' step in the '{config.pipeline_name}' "
            f"pipeline for the '{config.model_name}' model is currently "
            f"running."
        )

    if not services[0].is_running:
        logging.info("--->>>>> Seldon not running")
        raise RuntimeError(
            f"The Seldon Core prediction server last deployed by the "
            f"'{config.step_name}' step in the '{config.pipeline_name}' "
            f"pipeline for the '{config.model_name}' model is not currently "
            f"running."
        )

    return cast(SeldonDeploymentService, services[0])


@step
def predictor(
    service: SeldonDeploymentService,
    data: np.ndarray,
) -> str: #Output(predictions=np.ndarray):
    """Run a inference request against a prediction service"""

    service.start(timeout=120)  # should be a NOP if already started
    prediction = service.predict(data)
    prediction = prediction.argmax(axis=-1)
    idx_to_class = {value: key for key, value in idx_to_class.items()}
    logging.info(idx_to_class)
    label = idx_to_class[prediction.numpy()[0]]
    logging.info("Return prediction.")
    return label.split(".")[-1].replace("_", " ")

@pipeline(
    enable_cache=True, required_integrations=[KUBEFLOW,SELDON],requirements="zenml_requirements.txt"
)
def continuous_deployment_pipeline(
    trainer,
    save_labels,
    evaluator,
    model_deployer
):
    model = trainer()
    labels = save_labels()
    evaluat = evaluator()
    model_deployer(evaluat, model)


@pipeline(
    enable_cache=True, required_integrations=[KUBEFLOW,SELDON],requirements="zenml_requirements.txt"
)
def inference_pipeline(
    dynamic_data_importer,
    prediction_service_loader,
    predictor,
):
    data = dynamic_data_importer()
    model_deployment_service = prediction_service_loader()
    predictor(model_deployment_service,data)


# @step(enable_cache=True)
# def seldon_model_deployer_step(
#   context: StepContext,
#   model: ModelArtifact,
# ) -> SeldonDeploymentService:
#   model_deployer = SeldonModelDeployer.get_active_model_deployer()

#   # get pipeline name, step name and run id
#   step_env = Environment()[STEP_ENVIRONMENT_NAME]
  
#   logging.info("Model uri -->",model.uri)
#   service_config = SeldonDeploymentConfig(
#       model_uri=model.uri,
#       model_name="dog_model.h5",
#       implementation="TENSORFLOW_SERVER",
#       pipeline_name = step_env.pipeline_name,
#       pipeline_run_id = step_env.pipeline_run_id,
#       pipeline_step_name = step_env.step_name,
#   )

#   service = model_deployer.deploy_model(
#       service_config, replace=True, timeout=300
#   )

#   print(
#       f"Seldon deployment service started and reachable at:\n"
#       f"    {service.prediction_url}\n"
#   )

#   return service


# @pipeline(
#     enable_cache=True, required_integrations=[KUBEFLOW,SELDON],requirements="zenml_requirements.txt"
# )
# def first_pipeline(
#     train,
#     # save_labels,
#     seldon_model_deployer_step,
# ):
#     # train = train()
#     # save_labels()
#     seldon_model_deployer_step(train())

# if __name__ == "__main__":
#     my_pipeline = first_pipeline(
#         train().with_return_materializers(ModelMaterializer),
#         # save_labels = save_labels(),#.with_return_materializers(LabelsMaterializer)
#         seldon_model_deployer_step()
#     )

#     my_pipeline.run()
