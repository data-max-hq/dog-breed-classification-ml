import kfp
import kfp.dsl as dsl
import kfp.compiler as compiler
from kubernetes import client

EXPERIMENT_NAME = "Dog Breed Pipeline"


@dsl.pipeline(
    name="DogBreed Classification", description="Just a test pipeline for dogbreed"
)
def train_model_pipeline():
    """Create a Training Pipeline for dogbreed classification"""

    dsl.ContainerOp(name="train-model", image="trainmodel:minikube",).add_volume(
        client.V1Volume(
            name="model-volume", host_path=client.V1HostPathVolumeSource("/mnt")
        )
    ).add_volume_mount(client.V1VolumeMount(mount_path="/models", name="model-volume"))


if __name__ == "__main__":
    pipeline_func = train_model_pipeline
    pipeline_filename = pipeline_func.__name__ + ".yaml"
    compiler.Compiler().compile(pipeline_func, pipeline_filename)
    clien = kfp.Client()
    experiment = clien.create_experiment(EXPERIMENT_NAME)
    run_name = pipeline_func.__name__ + " run"
    run_result = clien.run_pipeline(experiment.id, run_name, pipeline_filename, {})
