import yaml
import kfp
import kfp.components as comp
import kfp.compiler as compiler
import kfp.dsl as dsl
from kubernetes import client, config

@dsl.pipeline(
    name="DogBreed Classification", description="Just a test pipeline for dogbreed"
)
def train_model_pipeline():

    dsl.ContainerOp(
         name="train-model",
         image="trainmodel:minikube",
     ).add_volume(client.V1Volume(name='model-volume' ,host_path=client.V1HostPathVolumeSource("/mnt"))).add_volume_mount(client.V1VolumeMount(
      mount_path='/models', name='model-volume')) 



if __name__ == "__main__":
    pipeline_func = train_model_pipeline
    pipeline_filename = pipeline_func.__name__ + ".yaml"
    compiler.Compiler().compile(pipeline_func, pipeline_filename)
