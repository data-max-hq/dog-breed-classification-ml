import yaml
import kfp
import kfp.components as comp
import kfp.compiler as compiler
import kfp.dsl as dsl


@dsl.pipeline(
    name="DogBreed Classification", description="Just a test pipeline for dogbreed"
)
def train_model_pipeline():
    vop1 = dsl.VolumeOp(
        name="Create Volume",
        resource_name="vol1",
        size="1Gi",
        modes=dsl.VOLUME_MODE_RWM,
    )

    step1 = dsl.ContainerOp(
        name="Train Model",
        image="trainmodel:minikube",
        pvolumes={"/models": vop1.volume},
    )


if __name__ == "__main__":
    pipeline_func = train_model_pipeline
    pipeline_filename = pipeline_func.__name__ + ".yaml"
    compiler.Compiler().compile(pipeline_func, pipeline_filename)
