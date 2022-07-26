import yaml
import kfp
import kfp.components as comp
import kfp.compiler as compiler
import kfp.dsl as dsl

# localhost:8877/ambassador/v0/diag/


def train_model():
    """
    Op to train model.
    """
    vop1 = dsl.VolumeOp(
        name="create_volume_1",
        resource_name="vol1",
        size="1Gi",
        modes=dsl.VOLUME_MODE_RWM
    )


    return dsl.ContainerOp(
        name="Train Model",
        image="trainmodel:minikube",
        pvolumes={"/data": vop1.volume}
    )

@dsl.pipeline(
    name="DogBreed Classification",
    description="Just a test pipeline for dogbreed"
)
def train_model_pipeline():
    train_task = train_model()



if __name__ == "__main__":
    pipeline_func = message_pipeline
    pipeline_filename = pipeline_func.__name__ + ".yaml"
    compiler.Compiler().compile(pipeline_func, pipeline_filename)
