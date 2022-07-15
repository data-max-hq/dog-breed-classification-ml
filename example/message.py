import yaml
import kfp
import kfp.components as comp
import kfp.compiler as compiler
import kfp.dsl as dsl



def print_op(msg):
    """
    Op to print a message.
    """
    return dsl.ContainerOp(
        name="Print message.",
        image="alpine:3.6",
        command=["echo", msg],
    )

@dsl.pipeline(
    name="Print Message Kubeflow",
    description="I want to print a new message to the Kubeflow Pipeline"
)
def message_pipeline():
    print_message = print_op("Hello World")



if __name__ == "__main__":
    pipeline_func = message_pipeline
    pipeline_filename = pipeline_func.__name__ + ".yaml"
    compiler.Compiler().compile(pipeline_func, pipeline_filename)
