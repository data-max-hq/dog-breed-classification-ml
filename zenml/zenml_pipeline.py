from zenml.pipelines import pipeline
from zenml.steps import step
from zenml.integrations.constants import KUBEFLOW
# from zenml.steps.step_output import Output
import logging

logging.basicConfig()

# Test pipeline
@step
def print_message() ->str:
    return "This is a test"



@pipeline(required_integrations=[KUBEFLOW])
def my_pipeline(
    print_message
):
    message = print_message()
    logging.info(message)


if __name__ == "__main__":
    first_pipeline = my_pipeline(
        print_message = print_message()
    )

    first_pipeline.run()