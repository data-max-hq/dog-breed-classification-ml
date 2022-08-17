from zenml_pipeline import (
    SeldonDeploymentLoaderStepConfig,
    TensorflowTrainerConfig,
    train,
    save_labels,
    dynamic_data_importer,
    prediction_service_loader,
    tf_evaluator,
    predictor,
    continuous_deployment_pipeline,
    inference_pipeline,
)
from zenml.integrations.seldon.model_deployers import SeldonModelDeployer
from zenml.integrations.seldon.services import (
    SeldonDeploymentConfig,
    SeldonDeploymentService,
)
from zenml.integrations.seldon.steps import (
    SeldonDeployerStepConfig,
    seldon_model_deployer_step,
)
from rich import print
from typing import cast
import click
import os

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default="deploy_and_predict",
    help="Optionally you can choose to only run the deployment "
    "pipeline to train and deploy a model (`deploy`), or to "
    "only run a prediction against the deployed model "
    "(`predict`). By default both will be run "
    "(`deploy_and_predict`).",
)
def main(config: str):
    """Run the Seldon example continuous deployment or inference pipeline
    Example usage:
        python run.py --deploy --predict --model-flavor tensorflow \
             --min-accuracy 0.80
    """
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    seldon_implementation = "TENSORFLOW_SERVER"
    model_name = "dog-breed"
    deployment_pipeline_name = "continuous_deployment_pipeline"
    deployer_step_name = "seldon_model_deployer_step"

    trainer_config = TensorflowTrainerConfig(epochs=1, lr=0.003)
    trainer = train(trainer_config)
    evaluator = tf_evaluator()

    model_deployer = SeldonModelDeployer.get_active_model_deployer()

    if deploy:
        # Initialize a continuous deployment pipeline run
        deployment = continuous_deployment_pipeline(
            trainer=trainer,
            save_labels=save_labels(),
            evaluator=evaluator,
            model_deployer=seldon_model_deployer_step(
                config=SeldonDeployerStepConfig(
                    service_config=SeldonDeploymentConfig(
                        model_name=model_name,
                        replicas=1,
                        implementation=seldon_implementation,
                    ),
                    timeout=120,
                )
            ),
        )

        deployment.run()

    if predict:
        # Initialize an inference pipeline run
        inference = inference_pipeline(
            dynamic_data_importer(),
            prediction_service_loader=prediction_service_loader(
                SeldonDeploymentLoaderStepConfig(
                    pipeline_name=deployment_pipeline_name,
                    step_name=deployer_step_name,
                    model_name=model_name,
                )
            ),
            predictor=predictor(),
        )

        inference.run()

    services = model_deployer.find_model_server(
        pipeline_name=deployment_pipeline_name,
        pipeline_step_name=deployer_step_name,
        model_name=model_name,
    )
    if services:
        service = cast(SeldonDeploymentService, services[0])
        if service.is_running:
            print(
                f"The Seldon prediction server is running remotely as a Kubernetes "
                f"service and accepts inference requests at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service, run "
                f"[italic green]`zenml served-models delete "
                f"{str(service.uuid)}`[/italic green]."
            )
        elif service.is_failed:
            print(
                f"The Seldon prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )

    else:
        print(
            "No Seldon prediction server is currently running. The deployment "
            "pipeline must run first to train a model and deploy it. Execute "
            "the same command with the `--deploy` argument to deploy a model."
        )


if __name__ == "__main__":
    os.system(
        "wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip"
    )
    os.system("unzip -qo dogImages.zip")
    os.system("rm dogImages.zip")
    main()
