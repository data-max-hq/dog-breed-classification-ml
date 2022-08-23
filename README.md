# Dog Breed Classification 🐶
#### A demo that uses a ResNet model to predict the breed of a dog given in a photo.

## Geting Started
**There are two ways you can run this project:**
- **Docker**
- **Kubernetes**

## Start by cloning the repository
```bash
git clone https://github.com/data-max-hq/dog-breed-classification-ml.git
```
## Run project with docker:
### *Prerequisites*:
- **Docker**
- **Docker Compose**


1. Open the project directory in terminal and type:
    ```bash
    make requirements
    ```
2. Train the model (only once):
    ```bash
    make local-train
    ```
3. Deploy model using:
- TensorFlow Serving
    ```bash
    make compose-tfserve
    ```
- Seldon Serving
    ```bash
    make compose-seldon
    ```
4. Open streamlit UI at http://localhost:8502. Enjoy predicting 🪄

5. Stop docker containers:
    ```bash
    docker compose down
    ```

### Run project with Kubernetes
### *Prerequisites:*
- **Docker**
- **Helm**
- **Helmfile**
- **Minikube**

1. Create a kubernetes cluster (minikube)
    ```bash
    make start
    ```
2. Deploy model using:
- TensorFlow Serving
    1. Build images:
        ```bash
        make build-tfserve
        ```
    2. Load images to minikube
        ```bash
        make load-tfserve
        ```
- Seldon Serving
    1. Build images:
        ```bash
        make build-seldon
        ```
    2. Load images to minikube
        ```bash
        make load-seldon
        ```
3. Install kubeflow:
    ```bash
    make install-kubeflow
    #Wait till all pods of kubeflow are running
    ```
4. Train model:
    ```bash
    make run
    ```
5. Use the following command to set up port forwarding for the pipeline dashboard:
    ```bash
    make port-kubeflow
    ```
6. Access kubeflow dashboard at:
    ```bash
    http://localhost:8080/
    ```
7. Continue deployment using:
- TensorFlow Serving
    1. Install emissary ingress:
        ```bash
        make install-emissary
        ```
    2. Deploying Tensorflow Serving, Streamlit. Also creating two listeners and a mapping, respectively for tfserving, streamlit and emissary
        ```bash
        make tfserve-deploy
        ```
- Seldon Serving
    1. Deploy seldon service:
        ```bash
        kubectl apply -f ./seldon
        ```
    2. Deploy Seldon-Core, Ambassador and Streamlit:
        ```bash
        make helm
        #Wait till pods are running
        ```

8. Open streamlit UI at http://localhost:8502. Enjoy predicting 🪄

9. Delete cluster
    ```bash
    make delete
    ```


