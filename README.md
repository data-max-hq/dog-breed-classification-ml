# Dog Breed Classification 🐶
#### A demo that uses a ResNet model to predict the breed of a dog given in a photo.
#### Link to our blog post https://www.data-max.io/post/dog-breed-classification.

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


1. Open the project directory in terminal and type
    ```bash
    make requirements
    ```
2. Train the model (only once)
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
4. Open Streamlit UI at http://localhost:8502. Enjoy predicting 🪄

5. Stop docker containers
    ```bash
    docker compose down
    ```

## Run project with Kubernetes:
### *Prerequisites:*
- **Docker**
- **Helm**
- **Helmfile**
- **Minikube**

1. Deploy model using:
- TensorFlow Serving
    1. Create a kubernetes cluster (minikube)
        ```bash
        make start-tfserve
        ```
    2. Build images
        ```bash
        make build-tfserve
        ```
    3. Load images to minikube
        ```bash
        make load-tfserve
        ```
    4. Install Kubeflow
        ```bash
        make install-kubeflow
        #Wait till all pods of kubeflow are running
        ```
    5. Expose Kubeflow port so you can access Kubeflow dashboard at http://localhost:8080 (optional)
        ```bash
        make port-kubeflow
        ```
    6. Deploy TensorFlow Serving, Ambassador and Streamlit
        ```bash
        make helm-tfserve
        ```
    7. Apply mapping resources
        ```bash
        make deploy-tfserve
        ```
    8. Expose Emissary-ingress port
        ```bash
        make port-emissary
        ```
- Seldon Serving
    1. Create a kubernetes cluster (minikube)
    ```bash
    make start-seldon
    ```
    2. Build images
        ```bash
        make build-seldon
        ```
    3. Load images to minikube
        ```bash
        make load-seldon
        ```
    4. Install Kubeflow
        ```bash
        make install-kubeflow
        #Wait till all pods of kubeflow are running
        ```
    5. Expose Kubeflow port so you can access Kubeflow dashboard at http://localhost:8080 (optional)
        ```bash
        make port-kubeflow
        ```
    6. Deploy Seldon-Core, Ambassador and Streamlit
        ```bash
        make helm-seldon
        ```
    7. Deploy Seldon application and apply mapping resources
        ```bash
        make deploy-seldon
        ```
    8. Expose Emissary-ingress port
        ```bash
        make port
        ```
2. Open Streamlit UI at http://localhost:8080/streamlit. Enjoy predicting 🪄
3. Delete cluster
    ```bash
    make delete
    ```