# Dog Breed Classification
#### A demo that uses a ResNet model to predict the breed of a dog given in a photo.

### *Prerequisites:*
- **Python 3.8.13**
- **Docker and Docker Compose**



### *How to run the project with docker:*

1. Start by cloning the repository.
```bash
git clone https://github.com/data-max-hq/dog-breed-classification-ml.git
```
2. Open the project directory in terminal and type:
```bash
pip install -r requirements.txt
```
3. Build the docker images and run the containers.
```bash
docker compose up --build
```

### *Prerequisites:*
- **Python 3.8.13**
- **Docker**
- **Helm**
- **Helmfile**
- **Minikube**

1. Start by cloning the repository.
```bash
git clone https://github.com/data-max-hq/dog-breed-classification-ml.git
```
2. Open the project directory in terminal and type:
```bash
make all
```
3. Run the following command
```bash
python kubeflow_pipeline.py
```
4. Run the following command
```bash
make helm
```

