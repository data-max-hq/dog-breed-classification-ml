# Dog Breed Classification
#### A demo that uses a ResNet model to predict the breed of a dog given in a photo.

### *Prerequisites:*
- **Python 3.8.13**
- **Docker and Docker Compose**
- **k3d Cluster**
- **ZenML**


### *How to run the project with docker:*

1. Start by cloning the repository.
```bash
git clone https://github.com/data-max-hq/dog-breed-classification-ml.git
```
2. Open the project directory in terminal and type:
```bash
pip install -r requirements.txt
```
3. Train the model (only once):
```bash
python3 train_model.py
```
4. Build the docker images and run the containers.
```bash
docker compose up --build
```

<!-- # Model link -> https://drive.google.com/file/d/14vhLCEYqkYKIQJ3buY-bKJhjiUtGPJhX/view?usp=sharing -->