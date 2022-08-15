# ZenML

## Setting up environment

### Install k3d
```bash
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
```

### Install zenml
```bash
pip install zenml=0.12.0
```

### Install ZenML integrations
```bash
zenml integration install kubeflow seldon -y
```

### Create a ZenML stack
```bash
zenml container-registry register local_registry --flavor=default --uri=localhost:5000 
zenml orchestrator register kubeflow_orchestrator --flavor=kubeflow
zenml stack register my_stack \
    -m default \
    -a default \
    -o kubeflow_orchestrator \
    -c local_registry
```

### Set the newly created stack as active
```bash
zenml stack set my_stack
```

### Start up stack
```bash
zenml stack up
```

### Install seldon
```bash
cd ..
make install-seldon-core
```

### Install ambassador
```bash
kubectl create namespace emissary && \
kubectl apply -f https://app.getambassador.io/yaml/emissary/3.1.0/emissary-crds.yaml && \
kubectl wait --timeout=90s --for=condition=available deployment emissary-apiext -n emissary-system
kubectl apply -f https://app.getambassador.io/yaml/emissary/3.1.0/emissary-emissaryns.yaml && \
kubectl -n emissary wait --for condition=available --timeout=90s deploy -lproduct=aes
```

### Create a model-deployer
```bash
zenml model-deployer register seldon_deployer --flavor=seldon \
  --kubernetes_context=<CLUSTER_NAME> \
  --kubernetes_namespace=kubeflow \
  --base_url=http://localhost:8080/seldon/kubeflow/zenml/api/v1.0/
 
zenml stack update my_stack --model_deployer=seldon_deployer
```
### Run pipeline
```bash
cd zenml
python run.py
```

