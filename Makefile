all: build start load install-kubeflow 

start:
	minikube start --driver=docker --kubernetes-version=v1.21.6 

install-kubeflow:
	export PIPELINE_VERSION=1.8.2
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$$PIPELINE_VERSION"
	kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$$PIPELINE_VERSION"

port:
	kubectl port-forward svc/ambassador -n ambassador 8080:80

port-admin:
	kubectl port-forward svc/ambassador-admin -n ambassador 8877:8877


tf-deploy:
	kubectl apply -f ./tensorflow

delete:
	minikube delete --all

port-kubeflow:
	kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8081:80


install-emissary:
	helm repo add datawire https://app.getambassador.io
	helm repo update
	kubectl create namespace emissary && \
	kubectl apply -f https://app.getambassador.io/yaml/emissary/3.1.0/emissary-crds.yaml
	kubectl wait --timeout=90s --for=condition=available deployment emissary-apiext -n emissary-system
	helm install emissary-ingress --namespace emissary datawire/emissary-ingress && \
	kubectl -n emissary wait --for condition=available --timeout=90s deploy -lapp.kubernetes.io/instance=emissary-ingress


build-tfserve:
	docker pull tensorflow/serving
	docker tag tensorflow/serving tfserve:minikube
	docker build -t trainmodel:minikube --build-arg CONFIG="KUBERNETES" --file Dockerfile.train .
	docker build -t streamlit:minikube --build-arg CONFIG="TENSORFLOW" --file Dockerfile.streamlit .

load-tfserve:
	minikube image load tfserve:minikube
	minikube image load trainmodel:minikube
	minikube image load streamlit:minikube


load-seldon:
	minikube image load dogbreed:minikube
	minikube image load trainmodel:minikube
	minikube image load streamlit:minikube

build-seldon:
	docker build -t dogbreed:minikube .
	docker build -t trainmodel:minikube --build-arg CONFIG="KUBERNETES" --file Dockerfile.train .
	docker build -t streamlit:minikube --file Dockerfile.streamlit .

helm:
	helmfile sync

compose-seldon:
	docker compose up --build streamlit-seldon seldon

compose-tfserve:
	docker compose up --build streamlit-tfserve tfserve

local-train:
	python3 train_model.py

run:
	python3 kubeflow_pipeline.py