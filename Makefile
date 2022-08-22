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

	  
delete:
	minikube delete --all

port-kubeflow:
	kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8081:80


load:
	minikube image load dogbreed:minikube
	minikube image load trainmodel:minikube
	minikube image load streamlit:minikube

build:
	docker build -t dogbreed:minikube .
	docker build -t trainmodel:minikube --file Dockerfile.train .
	docker build -t streamlit:minikube --file Dockerfile.streamlit .

helm:
	helmfile sync
