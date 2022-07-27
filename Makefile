all: minikube install-seldon-core  install-ambassador install-kubeflow build load ns-seldon apply

minikube:
	minikube start --driver=docker --kubernetes-version=v1.21.6 --cpus 4

install-kubeflow:
	./script/install-kubeflow.sh

install-ambassador:
	./script/install-ambassador.sh

port:
	kubectl port-forward svc/ambassador -n ambassador 8080:80

port-admin:
	kubectl port-forward svc/ambassador-admin -n ambassador 8877:8877

install-seldon-core:
	./script/install-seldon-core.sh 
	  
delete:
	minikube delete

port-kubeflow:
	kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8081:80

download-images:
	./script/download-dataset.sh

load:
	minikube image load dogbreed:minikube

build:
	docker build -t dogbreed:minikube .
	docker build -t dogbreed:minikube .
	docker build -t dogbreed:minikube .

apply:
	kubectl apply -f ./seldon/dogbreed.yaml
ns-seldon:
	kubectl create namespace seldon
	kubectl label namespace seldon serving.kubeflow.org/inferenceservice=enabled

helm:
	helmfile apply --concurrency 1	