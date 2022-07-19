minikube:
	minikube start --driver=docker --kubernetes-version=v1.21.6

install-kubeflow:
	./script/install-kubeflow.sh

install-ambassador:
	./script/install-ambassador.sh

port:
  	kubectl port-forward svc/ambassador-admin -n ambassador 8080:8080

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