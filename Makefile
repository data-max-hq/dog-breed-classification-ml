minikube:
	minikube start --driver=docker --kubernetes-version=v1.21.6

kubeflow:
	export PIPELINE_VERSION=1.8.2
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
	kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
	kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"

ambassador:
	helm repo add datawire https://www.getambassador.io
	helm upgrade --install ambassador datawire/ambassador \
      --set image.repository=docker.io/datawire/ambassador \
      --create-namespace \
      --namespace ambassador \
	  --set service.type=ClusterIP 

port-admin:
	kubectl port-forward svc/ambassador-admin -n ambassador 8877:8877

seldon-core:
	helm upgrade --install seldon-core seldon-core-operator \
      --repo https://storage.googleapis.com/seldon-charts \
	  --create-namespace \
	  --namespace seldon-system \
	  --set ambassador.enabled=true \
	  --set usageMetrics.enabled=false \
	  --set replicaCount=1 \
	  --set certMenager.enabled=false \
	  --set enableAES=false 
	  
delete:
	minikube delete