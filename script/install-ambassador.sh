helm repo add datawire https://www.getambassador.io
helm upgrade --install ambassador datawire/ambassador \
    --set image.repository=docker.io/datawire/ambassador \
    --create-namespace \
    --namespace ambassador \
	--set service.type=ClusterIP 
