helm upgrade --install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
	--create-namespace \
	--namespace seldon-system \
	--set ambassador.enabled=true \
	--set usageMetrics.enabled=false \
	--set replicaCount=1 \
	--set certMenager.enabled=false \
	--set enableAES=false 