helm upgrade --install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
	--values ./charts/seldon-core/values.local.yaml \
	--create-namespace \
	--namespace seldon-system