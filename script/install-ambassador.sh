helm repo add datawire https://www.getambassador.io
helm upgrade --install ambassador datawire/ambassador \
    --set image.repository=docker.io/datawire/ambassador \
    --values ./charts/ambassador/values.ambassador.local.yaml \
    --create-namespace \
    --namespace ambassador