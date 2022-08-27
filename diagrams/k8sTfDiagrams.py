from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.onprem.container import Docker
from diagrams.onprem.network import Ambassador, Internet
from diagrams.onprem.workflow import KubeFlow
from diagrams.onprem.client import User, Client
from diagrams.k8s.ecosystem import Helm
from diagrams.onprem.network import Ambassador


with Diagram("kubernetes with TFserve", "output/k8sTFdiagram", "TB", "ortho", ["pdf", "png"]):

    user = User("user")
    ui = Client("Streamlit UI")
    output = Client("Kubeflow UI")
    with Cluster("minikube"):
        with Cluster("Kubeflow Pipeline", "TB"):
            kubeflow = KubeFlow()
            with Cluster("Model training", "TB"):
                source = Internet("source")
                tf = Custom("trainer", "../resources/tf.png")
        model = Custom("model", "../resources/model.png")
        kubeflow >> Edge(color = "#FF000000" ) >> source
        source >> Edge(color="Black") >> tf >> Edge(color="#F8BF3C") >> model
        with Cluster("    ."):
            st = Custom("Streamlit", "../resources/streamlit.png")
        with Cluster("TensorFlow serve"):
            seldon = Custom("model serve", "../resources/tf.png")
        with Cluster(""):
            ambassador = Ambassador("Ambassador")
    source >> Edge(color="#FF000000") >> model
    user >> Edge(color="#FF000000") >> model
    kubeflow >> Edge(color="Black", style="bold") >> output
    ambassador >> Edge(color="Black", style="bold") >> st
    ambassador >> Edge(color="Black", style="bold") >> seldon
    st >> Edge(color="Red", style="bold") >>  ambassador
    model >> Edge(color="#5F1E00", style="bold") >> st
    model >> Edge(color="#5F1E00", style="bold") >> seldon
    seldon >> Edge(color="#F8BF3C", style="bold") >>  ambassador
    user >> Edge(color="Black", style="bold") >> ui >> Edge(color = "Blue", style="bold") >> st >> Edge(color ="#0F6B2A", style="bold") >> ui