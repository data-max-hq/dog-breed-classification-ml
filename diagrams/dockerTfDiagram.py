from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.onprem.container import Docker
from diagrams.onprem.network import Ambassador, Internet
from diagrams.onprem.workflow import KubeFlow
from diagrams.onprem.client import User, Client
from diagrams.k8s.ecosystem import Helm

with Diagram(
    name="Local w/o K8s & TFserve",
    filename="output/TFdiagram",
    direction="TB",
    outformat=["pdf", "png"],
    curvestyle="ortho",
):

    user = User("user")
    output = Client("output")
    with Cluster("       training"):
        source = Internet("source")
        tf = Custom("trainer", "../resources/tf.png")
        model = Custom("model", "../resources/model.png")
    with Cluster("                     docker compose", "TB"):
        docker = Custom("", "../resources/compose.png")
        with Cluster("    dashboard"):
            st = Custom("UI", "../resources/streamlit.png")
        with Cluster("    model server"):
            seldon = Custom("TF serve", "../resources/tf.png")
        (
            st
            >> Edge(color="red", style="dashed")
            >> seldon
            >> Edge(color="red", style="dashed")
            >> st
        )
        docker >> Edge(color="#FF000000") >> seldon
        docker >> Edge(color="#FF000000") >> st

    (
        source
        >> Edge(color="orange", style="bold")
        >> tf
        >> Edge(color="orange", style="bold")
        >> model
        >> Edge(color="orange", style="bold")
        >> seldon
    )
    user >> Edge(color="#0F6B2A", style="bold") >> st >> Edge(color="red", style="bold") >> output