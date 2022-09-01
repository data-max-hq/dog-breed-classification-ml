from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.onprem.network import Internet
from diagrams.onprem.client import User, Client

with Diagram(
    name="Docker Compose with TensorFlow Serving",
    filename="output/TFdiagram",
    direction="TB",
    outformat=["pdf", "png"],
    curvestyle="ortho",
):

    user = User("user")
    output = Client("Streamlit UI")
    with Cluster("       training"):
        source = Internet("source")
        tf = Custom("TensorFlow", "../resources/tf.png")
        model = Custom("model", "../resources/model.png")
    with Cluster("                     docker compose", "TB"):
        docker = Custom("", "../resources/compose.png")
        with Cluster("    dashboard"):
            st = Custom("Streamlit", "../resources/streamlit.png")
        with Cluster("    model server"):
            seldon = Custom("TensorFlow Serving", "../resources/tf.png")
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
    (
        user
        >> Edge(color="#0F6B2A", style="bold")
        >> st
        >> Edge(color="red", style="bold")
        >> output
    )
