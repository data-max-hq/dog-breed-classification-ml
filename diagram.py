from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.onprem.container import Docker
from diagrams.onprem.network import Ambassador, Internet
from diagrams.onprem.workflow import KubeFlow
from diagrams.onprem.client import User, Client
from diagrams.k8s.ecosystem import Helm

# seldon = Custom('serve', 'resources/seldon.png')
# user = User('User')
# docker = Docker('docker')
# amb = Ambassador('ingress')
# kbfl = KubeFlow('orchestrator')
# helm = Helm('package manager')
# st = Custom('UI','resources/streamlit.png')

with Diagram("Local w/o K8s", "diagram1", "TB", "ortho", "svg"):

    user = User("user")

    with Cluster("training"):
        source = Internet("source")
        tf = Custom("trainer", "resources/tf.png")
        model = Custom("model", "resources/model.png")
    with Cluster("docker compose", "TB"):
        docker = Docker("docker")
        with Cluster("dashboard"):
            st = Custom("UI", "resources/streamlit.png")
            output = Client("output")
        with Cluster("model server"):
            seldon = Custom("serve", "resources/seldon.png")
        (
            st
            >> Edge(color="red", style="dashed")
            >> seldon
            >> Edge(color="red", style="dashed")
            >> st
        )
        docker >> Edge(color="blue") >> seldon
        docker >> Edge(color="blue") >> st

    (
        source
        >> Edge(color="orange", style="bold")
        >> tf
        >> Edge(color="orange", style="bold")
        >> model
        >> Edge(color="orange", style="bold")
        >> seldon
    )
    user >> Edge(color="green") >> st >> Edge(color="red") >> output
