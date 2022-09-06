from kafka import KafkaProducer
import cv2
producer=KafkaProducer(bootstrap_servers=['localhost:9092'])
image = cv2.imread("./dogImages/test/078.Great_dane/Great_dane_05322.jpg")
ret, buffer = cv2.imencode('.jpg', image)
producer.send("dogtopic",buffer.tobytes())
producer.flush()