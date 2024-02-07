
from face_detection.frontal_face_detector import Detector

model = Detector('.', './params.txt')
model.detect('./523000.jpg')
