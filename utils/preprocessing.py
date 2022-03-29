import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


current_path = Path(__file__).parent
face_xml = current_path / 'cat_face.xml'
cat_face = current_path.parent / 'datasets/images_background/0001/0001_000.JPG'

face_cascade = cv2.CascadeClassifier(face_xml.__str__())

img = cv2.imread(cat_face.__str__())

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5)
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

plt.imshow(img)
plt.show()
