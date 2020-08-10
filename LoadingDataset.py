import glob
import cv2
import matplotlib.pyplot as plt

classification = []
data = []
for name in glob.glob('fruits-360/Training/*/*.jpg'):
    print(name)
    img = cv2.imread(name)
    # Invert the colors from BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize so there is enough memory
    img = cv2.resize(img, (50, 50))
    data.append(img)
    # Split the path string so it can be used in the
    name = name.lstrip("fruits-360/Training\ ")
    name = name.split("\r")[0]
    classification.append(name)

img = data[45]
plt.imshow(img)
plt.show()

