import cv2
import matplotlib.pyplot as plt

img = cv2.imread(
    "C:/Users/rbren/OneDrive - Northeastern University/DS5220/Project-1/Data/fruits-360/Training/Apple Braeburn/0_100.jpg")
print(type(img))
print(img.shape)
print('\n')

# Switch the color scheme since CV2 colors are BGR instead of RGB (otherwise our apples are blue)
new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(new_img)
plt.show()
