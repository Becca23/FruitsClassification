import glob
import cv2
from string import digits
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D
from keras.utils import to_categorical
from sklearn.decomposition import PCA

train_data = []
train_class = []
class_list = []
for name in glob.glob('fruits-360/Training/*/*.jpg'):
    # Choose random image from each directory
    img = cv2.imread(name)
    # Invert the colors from BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize so there is enough memory
    img = cv2.resize(img, (50, 50))
    train_data.append(img)
    # Split the path string so it can be used in the
    name = name.split("\\")[1]
    # Multi class images do not differentiate between "Apple Red 1" and "Apple Red 2",
    # They just have "apple" so classify training data the same way
    name = name.split(" ")[0]
    train_class.append(name)
    if [name] not in class_list:
        class_list.append([name])
"""
test_data = []
test_class = []

for name in glob.glob('fruits-360/Test/*/*.jpg'):
    # Choose random image from each directory
    img = cv2.imread(name)
    # Invert the colors from BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize so there is enough memory
    img = cv2.resize(img, (50, 50))
    test_data.append(img)
    # Split the path string so it can be used in the
    name = name.split("\\")[1]
    # Multi class images do not differentiate between "Apple Red 1" and "Apple Red 2",
    # They just have "apple" so classify training data the same way
    name = name.split(" ")[0]
    test_class.append(name)
"""

multi_data = []
multi_class = []
remove_digits = str.maketrans("","",digits)
for name in glob.glob('fruits-360/test-multiple_fruits/*.jpg'):
    # Choose random image from each directory
    img = cv2.imread(name)
    # Invert the colors from BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize so there is enough memory
    img = cv2.resize(img, (50, 50))
    multi_data.append(img)
    # Split the path string so it can be used in the
    name = name.split("\\")[1][:-4]
    name = name.translate(remove_digits).split("_")
    for i in range(len(name)):
         name[i] = name[i].capitalize()
    multi_class.append(name)

# Turn the image arrays into np arrays for use with specific methods
train_data = np.array(train_data)
# test_data = np.array(test_data)
multi_data = np.array(multi_data)

# scale all the images
scaler = StandardScaler()
train_scaled = scaler.fit_transform([i.flatten() for i in train_data])
# test_scaled = scaler.fit_transform([i.flatten() for i in test_data])
multi_scaled = scaler.fit_transform([i.flatten() for i in multi_data])

pca = PCA(n_components=50)
train_scaled = pca.fit_transform(train_scaled)

# CNN
X_train = train_data.reshape(train_data.shape[0], 50, 50, 3)
# X_test = test_data.reshape(test_data.shape[0], 50, 50, 3)
X_multi = multi_data.reshape(multi_data.shape[0], 50, 50, 3)

# One hot encode the test and train
le = LabelEncoder()
y_train = le.fit_transform(train_class)
# y_test = le.fit_transform(test_class)

mlb = MultiLabelBinarizer()
mlb.fit(class_list)
y_multi = mlb.transform(multi_class)

y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

input_shape = (50, 50, 3)

model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))

model.add(AveragePooling2D())

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(AveragePooling2D())

model.add(Flatten())

model.add(Dense(len(class_list), activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, verbose=0)
model.fit(X_train, y_train, epochs=3, verbose=0)

"""
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss: ", score[0])
print("Test Accuracy: ", score[1])
"""

score = model.evaluate(X_multi, y_multi, verbose=0)
print("Test loss: ", score[0])
print("Test Accuracy: ", score[1])

predictions = model.predict(X_multi)
# show the inputs and predicted outputs
labels = (predictions > 0.5).astype(np.int)
print(labels)
print(mlb.inverse_transform(labels))
