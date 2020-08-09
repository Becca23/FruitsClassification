import glob
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import svm

classification = []
data = []
# Load in the Training data
for dir in glob.glob('fruits-360/Training/*'):
    # Choose random image from each directory
    name = random.choice(glob.glob(dir+"/*.jpg"))
    img = cv2.imread(name)
    # Invert the colors from BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize so there is enough memory
    img = cv2.resize(img, (50, 50))
    data.append(img)
    # Split the path string so it can be used in the
    name = name.lstrip("fruits-360/Training\ ")
    name = name.split("\\")[0]
    classification.append(name)
data = np.array(data)

# Load in the test data
test_classification = []
test_data = []
for name in glob.glob('fruits-360/Test/*/*.jpg'):
    # Choose random image from each directory
    img = cv2.imread(name)
    # Invert the colors from BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize so there is enough memory
    img = cv2.resize(img, (50, 50))
    test_data.append(img)
    # Split the path string so it can be used in the
    name = name.lstrip("fruits-360/Test\ ")
    name = name.split("\\")[0]
    test_classification.append(name)
test_data = np.array(test_data)

# Standardize the features and perform PCA to reduce the dimensionality of the training set
scaler = StandardScaler()
images_scaled = scaler.fit_transform([i.flatten() for i in data])
test_images_scaled = scaler.fit_transform([i.flatten() for i in test_data])
# Encode every class as a number so we can incorporate class labels into plot later
le = LabelEncoder()
y = le.fit_transform(classification)

# Perform Linear Discriminant Analysis on the training data
#lda = LinearDiscriminantAnalysis()
#X_lda = lda.fit_transform(images_scaled, y)

#print(lda.explained_variance_ratio_)

# As we had already determined during the first question, pca is the best way to reduce the dimensionality of the data
#pca = PCA(n_components=50)
#X_pca = pca.fit_transform(images_scaled, y)
#print(pca.explained_variance_ratio_)

# Given that we only have 1 image per class we cannot perform dim reduction on the dataset as the number of samples
# must be more than the number of classes

# Due to the fact we are only using one angle per class we will not be splitting the train data into a train/test
# split as it would not contain all the classes and would therefore be inaccurate
# Train random forest classifier
forest = RandomForestClassifier(n_estimators=10)
forest = forest.fit(images_scaled, classification)

# Test how well the random forest model generalizes to unseen data
test_predictions = forest.predict(test_images_scaled)
precision = accuracy_score(test_predictions, test_classification) * 100
print("Test Accuracy with Random Forest: {0:.2f}%".format(precision))

# Train SVM model
svm_clf = svm.SVC()
svm_clf = svm_clf.fit(images_scaled, classification)

# Test how well svm model generalizes to unseen data
test_predictions = svm_clf.predict(test_images_scaled)
precision = accuracy_score(test_predictions, test_classification) * 100
print("Accuracy with SVM: {0:.2f}%".format(precision))

# Convolution Neural Networks
# Reshpae the test and train sets to something that can be used for CNN
