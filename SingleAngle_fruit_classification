# In[1]
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import glob
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
# In[2]

fruit_images = []
labels = [] 
for fruit_dir_path in glob.glob("/Users/biubiubiu/Downloads/Fruit-Images-Revised/Training/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        fruit_images.append(image)
        labels.append(fruit_label)
        
###array of fruit images  
fruit_images = np.array(fruit_images)
labels = np.array(labels)

# In[3]
####creating labels for images
label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

# In[4]
id_to_label_dict

# In[5]
label_ids = np.array([label_to_id_dict[x] for x in labels])

# In[6]
###Visualize untouched images
def plot_image_grid(images, rows, columns):
    figure = plt.figure(figsize=(columns * 3, rows * 3))
    for i in range(columns * rows):
        figure.add_subplot(rows, columns, i + 1)
        plt.imshow(images[i])
        
    plt.show()


# In[7]          
plot_image_grid(fruit_images[0:100], 10, 10)


# In[8]
scaler = StandardScaler()

# In[9]
images_scaled = scaler.fit_transform([i.flatten() for i in fruit_images])


# In[10]

pca = PCA(n_components=50)
pca_result = pca.fit_transform(images_scaled)

# In[11]
tsne = TSNE(n_components=2, perplexity=40.0)
tsne_result = tsne.fit_transform(pca_result)
tsne_result_scaled = StandardScaler().fit_transform(tsne_result)


# In[12]
def visualize_scatter(data_2d, label_ids, figsize=(20,20)):
    plt.figure(figsize=figsize)
    plt.grid()
    
    nb_classes = len(np.unique(label_ids))
    
    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color= plt.cm.Set1(label_id / float(nb_classes)),
                    linewidth='1',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    plt.legend(loc='best')
    

# In[13]
def visualize_scatter_with_images(data_2d, images, figsize=(45,45), image_zoom=1):
    fig, ax = plt.subplots(figsize=figsize)
    plt.grid()
    artists = []
    for xy, i in zip(data_2d, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(data_2d)
    ax.autoscale()
    plt.show()
    

# In[14]
visualize_scatter(tsne_result_scaled, label_ids, figsize=(25, 25))

# In[15]
visualize_scatter_with_images(tsne_result_scaled, fruit_images, image_zoom=0.5, figsize=(25, 25))

# In[16]
X_train, X_test, y_train, y_test = train_test_split(pca_result, label_ids, test_size=0.25, random_state=42)

# In[17]
#Train Random Forest Classifier¶
forest = RandomForestClassifier(n_estimators=10)
forest = forest.fit(X_train, y_train)

# In[18]
test_predictions = forest.predict(X_test)

# In[19]
precision = accuracy_score(test_predictions, y_test) * 100
print("Accuracy with RandomForest: {0:.6f}".format(precision))

# In[20]
#Train SVM¶
svm_clf = svm.SVC()
svm_clf = svm_clf.fit(X_train, y_train) 

# In[21]
test_predictions = svm_clf.predict(X_test)
# In[22]
precision = accuracy_score(test_predictions, y_test) * 100
print("Accuracy with SVM: {0:.6f}".format(precision))

# In[23]
#Train Decision Tree
tree = DecisionTreeClassifier()
tree = tree.fit(X_train,y_train)

# In[24]
test_predictions = tree.predict(X_test)
# In[25]
precision = accuracy_score(test_predictions, y_test) * 100
print("Accuracy with DecisionTree: {0:.6f}".format(precision))
# In[26]
#Train KNN
neigh = KNeighborsClassifier(n_neighbors=1)
neigh = neigh.fit(X_train, y_train)

# In[27]
test_predictions = neigh.predict(X_test)

# In[28]
precision = accuracy_score(test_predictions, y_test) * 100
print("Accuracy with KNN: {0:.6f}".format(precision))

# In[29]
#Validate the models on the Validation Data¶
validation_fruit_images = []
validation_labels = [] 
for fruit_dir_path in glob.glob("/Users/biubiubiu/Downloads/Fruit-Images-Revised/Validation/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        validation_fruit_images.append(image)
        validation_labels.append(fruit_label)
validation_fruit_images = np.array(validation_fruit_images)
validation_labels = np.array(validation_labels)
# In[30]
validation_label_ids = np.array([label_to_id_dict[x] for x in validation_labels])

# In[31]
validation_images_scaled = scaler.transform([i.flatten() for i in validation_fruit_images])
# In[32]
validation_pca_result = pca.transform(validation_images_scaled)
# In[33]
#Random Forest
test_predictions = forest.predict(validation_pca_result)
# In[34]
precision = accuracy_score(test_predictions, validation_label_ids) * 100
print("Validation Accuracy with Random Forest: {0:.6f}".format(precision))
# In[35]
#SVM
test_predictions = svm_clf.predict(validation_pca_result)
# In[36]
precision = accuracy_score(test_predictions, validation_label_ids) * 100
print("Validation Accuracy with SVM: {0:.6f}".format(precision))
# In[37]
#DecisionTree
test_predictions = tree.predict(validation_pca_result)
# In[38]
precision = accuracy_score(test_predictions, validation_label_ids) * 100
print("Validation Accuracy with DecisionTree: {0:.6f}".format(precision))
# In[39]
#KNN
test_predictions = neigh.predict(validation_pca_result)
# In[40]
precision = accuracy_score(test_predictions, validation_label_ids) * 100
print("Validation Accuracy with KNN: {0:.6f}".format(precision))

