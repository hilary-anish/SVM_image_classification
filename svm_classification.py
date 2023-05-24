# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:26:20 2023

@author: ANISH HILARY
"""

from glob import glob
import PIL
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import time
import pickle

train_dir = glob('./dataset/train/*')
ext = '/*.jpg'
# test_dir = glob('./dataset/test*')


intel_class = ['forest','mountain','street']
dataset = []


for label_dir in train_dir:
    if label_dir.split('\\')[-1] in intel_class:
        label = label_dir.split('\\')[-1]
        for im in glob(f"{label_dir}{ext}"):
            with PIL.Image.open(im) as img:
                try:
                    img = img.resize((100,100))
                    img = np.array(img)
                    
                    if img.shape[2]==1:
                        img = np.cat((img,img,img),dim=0)
                    if img.shape[2]==4:
                        img = img[:,:,:3]
    
                    img = img.flatten()
                    
                    dataset.append([img, intel_class.index(label)])
                    
                except Exception as e:
                    pass
    
random.shuffle(dataset)
features = []
label = []

for data in dataset:
    img, lab = data
    features.append(img)
    label.append(lab)
    
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.15,
                                                    random_state=42)

classifier = SVC(C=2, kernel='poly', gamma='auto')

start = time.time()
classifier.fit(X_train, y_train)
stop = time.time()

with open('svm_classfier_3.pkl','wb') as f:
    pickle.dump(classifier, f)
    
time_taken = stop-start
print('Time taken: ', time_taken)

prediction = classifier.predict(X_test)

accuracy = classifier.score(X_test, y_test)

print('Accuracy: ', accuracy)
print('Prediction: ', intel_class[prediction[0]])


shoe = X_test[0].reshape(100,100,3)

plt.imshow(shoe)
plt.show()
    
    
    
    