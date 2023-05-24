# -*- coding: utf-8 -*-
"""
Created on Tue May 23 21:28:54 2023

@author: ANISH HILARY
"""


import pickle
from glob import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt


test_dir = glob('./dataset/test/mountain/*.jpg')
intel_class = ['forest','mountain','street']

with open('svm_classfier_3.pkl','rb') as f:
    svm_model = pickle.load(f)

tp = 0
fp = 0

for im in test_dir:
    with Image.open(im) as img:
        try:
            img_n = img.resize((100,100))
            img_n = np.array(img_n)
            
            if img_n.shape[2]==1:
                img_n = np.cat((img_n,img_n,img_n),dim=0)
            if img_n.shape[2]==4:
                img_n = img_n[:,:,:3]

            img_n = img_n.flatten()
            
        except Exception as e:
            pass
        img_n = img_n.reshape(1,-1)
        prediction = svm_model.predict(img_n)
        
        print('Prediction: ', intel_class[prediction[0]])
        
        if prediction[0] == 1:
            tp+=1
        else:
            fp+=1
    

        # Add text to the image
        text = intel_class[prediction[0]]
        text_position = (20, 20)  # top-left corner coordinates
        
        font = ImageFont.truetype("arial.ttf", 20)
        draw = ImageDraw.Draw(img)
        draw.text(text_position, text, font=font, fill="purple")
        
        # Show the image with text
        plt.imshow(img)
        plt.show(img)
        
        
precision = tp/(tp+fp)
recall = tp/len(test_dir)