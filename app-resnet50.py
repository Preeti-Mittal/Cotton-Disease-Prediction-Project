#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[3]:


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# In[5]:


os.chdir("E:\Data Science Project\Cotton Disease Prediction")


# In[6]:


app = Flask(__name__)


# In[7]:


#Model saved with Keras model.save()
MODEL_PATH ='model_resnet50.h5'


# In[8]:


# Load your trained model
model = load_model(MODEL_PATH)


# In[11]:


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)
    
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds==0:
        preds="The leaf is diseased cotton leaf"
    elif preds==1:
        preds="The leaf is diseased cotton plant"
    elif preds==2:
        preds="The leaf is fresh cotton leaf"
    else:
        preds="The leaf is fresh cotton plant"
        
    return preds


# In[13]:


@app.route('/', methods=["GET"])

def index():
    return render_template('index.html')


# In[14]:


@app.route('/predict', methods=["GET", "POST"])

def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__) #save the file to ./uploads
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        #Make Prediction
        preds = model.prediction(file_path, model)
        result = preds
        return result
    return None


# In[16]:


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




