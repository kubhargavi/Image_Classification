# Image Classification

A simple image classification web app built using Streamlit and OpenCV DNN with a pretrained DenseNet-121 model.

##  Features
- Upload an image or provide an image URL  
- Classifies images into ImageNet categories  
- Displays predicted label with confidence  

##  Files
- image_classification.py – main app  
- DenseNet_121.prototxt – model architecture  
- DenseNet_121.caffemodel – model weights (not included)  
- classification_classes_ILSVRC2012.txt – Class labels 
- requirements.txt – project dependencies 

##  Setup
```bash
pip install -r requirements.txt
streamlit run image_classification.py
```
