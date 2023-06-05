# Car Brand Classifier model using Transfer Learning (Resnet 50)+Streamlit
This project is based on Transfer learning technique to classify car brands among Audi , Mercedes and Lamborghini. 

A user can also make his own dataset using the automated script https://github.com/kartikk61/BrandNameClassifier/blob/main/imagepygoogledownloadscript.py

User may need to download the library using

    $ pip install google_images_download

## Notebook
The *carbrand.ipynb* notebook contains the model training and testing of the model . 

Model is using the "imagenet" weights . 


## Website Screenshot
![Demo](https://github.com/kartikk61/BrandNameClassifier/blob/main/Screenshot%20from%202023-06-05%2021-34-10.png?raw=true "Screenshot Website")

## Terminal run of streamlit
![Streamlit ss](https://github.com/kartikk61/BrandNameClassifier/blob/main/Screenshot%20from%202023-06-05%2021-26-05.png?raw=true)

## Model saving 

Save the model using the *.h5* format

## Streamlit script
Streamlit script in *application.py* file

Run using

    $ streamlit run application.py 

## Reminder
DO NOT FORGET TO MAKE *Procfile*

