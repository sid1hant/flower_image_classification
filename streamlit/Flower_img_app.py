import tensorflow as tf
import os
eff_model = tf.keras.models.load_model(os.path.join("streamlit/model/","eff_model.hdf5"))
MobileNet_model = tf.keras.models.load_model(os.path.join("streamlit/model/","MobileNet_model.hdf5"))
#res_model =  tf.keras.models.load_model('model\Res_model.h5',compile=False)
#DenseNet_model =  tf.keras.models.load_model('model\DenseNet_model.hdf5',compile=False)
#VGG_model =  tf.keras.models.load_model('model\VGG_model.h5',compile=False)

import streamlit as st
st.write("""
         # Flower Image Classification
         """
         )
st.write("This is a simple image classification web app to predict the name of the Flower")
file = st.file_uploader("Please upload an image file {.jpg| .jpeg}", type=["jpg","jpeg"])
choose_model = st.selectbox('Select a trained model:', ('MobileNet','EfficientNet'))

if choose_model == 'MobileNet':
    model = MobileNet_model
elif choose_model == 'EfficientNet':
    model = eff_model
    

    
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (244,244)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
    
        prediction = model.predict(image)
        
        return prediction

def validate_set(img):

    X_valid = []

        #image = ImageOps.grayscale(image)
        
    image = np.array(img)
    image_data_as_arr = np.asarray(image)
        
    X_valid.append(image_data_as_arr)
    X_valid = np.asarray(X_valid)   
    X_valid = tf.expand_dims(X_valid, axis=-1)
    return X_valid

label = {'garden phlox': 0, 'wild rose': 1, 'canna lily': 2, 'fire lily': 3, 'morning glory': 4, 'primula': 5, 'fritillary': 6, 'globe thistle': 7, 'canterbury bells': 8, 'wild pansy': 9, 'gazania': 10, 'king protea': 11, 'daffodil': 12, 'tree mallow': 13, 'red ginger': 14, 'petunia': 15, 'columbine': 16, 'silverbush': 17, 'tiger lily': 18, 'sweet pea': 19, 'barberton daisy': 20, 'alpine sea holly': 21, 'hard-leaved pocket orchid': 22, 'purple coneflower': 23, 'wild geranium': 24, 'spring crocus': 25, 'siam tulip': 26, 'bougainvillea': 27, 'bird of paradise': 28, 'cape flower': 29, 'hippeastrum ': 30, 'mallow': 31, 'water lily': 32, 'toad lily': 33, 'blanket flower': 34, 'pink quill': 35, 'lenten rose': 36, 'trumpet creeper': 37, 'artichoke': 38, 'pincushion flower': 39, 'sweet william': 40, 'orange dahlia': 41, 'hibiscus': 42, 'globe-flower': 43, 'tree poppy': 44, 'moon orchid': 45, 'cosmos': 46, 'snapdragon': 47, 'passion flower': 48, 'mexican petunia': 49, 'common dandelion': 50, 'yellow iris': 51, 'corn poppy': 52, 'marigold': 53, 'azalea': 54, 'clematis': 55, 'wallflower': 56, 'foxglove': 57, 'black-eyed susan': 58, 'stemless gentian': 59, 'geranium': 60, 'love in the mist': 61, 'buttercup': 62, 'pink-yellow dahlia': 63, 'daisy': 64, 'giant white arum lily': 65, 'japanese anemone': 66, 'cautleya spicata': 67, 'californian poppy': 68, 'spear thistle': 69, 'blackberry lily': 70, 'bishop of llandaff': 71, 'watercress': 72, 'carnation': 73, 'bolero deep blue': 74, 'monkshood': 75, 'desert-rose': 76, 'osteospermum': 77, 'grape hyacinth': 78, 'anthurium': 79, 'bee balm': 80, 'prince of wales feathers': 81, 'bromelia': 82, 'cyclamen ': 83, 'windflower': 84, 'lilac hibiscus': 85, 'ruby-lipped cattleya': 86, 'balloon flower': 87, "colt's foot": 88, 'great masterwort': 89, 'sunflower': 90, 'peruvian lily': 91, 'poinsettia': 92, 'frangipani': 93, 'lotus': 94, 'gaura': 95, 'rose': 96, 'sword lily': 97, 'thorn apple': 98, 'camellia': 99, 'magnolia': 100, 'iris': 101, 'common tulip': 102, 'pink primrose': 103}
from resizeimage import resizeimage
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    image = resizeimage.resize_cover(image, [224, 224])
    st.image(image, use_column_width=True)
    X_val = validate_set(image)


    y_pred = model.predict(X_val)
    prediction = np.argmax(y_pred,axis=1)

    keys = [k for k, v in label.items() if v == prediction]
    
    st.write(f"This is {keys[0]}")
