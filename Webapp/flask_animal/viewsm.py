import numpy as np
import os
from flask_animal import app
from flask import render_template
from flask import request, redirect, url_for
import json
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array #, ImageDataGenerator
from collections import OrderedDict


#app = Flask(__name__)

#set flask variables
UPLOAD_FOLDER = os.path.join( os.getcwd(), 'flask_animal/static/')  #'uploads/' changed to 'static/'
ALLOWED_EXTENSIONS = set(['jpg'])

#is file extension allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
                          
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#load finetuned VGG16 CNN model
global json_model
JSON_PATH = 'VGG16model_finetune_ele_zeb_wilde_json_rev0_softmax.json'
with open (JSON_PATH, 'r') as json_file:
    loaded_model_json = json_file.read()

json_model = model_from_json(loaded_model_json)

#load weights into VGG16 CNN model
WEIGHT_PATH = 'VGG16_finetune_weights_and_bottleneck_features_ele_zeb_wilde_softmax.h5'
json_model.load_weights(WEIGHT_PATH)
graph = tf.get_default_graph()

          
#classify image with VGG16 CNN
def VGG16finetuneclassify(filename):
    # dimensions of our images.
    img_width, img_height = 150, 150
    
    img = load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(img_width, img_height))
    input_image = img_to_array(img)
    img.close()
    # the images are scaled during training so they need to be scaled for predictions too
    input_image = input_image/255.
    input_image = np.expand_dims(input_image, axis=0)
    with graph.as_default():
        prediction = json_model.predict(input_image)
    zeb_thresh = 0.65
    ele_thresh = 0.6
    wilde_thres = 0.6
    predict = OrderedDict()
    
    #make image classification
    predict = {'Elephant':round(prediction[0][0]*100,2), 'Wildebeest':round(prediction[0][1]*100,2), 'Zebra':round(prediction[0][2]*100,2)}
    
    if np.sort(prediction[0])[2]-np.sort(prediction[0])[1] <0.15:
        predict["Classification"] = 'Unable to classify with high confidence, possibly another class'
    else:
    
        if (prediction[0].max() == prediction[0][2]) & (prediction[0][2]>zeb_thresh):
            predict["Classification"] = 'Zebra'
        elif (prediction[0].max() == prediction[0][2]) & (prediction[0][2]<zeb_thresh):
            predict["Classification"] = 'Zebra or another class'

        if (prediction[0].max() == prediction[0][1]) & (prediction[0][1]>wilde_thres):
            predict["Classification"] = 'Wildebeest'
        elif (prediction[0].max() == prediction[0][1]) & (prediction[0][1]<wilde_thres):
            predict["Classification"] = 'Wildebeest or another class'

        if (prediction[0].max() == prediction[0][0]) & (prediction[0][0]>ele_thresh):
            predict["Classification"] = 'Elephant'
        elif (prediction[0].max() == prediction[0][0]) & (prediction[0][0]<ele_thresh):
            predict["Classification"] = 'Elephant or another class'
   
    return predict          


#classify multiple images with VGG16 CNN
def VGG16finetuneclassifymultiple(filenames):
    
    # dimensions of our images.
    img_width, img_height = 150, 150
    
    predlist =[]
    statistics = {}
    
    ele_count = 0
    maybe_ele = 0
    
    zeb_count = 0
    maybe_zeb=0
    
    wilde_count = 0
    maybe_wilde=0
    
    other_count = 0
    
    for filename in filenames:
        img = load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(img_width, img_height))
        input_image = img_to_array(img)
        img.close()
        # the images are scaled during training so they need to be scaled for predictions too
        input_image = input_image/255.
        input_image = np.expand_dims(input_image, axis=0)
        with graph.as_default():
            prediction = json_model.predict(input_image)
        zeb_thresh = 0.65
        ele_thresh = 0.6
        wilde_thres = 0.6
        
        predict = {}
        predict["Filename"] = filename

        #make image classification

        if np.sort(prediction[0])[2]-np.sort(prediction[0])[1] <0.15:
            predict["Classification"] = 'Unable to classify with high confidence, possibly another class'
            other_count = other_count+1
        else:

            if (prediction[0].max() == prediction[0][2]) & (prediction[0][2]>zeb_thresh):
                predict["Classification"] = 'Zebra'
                zeb_count = zeb_count+1
            elif (prediction[0].max() == prediction[0][2]) & (prediction[0][2]<zeb_thresh):
                predict["Classification"] = 'Zebra or another class'
                maybe_zeb = maybe_zeb+1

            if (prediction[0].max() == prediction[0][1]) & (prediction[0][1]>wilde_thres):
                predict["Classification"] = 'Wildebeest'
                wilde_count =wilde_count+1
            elif (prediction[0].max() == prediction[0][1]) & (prediction[0][1]<wilde_thres):
                predict["Classification"] = 'Wildebeest or another class'
                maybe_wilde=maybe_wilde+1

            if (prediction[0].max() == prediction[0][0]) & (prediction[0][0]>ele_thresh):
                predict["Classification"] = 'Elephant'
                ele_count=ele_count+1
            elif (prediction[0].max() == prediction[0][0]) & (prediction[0][0]<ele_thresh):
                predict["Classification"] = 'Elephant or another class'
                maybe_ele=maybe_ele+1

        predlist.append(predict)
    
    statistics["Elephant"] = ele_count
    statistics["MaybeElephant"] = maybe_ele
    statistics["Wildebeest"] = wilde_count
    statistics["MaybeWildebeest"] = maybe_wilde
    statistics["Zebra"] = zeb_count
    statistics["MaybeZebra"] = maybe_zeb
    statistics["Other"] = other_count
    totalfiles = len(filenames)
    return predlist, statistics, totalfiles
     
          
@app.route('/')
def index():
    return render_template("MultIndex5.html")  

@app.route('/upload', methods=['GET', 'POST'])   #multiple/single upload
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        
        uploaded_files = request.files.getlist("file[]")
        filenames = []
        
        if len(uploaded_files) ==0 or len(uploaded_files)>100:
            return render_template("MultError.html")
        
        for file in uploaded_files:
            if file.filename == '' :        
                return render_template("MultError.html")
        
        for file in uploaded_files:
            if allowed_file(file.filename) == False:        
                return render_template("MultError.html")
            
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                filenames.append(filename)
                
        if len(filenames) == 1: 
            predlist = VGG16finetuneclassify(filenames[0])
            return render_template("MultUpload3table.html", preddict = predlist, image_name=filename)
        
        if len(filenames) > 1:
            predlist, stats, totalfiles = VGG16finetuneclassifymultiple(filenames)
            return render_template("MultUpload4.html", preddict = predlist, statistics = stats, total=totalfiles)     
 
    
@app.route('/about')
def about():
    return render_template('MultAbout2.html')


@app.route('/auto')
def auto(filenameeg=None):
    
    #initialize variables
    filenameeg = request.args.get('filenameeg')

    #query for classification
    predlist = VGG16finetuneclassify(filenameeg)
    
    return render_template("MultUpload3table.html", preddict = predlist, image_name=filenameeg)
    

#examples
@app.route('/examples')
def examples():
    return render_template("MultExamples.html")    
    

