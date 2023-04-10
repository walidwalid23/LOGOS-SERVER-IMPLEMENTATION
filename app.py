from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import json
from math import sqrt
import requests
# %matplotlib inline
import requests
from io import BytesIO
import json
from flask_mail import Mail, Message
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os
import threading
import time
global embed

load_dotenv()

# uploaded logos folder
UPLOAD_FOLDER = 'uploads'
# allowed logos images extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# preparing the server
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# THE DATABASE NAME MUST BE ADDED IN THE URI BEFORE THE ? OR THE DB OBJECT WILL BE NONE
app.config['MONGO_URI'] = "mongodb+srv://walidwalid:"+os.getenv(
    "DB_PASS")+"@cluster0.2k8t7ew.mongodb.net/Logos_Features_DB?retryWrites=true&w=majority"
mongo_client = PyMongo(app)
extractedFeaturesCollec = mongo_client.db.extracted_features_col
# configuration of mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'stylebustersinc@gmail.com'
app.config['MAIL_PASSWORD'] = 'temlvwashllnqpbh'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)
# loading the feature extraction model
saved_model_path = 'feature_extractor/PRE TRAINED MOBILE NET/'
embed = hub.KerasLayer(saved_model_path)
# Class Responsible For CONVERTING IMAGES TO FEATURE VECTORS (EMBEDDINGS)


class TensorVector(object):
    def __init__(self, FileName=None):
        self.FileName = FileName

    def process(self):
        img = tf.io.read_file(self.FileName)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize_with_pad(img, 224, 224)
        img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        features = embed(img)
        feature_set = np.squeeze(features)
        return list(feature_set)
# function to find the cosine similarity between feature vectors


def cosineSim(a1, a2):
    sum = 0
    suma1 = 0
    sumb1 = 0
    for i, j in zip(a1, a2):
        suma1 += i * i
        sumb1 += j*j
        sum += i*j
    cosine_sim = sum / ((sqrt(suma1))*(sqrt(sumb1)))
    return cosine_sim


@app.route('/sendLogo', methods=['POST'])
def postRoute():
    print("in route here")
    userEmail = request.form['email']
    image = request.files['image']
    country = request.form['country']
    if image and allowed_file(image.filename):
        imageName = secure_filename(image.filename)
        # store the image in the uploads folder
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], imageName))
        # prepare a success response since we received the image from the user
        successResponse = jsonify(
            {"successMessage": "You will receive an email with the results shortly"})
    else:
        return jsonify(
            {"errorMessage": "Invalid File Type"})
    print("there")

    # run the below code after the Response is sent to user
   # @successResponse.call_on_close
    def on_close():
        print("in on close")
        # If the user does not select a file, the browser submits an empty file without a filename.
        if image.filename == '':
            return jsonify({"error": "no selected image"})

        if image and allowed_file(image.filename):
            imageName = secure_filename(image.filename)
            resultsFound = False
            # get image path
            logoImgPath = 'uploads/'+imageName
            # display the image
            # logoImage = cv2.imread(logoImgPath, cv2.IMREAD_ANYCOLOR)
            # cv2.imshow("main logo", logoImage)
            # cv2.waitKey(0)
            # getting feature vector of the main logo image
            helper = TensorVector(logoImgPath)
            vector1 = helper.process()
            print(len(vector1))
            # getting feature vector of the retrieved logos images and displaying matches

            URL = "https://logos-web-scraper.onrender.com/WalidLogosApi?country=" + country
            headers = {
                'Content-Type': 'application/json'
            }
            # send a request to get stream of logo json objects
            time.sleep(0.001)
            resp = requests.request(
                "GET", URL, headers=headers, stream=True)
            # print(resp.headers['content-type'])
            # print(resp.encoding)
            # we iterate by lines since we added new line after each response from server side
            for line in resp.iter_lines():
                if line:
                    # the remote hosts encodes chunks using utf-8 but localhost doesn't they use (https*)
                    decoded_chunk = line.decode('utf-8')
                # converting json to dict
                    decodedLogoObj = json.loads(decoded_chunk)

                    companyName = decodedLogoObj["companyName"]
                    logoImageUrl = decodedLogoObj["logoImageUrl"]
                    # print(decodedLogoObj)
                    # check if the logo Image path is hidden or not by the website (if not it starts with http)
                    if logoImageUrl[0] != "h":
                        # display default image
                        logoImageUrl = "uploads/hidden-logo.png"
                    # check if the features were already extracted and stored before or not
                    result = extractedFeaturesCollec.find_one(
                        {'logoImgPath': logoImageUrl})
                    if result == None:
                        # if the features weren't stored before
                        # extract features from each retrieved logo image
                        helper2 = TensorVector(logoImageUrl)
                        vector2 = helper2.process()
                        # store the features in the database
                        # https://stackoverflow.com/questions/1614236/in-python-how-do-i-convert-all-of-the-items-in-a-list-to-floats
                        extractedFeaturesCollec.insert_one({'logoImgPath': logoImageUrl,
                                                            'features': [float(i) for i in vector2]})
                    else:
                        # convert the features array back to float32
                        vector2 = np.array(result['features'], dtype="float32")

                    # get the cosine similarity
                    cosine_similarity = cosineSim(vector1, vector2)
                    # push to results list if similarity is above a certain limit
                    print(" Cosine Similarity of The Main Image and Image: " +
                          companyName + " is: " + str(round(cosine_similarity*100))+"%")
                    if cosine_similarity > 0.90:
                        resultsFound = True
                        # send email including the details of the matched logo
                        msg = Message('Found A Match!',
                                      sender='stylebustersinc@gmail.com',
                                      recipients=[userEmail]
                                      )
                        msg.body = 'We found a matched logo! \n Company Name is: ' + \
                            companyName+'\n Company Logo URL Is: '+logoImageUrl
                        # send the email to the user (you must put the mail.send inside the app context)
                        with app.app_context():
                            mail.send(msg)
            if resultsFound == False:
                # Send an email telling the user that no results were found
                msg = Message('No Results Found',
                              sender='stylebustersinc@gmail.com',
                              recipients=[userEmail]
                              )
                msg.body = 'No Matches Has Been Found'
                # send the email to the user (you must put the mail.send inside the app context)
                with app.app_context():
                    mail.send(msg)

        # display image
        # urllib.request.urlretrieve(logoPath, "matchedlogo")
        # img = Image.open("matchedlogo")
        # img.show()
    thread = threading.Thread(target=on_close)
    thread.start()
    return successResponse


def get_stream(url):
    session = requests.Session()
    resp = session.get(url, stream=True)

    for line in resp.iter_content(chunk_size=1280):
        if line:
            decoded_line = line.decode('utf-8')
            print(json.loads(decoded_line))


# check if this file is being excuted from itself or being excuted by being imported as a module
if __name__ == "__main__":
    from waitress import serve
    print("server is running at port "+str(os.getenv("PORT")))
    serve(app, host="0.0.0.0", port=os.getenv("PORT"))
