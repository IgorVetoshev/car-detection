import logging
import time

import flask

import telebot
from telebot import types

import keras
import tensorflow as tf
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.backend import set_session

from random import randint

API_TOKEN = 'SOME_TOKEN'

WEBHOOK_HOST = 'SOME_HOST'
WEBHOOK_PORT = 443
WEBHOOK_LISTEN = '0.0.0.0' 

WEBHOOK_SSL_CERT = './webhook_cert.pem'  # Path to the ssl certificate
WEBHOOK_SSL_PRIV = './webhook_pkey.pem'  # Path to the ssl private key

WEBHOOK_URL_BASE = "https://%s:%s" % (WEBHOOK_HOST, WEBHOOK_PORT)
WEBHOOK_URL_PATH = "/%s/" % (API_TOKEN)

logger = telebot.logger
telebot.logger.setLevel(logging.INFO)

bot = telebot.TeleBot(API_TOKEN)
app = flask.Flask(__name__)


filepath = './uploads/image_'
model_path_car = './ft_model_car_or_not.h5'
model_path_total = './ft_model_damaged.h5'

sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

model_car = load_model(model_path_car)
model_total = load_model(model_path_total)


# Empty webserver index, return nothing, just http 200
@app.route('/', methods=['GET', 'HEAD'])
def index():
    return ''


# Process webhook calls
@app.route(WEBHOOK_URL_PATH, methods=['POST'])
def webhook():
    if flask.request.headers.get('content-type') == 'application/json':
        json_string = flask.request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return ''
    else:
        flask.abort(403)


# Handle '/start'
@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Hello! Please upload a photo of your car')


# Handle all other messages
@bot.message_handler(content_types=['photo'])
def send_text(message):
    
    file_info = bot.get_file(message.photo[0].file_id)
    downloaded_photo = bot.download_file(file_info.file_path)
    
    number = randint(0,10**6)
    image_path = filepath + str(number) + '.jpg'
    with open(image_path, 'wb') as file:
        file.write(downloaded_photo)
    
    img = load_img(image_path, target_size=(224, 224)) 
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)/255 
    
    global sess
    global graph
    
    with graph.as_default():
        set_session(sess)
        
        bot.send_message(message.chat.id, 'Validating the image...')
        
        pred_car = model_car.predict(x)
        
        if pred_car[0][0] <=.5:
            
            bot.send_message(message.chat.id, "Yes, I'm pretty sure it's a car on the image.") 
            
            pred_total = model_total.predict(x)

            if pred_total[0][0] <=.95:
                bot.send_message(message.chat.id, "Sorry, but your car cannot be insured. Please try another one.") 

            else:
                bot.send_message(message.chat.id, "Success! Your car is not damaged.")
                
        else:
            
            bot.send_message(message.chat.id, "It's not a car. Please use just photos with cars.")    


# Remove webhook, it fails sometimes the set if there is a previous webhook
bot.remove_webhook()

time.sleep(0.1)

# Set webhook
bot.set_webhook(url=WEBHOOK_URL_BASE + WEBHOOK_URL_PATH,
                certificate=open(WEBHOOK_SSL_CERT, 'r'))

# Start flask server
app.run(host=WEBHOOK_LISTEN,
        port=WEBHOOK_PORT,
        ssl_context=(WEBHOOK_SSL_CERT, WEBHOOK_SSL_PRIV),
        debug=True)