import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, concatenate, Activation, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from tensorflow.keras.optimizers import Adam
import h5py

IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CHANNELS = 12
NUM_CLASSES = 1

MODEL_PATH = 'h5_path_ASUS11G/unet_resnet_model.h5'

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def weighted_dice_loss(y_true, y_pred, class_weights={0: 1.0, 1: 1.0}):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    weights = y_true_f * class_weights[1] + (1 - y_true_f) * class_weights[0]
    intersection = K.sum(y_true_f * y_pred_f * weights)
    union = K.sum(y_true_f * weights) + K.sum(y_pred_f * weights)
    return 1.0 - (2. * intersection + K.epsilon()) / (union + K.epsilon())

def bce_weighted_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return bce(y_true, y_pred) + (1 - dice_coeff(y_true, y_pred))

def build_unet_resnet(input_shape, num_classes, backbone_name='ResNet50', freeze_backbone=False):
    inputs = Input(shape=input_shape)
    x_rgb_equivalent = Conv2D(3, (1, 1), padding='same', activation='relu')(inputs)
    x_rgb_equivalent = BatchNormalization()(x_rgb_equivalent)
    x_preprocessed_for_backbone = resnet_preprocess_input(x_rgb_equivalent * 255.0)

    if backbone_name == 'ResNet50':
        backbone = ResNet50(include_top=False, weights='imagenet')
    else:
        raise ValueError(f"Unsupported ResNet backbone name: {backbone_name}. Currently only ResNet50 is supported.")

    if freeze_backbone:
        for layer in backbone.layers:
            layer.trainable = False

    layer_names_for_skips = [
        'conv1_relu',
        'conv2_block3_out',
        'conv3_block4_out',
        'conv4_block6_out',
        'conv5_block3_out'
    ]
    encoder_submodel = Model(inputs=backbone.input, outputs=[backbone.get_layer(name).output for name in layer_names_for_skips])
    c1, c2, c3, c4, c5 = encoder_submodel(x_preprocessed_for_backbone)

    u6 = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.2)(c6)

    u7 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)

    u8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1)(c8)

    u9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)

    u10 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(c9)
    c10 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u10)
    c10 = BatchNormalization()(c10)
    c10 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c10)
    c10 = BatchNormalization()(c10)

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c10)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

model = None
try:
    model = build_unet_resnet(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=NUM_CLASSES)
    model.load_weights(MODEL_PATH)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=bce_weighted_dice_loss,
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES + 1, name='iou_score'), dice_coeff])
    print("Model architecture built and weights loaded successfully!")
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'L':
                image = image.convert('L')
            
            image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
            image_np = np.array(image_resized)
            input_image_final = np.stack([image_np] * NUM_CHANNELS, axis=-1)
            input_image_normalized = input_image_final.astype(np.float32) / 255.0
            input_image_batch = np.expand_dims(input_image_normalized, axis=0)

            prediction = model.predict(input_image_batch)
            predicted_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
            active_pixels_count = np.sum(predicted_mask == 255)
            total_pixels = IMG_HEIGHT * IMG_WIDTH
            active_pixels_ratio = float(active_pixels_count / total_pixels) if total_pixels > 0 else 0.0

            return jsonify({
                "message": "Prediction successful",
                "predicted_active_pixel_ratio": active_pixels_ratio,
            })

        except Exception as e:
            print(f"Error processing file: {e}")
            return jsonify({"error": f"Error processing file: {e}"}), 500
    
    return jsonify({"error": "Something went wrong"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)