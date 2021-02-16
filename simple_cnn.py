import os
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

train_datagen = ImageDataGenerator(rescale=1. / 255)
training_set = train_datagen.flow_from_directory('training_gallery',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='binary')
cnn = Sequential([
    Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(128, 128, 3)),
    MaxPool2D(pool_size=2, strides=2),
    Dropout(0.2),

    Conv2D(filters=64, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=2, strides=2),
    Dropout(0.2),

    Conv2D(filters=64, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=2, strides=2),
    Dropout(0.2),

    Flatten(),

    Dense(units=128, activation='relu'),
    Dropout(0.2),
    Dense(units=128, activation='relu'),
    Dropout(0.2),

    Dense(units=1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_path = 'checkpoints/best_simple_cnn.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    verbose=1,
    save_weights_only=False,
    save_freq='epoch',
    mode='auto',
    save_best_only=True)

early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

cnn.fit(x=training_set, validation_split=0.2, epochs=200, callbacks=[tensorboard_callback,
                                                                     checkpoint_callback,
                                                                     early_stop_callback])
