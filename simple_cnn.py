import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard

train_datagen = ImageDataGenerator(rescale=1. / 255)
training_set = train_datagen.flow_from_directory('training_gallery',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='binary')
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('testing_gallery',
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

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

cnn.fit(x=training_set, validation_data=test_set, epochs=50, callbacks=[tensorboard_callback])

cnn_json_structure = cnn.to_json()
with open("simple_cnn.json", "w") as json_file:
    json_file.write(cnn_json_structure)
cnn.save_weights("simple_cnn.h5")
