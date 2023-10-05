import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.metrics import classification_report

def get_data():
    train_data = "Data/train"
    val_data = "Data/val"
    return train_data, val_data

def get_preprocessed_data(train_data, val_data):
    image_size = 224
    batch_size = 20
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255,
        shear_range= 0.2,
        horizontal_flip = True
    )
    test_val_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    train_generator = train_datagen.flow_from_directory(
        directory = train_data,
        target_size=(image_size, image_size),
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True
    )
    val_generator = test_val_datagen.flow_from_directory(
        directory = val_data,
        target_size=(image_size, image_size),
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False
    )
    return image_size, train_generator, val_generator

def create_model(image_size, train_ds, val_ds):
    print("******************************************************************************************")
    print("")
    print("Getting the model")
    print("******************************************************************************************")
    print("")
    net = keras.applications.mobilenet_v2.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3)
    )
    net.summary()   
    for layer in net.layers[:-2]:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(net.output)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    prediction = layers.Dense(2, activation="softmax")(x)
    model = models.Model(inputs = net.input, outputs = prediction)
    model.summary()

    model.compile(
        loss = "categorical_crossentropy",
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        metrics = ['acc']
    )
    print("******************************************************************************************")
    print("")
    print("Training")
    print("******************************************************************************************")
    print("")
    history = model.fit(
        train_ds,
        epochs = 10,
        validation_data=val_ds
    )
    print("******************************************************************************************")
    print("")
    print("Evaluating")
    print("******************************************************************************************")
    print("")
    scores = model.evaluate(val_ds, verbose = 1)
    print(f"Loss : {scores[0]} Accuracy:{scores[1]}")
    print("******************************************************************************************")
    print("Testing")
    print("******************************************************************************************")
    print("")
    test_pred = model.predict(val_ds, verbose = 1)
    test_labels = np.argmax(test_pred, axis=1)
    print("******************************************************************************************")
    print("")
    print("Prediction Report")
    print("******************************************************************************************")
    print("")
    class_labels = val_ds.class_indices
    class_labels = {v:k for k,v in class_labels.items()}
    classes = list(class_labels.values())
    print(classes)
    print('Classification Report')
    print(classification_report(val_ds.classes, test_labels, target_names=classes))
    return model

def main():
    train_data, val_data = get_data()
    image_size, train_ds, val_ds = get_preprocessed_data(train_data, val_data)
    model = create_model(image_size, train_ds, val_ds)
    model.save("model/BellPepperMobileNet.h5")

if __name__ == "__main__":
    main()