from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pathlib
import numpy as np

batch_size = 16

IMAGE_SHAPE = (224, 224, 3)

data_dir = pathlib.Path("dataset")
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Number of images:", image_count)

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
#20% val 80% Train
image_generator = ImageDataGenerator(rescale=1 / 255, validation_split=0.2,
                                     rotation_range=20,
                                     zoom_range=0.15,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.15,
                                     horizontal_flip=True,)

trainDataset = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size,
                                                     classes=list(CLASS_NAMES),
                                                     target_size=(224, 224),
                                                     shuffle=True, subset="training")

testDataset = image_generator.flow_from_directory(directory=str(data_dir), batch_size=batch_size,
                                                    classes=list(CLASS_NAMES),
                                                    target_size=(224, 224),
                                                    shuffle=True, subset="validation")

model = MobileNetV2(input_shape=(224, 224, 3))
model.summary()

#remove last layes
model.layers.pop()

# freeze all the weights of the model except the last 4 layers
for layer in model.layers[:-4]:
    layer.trainable = False

#O/p
output = Dense(2, activation="softmax")

# connect that dense layer to the model
output = output(model.layers[-1].output)
model = Model(inputs=model.inputs, outputs=output)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

checkpoint = ModelCheckpoint("model_mobilenet.h5",
                             save_best_only=True,
                             verbose=1)

# count number of steps per epoch
trainingStepsPerEpoch = np.ceil(trainDataset.samples / batch_size)
validationStepsPerEpoch = np.ceil(testDataset.samples / batch_size)

# train using the generators
history = model.fit_generator(trainDataset, steps_per_epoch=trainingStepsPerEpoch,
                    validation_data=testDataset, validation_steps=validationStepsPerEpoch,
                    epochs=10, verbose=1, callbacks=[checkpoint])

model.load_weights("model_mobilenet.h5")
validationStepsPerEpoch = np.ceil(testDataset.samples / batch_size)
evaluation = model.evaluate(testDataset)
print("Val loss:", evaluation[0])
print("Val Accuracy:", evaluation[1]*100)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()