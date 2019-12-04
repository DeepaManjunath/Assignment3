# Assignment3
Have used Separable Convolution 2D 32x32 with padding in the beginning
model = Sequential()

model.add(SeparableConv2D(95,kernel_size=(3, 3),activation='relu',padding='same',input_shape=(32, 32, 3)))#RF=1
model.add(BatchNormalization(momentum=0.011))
model.add(Dropout(0.1622))

model.add(SeparableConv2D(95,kernel_size=(3,3),padding='same',activation='relu'))#32 #RF=3x3
model.add(BatchNormalization(momentum=0.011))
model.add(Dropout(0.1622))
model.add(SeparableConv2D(95,kernel_size=(3,3),strides=2,padding='same',activation='relu'))#32 #RF=5x5
model.add(BatchNormalization(momentum=0.011))
model.add(Dropout(0.1622))

model.add(SeparableConv2D(95,kernel_size=(3,3),padding='same',activation='relu'))#16 RF=7x7
model.add(BatchNormalization(momentum=0.011))
model.add(Dropout(0.1622))

model.add(Convolution2D(10,1,1,activation='relu'))#16 RF=7x7
model.add(BatchNormalization(momentum=0.011))
model.add(Dropout(0.1622))
model.add(SeparableConv2D(10,kernel_size=(3,3),strides=2,padding='same',activation='relu'))#8 RF=11x11
model.add(BatchNormalization(momentum=0.011))
model.add(Dropout(0.1622))

model.add(SeparableConv2D(10,kernel_size=(3,3),strides=2,padding='same',activation='relu'))#4 RF=19x19

model.add(GlobalAveragePooling2D())#

model.add(Activation('softmax'))

Strategy used.
Tried to increase the z axis depth and reduce the x and y with some balance.
No Dense layers used
Convolution 2D not used
For Data augmentation added: feature_std_normalization
Have used Dropout to minimize overfitting and adjusted Batch Normalization value accordingly. used momentum as it is not mini batch.
Have increased the samples per epoch, was just trying to increase the train data to minimize overfitting.
Tried to optimize the learning rate with with scheduler.
--------------------------------------------------------------------------------------------------------------------------------

Final VAlidation Accuracy of BAse model:Accuracy on test data is: 82.66
=================================================================================================================================
Epoch 00001: LearningRateScheduler setting learning rate to 0.009942.
2734/2734 [==============================] - 92s 34ms/step - loss: 1.1000 - acc: 0.6014 - val_loss: 0.8428 - val_acc: 0.6998
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0075375284.
2734/2734 [==============================] - 88s 32ms/step - loss: 0.8114 - acc: 0.7104 - val_loss: 0.7044 - val_acc: 0.7529
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0060695971.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.7168 - acc: 0.7458 - val_loss: 0.6590 - val_acc: 0.7731
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0050802248.
2734/2734 [==============================] - 88s 32ms/step - loss: 0.6650 - acc: 0.7660 - val_loss: 0.6160 - val_acc: 0.7866
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0043681898.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.6289 - acc: 0.7792 - val_loss: 0.6317 - val_acc: 0.7843
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0038312139.
2734/2734 [==============================] - 88s 32ms/step - loss: 0.6064 - acc: 0.7863 - val_loss: 0.5951 - val_acc: 0.7948
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0034118051.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.5870 - acc: 0.7936 - val_loss: 0.5712 - val_acc: 0.8014
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0030751624.
2734/2734 [==============================] - 88s 32ms/step - loss: 0.5715 - acc: 0.7990 - val_loss: 0.5674 - val_acc: 0.8052
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0027989865.
2734/2734 [==============================] - 88s 32ms/step - loss: 0.5582 - acc: 0.8039 - val_loss: 0.5502 - val_acc: 0.8103
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0025683286.
2734/2734 [==============================] - 88s 32ms/step - loss: 0.5492 - acc: 0.8069 - val_loss: 0.5443 - val_acc: 0.8145
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0023727924.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.5427 - acc: 0.8093 - val_loss: 0.5403 - val_acc: 0.8163
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0022049235.
2734/2734 [==============================] - 88s 32ms/step - loss: 0.5338 - acc: 0.8130 - val_loss: 0.5474 - val_acc: 0.8167
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0020592378.
2734/2734 [==============================] - 88s 32ms/step - loss: 0.5268 - acc: 0.8150 - val_loss: 0.5358 - val_acc: 0.8183
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0019316106.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.5213 - acc: 0.8160 - val_loss: 0.5392 - val_acc: 0.8157
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0018188804.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.5147 - acc: 0.8188 - val_loss: 0.5338 - val_acc: 0.8197
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0017185825.
2734/2734 [==============================] - 88s 32ms/step - loss: 0.5110 - acc: 0.8213 - val_loss: 0.5171 - val_acc: 0.8254
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.001628768.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.5049 - acc: 0.8225 - val_loss: 0.5168 - val_acc: 0.8248
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0015478748.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.5045 - acc: 0.8218 - val_loss: 0.5150 - val_acc: 0.8229
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0014746366.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4990 - acc: 0.8242 - val_loss: 0.5167 - val_acc: 0.8249
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.0014080159.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4954 - acc: 0.8260 - val_loss: 0.5326 - val_acc: 0.8231
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0013471545.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4940 - acc: 0.8263 - val_loss: 0.5081 - val_acc: 0.8305
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.0012913365.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4915 - acc: 0.8269 - val_loss: 0.5148 - val_acc: 0.8249
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0012399601.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4875 - acc: 0.8283 - val_loss: 0.5280 - val_acc: 0.8203
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0011925153.
2734/2734 [==============================] - 88s 32ms/step - loss: 0.4872 - acc: 0.8284 - val_loss: 0.5246 - val_acc: 0.8202
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0011485675.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4819 - acc: 0.8296 - val_loss: 0.4997 - val_acc: 0.8299
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0011077437.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4810 - acc: 0.8305 - val_loss: 0.5124 - val_acc: 0.8272
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0010697224.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4784 - acc: 0.8308 - val_loss: 0.5076 - val_acc: 0.8267
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0010342245.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4791 - acc: 0.8308 - val_loss: 0.5183 - val_acc: 0.8231
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0010010068.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4763 - acc: 0.8324 - val_loss: 0.5114 - val_acc: 0.8251
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0009698566.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4738 - acc: 0.8329 - val_loss: 0.5002 - val_acc: 0.8283
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0009405866.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4723 - acc: 0.8336 - val_loss: 0.5058 - val_acc: 0.8288
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0009130315.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4699 - acc: 0.8340 - val_loss: 0.5197 - val_acc: 0.8239
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.000887045.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4717 - acc: 0.8337 - val_loss: 0.5082 - val_acc: 0.8281
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0008624967.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4701 - acc: 0.8340 - val_loss: 0.4974 - val_acc: 0.8301
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0008392706.
2734/2734 [==============================] - 88s 32ms/step - loss: 0.4664 - acc: 0.8347 - val_loss: 0.4938 - val_acc: 0.8317
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0008172626.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4675 - acc: 0.8352 - val_loss: 0.5191 - val_acc: 0.8264
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0007963794.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4644 - acc: 0.8355 - val_loss: 0.5066 - val_acc: 0.8275
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0007765367.
2734/2734 [==============================] - 88s 32ms/step - loss: 0.4632 - acc: 0.8368 - val_loss: 0.4987 - val_acc: 0.8303
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0007576589.
2734/2734 [==============================] - 88s 32ms/step - loss: 0.4628 - acc: 0.8358 - val_loss: 0.4937 - val_acc: 0.8327
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0007396771.
2734/2734 [==============================] - 88s 32ms/step - loss: 0.4607 - acc: 0.8373 - val_loss: 0.5050 - val_acc: 0.8296
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0007225291.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4619 - acc: 0.8372 - val_loss: 0.4883 - val_acc: 0.8321
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0007061581.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4600 - acc: 0.8381 - val_loss: 0.4891 - val_acc: 0.8323
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0006905126.
2734/2734 [==============================] - 88s 32ms/step - loss: 0.4590 - acc: 0.8376 - val_loss: 0.5155 - val_acc: 0.8263
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0006755453.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4580 - acc: 0.8381 - val_loss: 0.5057 - val_acc: 0.8274
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0006612131.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4585 - acc: 0.8381 - val_loss: 0.4822 - val_acc: 0.8356
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0006474764.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4565 - acc: 0.8387 - val_loss: 0.5066 - val_acc: 0.8276
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0006342988.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4555 - acc: 0.8394 - val_loss: 0.4873 - val_acc: 0.8352
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.000621647.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4549 - acc: 0.8394 - val_loss: 0.4875 - val_acc: 0.8360
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0006094899.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4531 - acc: 0.8402 - val_loss: 0.4900 - val_acc: 0.8363
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0005977993.
2734/2734 [==============================] - 87s 32ms/step - loss: 0.4522 - acc: 0.8401 - val_loss: 0.4925 - val_acc: 0.8331
Model took 4374.22 seconds to train

Accuracy on test data is: 83.31
