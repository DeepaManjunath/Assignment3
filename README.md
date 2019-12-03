# Assignment3
Have used Separable Convolution 2D 32x32 with padding in the beginning
model = Sequential()
model.add(SeparableConv2D(95,kernel_size=(3, 3),activation='relu',padding='same',input_shape=(32, 32, 3)))# input size is 32x32x3 RF=1
model.add(BatchNormalization(momentum=0.155))# Have used BatchNormalization and momentum value as the batch size is not mini batch
model.add(Dropout(0.155))

model.add(SeparableConv2D(95,kernel_size=(3,3),padding='same',activation='relu'))# input size still remains 32 as padding is added in previous layer     #RF=3x3
model.add(BatchNormalization(momentum=0.155))
model.add(Dropout(0.155))
model.add(SeparableConv2D(95,kernel_size=(3,3),strides=2,padding='same',activation='relu'))#32 #RF=5x5 Have used strides instead of Maxpooling to increase the receptive field in the next layer
model.add(BatchNormalization(momentum=0.155))
model.add(Dropout(0.155))



model.add(SeparableConv2D(95,kernel_size=(3,3),padding='same',activation='relu'))#16 RF=7x7
model.add(BatchNormalization(momentum=0.155))
model.add(Dropout(0.155))

model.add(Convolution2D(10,1,1,activation='relu'))#16 RF=7x7
model.add(BatchNormalization(momentum=0.155))
model.add(Dropout(0.155))
model.add(SeparableConv2D(10,kernel_size=(3,3),strides=2,padding='same',activation='relu'))#8 RF=11x11
model.add(BatchNormalization(momentum=0.155))
model.add(Dropout(0.155))

model.add(SeparableConv2D(10,kernel_size=(3,3),strides=2,padding='same',activation='relu'))#4 RF=19x19






model.add(GlobalAveragePooling2D())# Used Global Average pooling instead of flatten

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
2734/2734 [==============================] - 98s 36ms/step - loss: 1.1177 - acc: 0.5977 - val_loss: 0.8557 - val_acc: 0.6961
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0075375284.
2734/2734 [==============================] - 94s 34ms/step - loss: 0.7998 - acc: 0.7162 - val_loss: 0.7491 - val_acc: 0.7347
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0060695971.
2734/2734 [==============================] - 94s 34ms/step - loss: 0.7023 - acc: 0.7511 - val_loss: 0.6650 - val_acc: 0.7712
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0050802248.
2734/2734 [==============================] - 91s 33ms/step - loss: 0.6449 - acc: 0.7729 - val_loss: 0.6370 - val_acc: 0.7822
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0043681898.
2734/2734 [==============================] - 91s 33ms/step - loss: 0.6065 - acc: 0.7869 - val_loss: 0.6118 - val_acc: 0.7923
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0038312139.
2734/2734 [==============================] - 92s 34ms/step - loss: 0.5813 - acc: 0.7956 - val_loss: 0.5692 - val_acc: 0.8053
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0034118051.
2734/2734 [==============================] - 92s 34ms/step - loss: 0.5602 - acc: 0.8038 - val_loss: 0.5852 - val_acc: 0.7982
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0030751624.
2734/2734 [==============================] - 93s 34ms/step - loss: 0.5442 - acc: 0.8086 - val_loss: 0.5614 - val_acc: 0.8078
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0027989865.
2734/2734 [==============================] - 94s 34ms/step - loss: 0.5311 - acc: 0.8135 - val_loss: 0.5739 - val_acc: 0.8050
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0025683286.
2734/2734 [==============================] - 94s 34ms/step - loss: 0.5203 - acc: 0.8172 - val_loss: 0.5676 - val_acc: 0.8097
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0023727924.
2734/2734 [==============================] - 92s 33ms/step - loss: 0.5117 - acc: 0.8200 - val_loss: 0.5727 - val_acc: 0.8079
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0022049235.
2734/2734 [==============================] - 92s 34ms/step - loss: 0.5037 - acc: 0.8232 - val_loss: 0.5692 - val_acc: 0.8109
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0020592378.
2734/2734 [==============================] - 92s 34ms/step - loss: 0.4981 - acc: 0.8251 - val_loss: 0.5530 - val_acc: 0.8130
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0019316106.
2734/2734 [==============================] - 93s 34ms/step - loss: 0.4917 - acc: 0.8274 - val_loss: 0.5497 - val_acc: 0.8151
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0018188804.
2734/2734 [==============================] - 94s 34ms/step - loss: 0.4855 - acc: 0.8290 - val_loss: 0.5524 - val_acc: 0.8168
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0017185825.
2734/2734 [==============================] - 94s 34ms/step - loss: 0.4814 - acc: 0.8305 - val_loss: 0.5457 - val_acc: 0.8198
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.001628768.
2734/2734 [==============================] - 91s 33ms/step - loss: 0.4766 - acc: 0.8319 - val_loss: 0.5301 - val_acc: 0.8235
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0015478748.
2734/2734 [==============================] - 92s 34ms/step - loss: 0.4735 - acc: 0.8332 - val_loss: 0.5465 - val_acc: 0.8168
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0014746366.
2734/2734 [==============================] - 92s 34ms/step - loss: 0.4698 - acc: 0.8347 - val_loss: 0.5419 - val_acc: 0.8180
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.0014080159.
2734/2734 [==============================] - 92s 34ms/step - loss: 0.4655 - acc: 0.8355 - val_loss: 0.5382 - val_acc: 0.8202
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0013471545.
2734/2734 [==============================] - 94s 34ms/step - loss: 0.4633 - acc: 0.8365 - val_loss: 0.5455 - val_acc: 0.8156
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.0012913365.
2734/2734 [==============================] - 94s 34ms/step - loss: 0.4605 - acc: 0.8369 - val_loss: 0.5271 - val_acc: 0.8214
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0012399601.
2734/2734 [==============================] - 93s 34ms/step - loss: 0.4577 - acc: 0.8386 - val_loss: 0.5474 - val_acc: 0.8185
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0011925153.
2734/2734 [==============================] - 92s 33ms/step - loss: 0.4565 - acc: 0.8388 - val_loss: 0.5380 - val_acc: 0.8168
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0011485675.
2734/2734 [==============================] - 92s 34ms/step - loss: 0.4539 - acc: 0.8403 - val_loss: 0.5310 - val_acc: 0.8228
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0011077437.
2734/2734 [==============================] - 92s 34ms/step - loss: 0.4509 - acc: 0.8404 - val_loss: 0.5472 - val_acc: 0.8179
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0010697224.
2734/2734 [==============================] - 93s 34ms/step - loss: 0.4472 - acc: 0.8416 - val_loss: 0.5334 - val_acc: 0.8253
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0010342245.
2734/2734 [==============================] - 94s 34ms/step - loss: 0.4477 - acc: 0.8423 - val_loss: 0.5348 - val_acc: 0.8239
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0010010068.
2734/2734 [==============================] - 94s 34ms/step - loss: 0.4443 - acc: 0.8428 - val_loss: 0.5336 - val_acc: 0.8242
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0009698566.
2734/2734 [==============================] - 91s 33ms/step - loss: 0.4441 - acc: 0.8429 - val_loss: 0.5290 - val_acc: 0.8220
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0009405866.
2734/2734 [==============================] - 92s 33ms/step - loss: 0.4424 - acc: 0.8438 - val_loss: 0.5151 - val_acc: 0.8286
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0009130315.
2734/2734 [==============================] - 92s 34ms/step - loss: 0.4416 - acc: 0.8436 - val_loss: 0.5388 - val_acc: 0.8238
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.000887045.
2734/2734 [==============================] - 93s 34ms/step - loss: 0.4394 - acc: 0.8445 - val_loss: 0.5222 - val_acc: 0.8268
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0008624967.
2734/2734 [==============================] - 94s 34ms/step - loss: 0.4397 - acc: 0.8441 - val_loss: 0.5472 - val_acc: 0.8203
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0008392706.
2734/2734 [==============================] - 94s 34ms/step - loss: 0.4355 - acc: 0.8464 - val_loss: 0.5339 - val_acc: 0.8258
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0008172626.
2734/2734 [==============================] - 93s 34ms/step - loss: 0.4367 - acc: 0.8459 - val_loss: 0.5468 - val_acc: 0.8215
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0007963794.
2734/2734 [==============================] - 92s 34ms/step - loss: 0.4346 - acc: 0.8464 - val_loss: 0.5227 - val_acc: 0.8281
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0007765367.
2734/2734 [==============================] - 91s 33ms/step - loss: 0.4335 - acc: 0.8469 - val_loss: 0.5255 - val_acc: 0.8251
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0007576589.
2734/2734 [==============================] - 93s 34ms/step - loss: 0.4310 - acc: 0.8477 - val_loss: 0.5280 - val_acc: 0.8295
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0007396771.
2734/2734 [==============================] - 93s 34ms/step - loss: 0.4319 - acc: 0.8475 - val_loss: 0.5163 - val_acc: 0.8295
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0007225291.
2734/2734 [==============================] - 94s 34ms/step - loss: 0.4291 - acc: 0.8485 - val_loss: 0.5402 - val_acc: 0.8215
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0007061581.
2734/2734 [==============================] - 93s 34ms/step - loss: 0.4288 - acc: 0.8486 - val_loss: 0.5249 - val_acc: 0.8290
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0006905126.
2734/2734 [==============================] - 92s 34ms/step - loss: 0.4263 - acc: 0.8482 - val_loss: 0.5366 - val_acc: 0.8256
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0006755453.
2734/2734 [==============================] - 92s 34ms/step - loss: 0.4282 - acc: 0.8490 - val_loss: 0.5175 - val_acc: 0.8290
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0006612131.
2734/2734 [==============================] - 92s 34ms/step - loss: 0.4247 - acc: 0.8498 - val_loss: 0.5211 - val_acc: 0.8293
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0006474764.
2734/2734 [==============================] - 93s 34ms/step - loss: 0.4263 - acc: 0.8491 - val_loss: 0.5285 - val_acc: 0.8286
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0006342988.
2734/2734 [==============================] - 94s 34ms/step - loss: 0.4242 - acc: 0.8499 - val_loss: 0.5233 - val_acc: 0.8271
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.000621647.
2734/2734 [==============================] - 94s 34ms/step - loss: 0.4238 - acc: 0.8494 - val_loss: 0.5335 - val_acc: 0.8262
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0006094899.
2734/2734 [==============================] - 92s 34ms/step - loss: 0.4230 - acc: 0.8505 - val_loss: 0.5253 - val_acc: 0.8280
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0005977993.
2734/2734 [==============================] - 93s 34ms/step - loss: 0.4213 - acc: 0.8506 - val_loss: 0.5241 - val_acc: 0.8306
Model took 4641.36 seconds to train

Accuracy on test data is: 83.06
