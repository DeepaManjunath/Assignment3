# Assignment3
Have used Separable Convolution 2D 32x32 with padding in the beginning
model = Sequential()
model.add(SeparableConv2D(95,kernel_size=(3, 3),activation='relu',padding='same',input_shape=(32, 32, 3)))# input size is 32x32x3 RF=1
model.add(BatchNormalization(momentum=0.655))# Have used BatchNormalization and momentum value as the batch size is not mini batch
model.add(Dropout(0.155))

model.add(SeparableConv2D(95,kernel_size=(3,3),padding='same',activation='relu'))# input size still remains 32 as padding is added in previous layer     #RF=3x3
model.add(BatchNormalization(momentum=0.655))
model.add(Dropout(0.155))
model.add(SeparableConv2D(95,kernel_size=(3,3),strides=2,padding='same',activation='relu'))#32 #RF=5x5 Have used strides instead of Maxpooling to increase the receptive field in the next layer
model.add(BatchNormalization(momentum=0.655))
model.add(Dropout(0.155))



model.add(SeparableConv2D(95,kernel_size=(3,3),padding='same',activation='relu'))#16 RF=7x7
model.add(BatchNormalization(momentum=0.655))
model.add(Dropout(0.155))

model.add(Convolution2D(10,1,1,activation='relu'))#16 RF=7x7
model.add(BatchNormalization(momentum=0.655))
model.add(Dropout(0.155))
model.add(SeparableConv2D(10,kernel_size=(3,3),strides=2,padding='same',activation='relu'))#8 RF=11x11
model.add(BatchNormalization(momentum=0.655))
model.add(Dropout(0.155))

model.add(SeparableConv2D(10,kernel_size=(3,3),strides=2,padding='same',activation='relu'))#4 RF=19x19
model.add(BatchNormalization())
model.add(Dropout(0.155))



model.add(Convolution2D(10,1,1,activation='relu'))#4 RF=19x19. Have used 1x1 for merging or concatenation and reducing the number of kernels

model.add(GlobalAveragePooling2D())# Used Global Average pooling instead of flatten

model.add(Activation('softmax'))
Strategy used.
Tried to increase the z axis depth and reduce the x and y with some balance.
No Dense layers used
Convolution 2D not used
For Data augmentation added: feature_std_normalization
Have used Dropout to minimize overfitting and adjusted Batch Normalization value accordingly
Have increased the samples per epoch, was just trying to increase the train data to minimize overfitting.
Tried to optimize the learning rate with with scheduler.
--------------------------------------------------------------------------------------------------------------------------------

Final VAlidation Accuracy of BAse model:Accuracy on test data is: 82.66

warnings.warn('This ImageDataGenerator specifies '

Epoch 00001: LearningRateScheduler setting learning rate to 0.009942.
2734/2734 [==============================] - 95s 35ms/step - loss: 1.1681 - acc: 0.5728 - val_loss: 0.9177 - val_acc: 0.6722
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0075375284.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.8480 - acc: 0.6960 - val_loss: 0.7657 - val_acc: 0.7315
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0060695971.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.7575 - acc: 0.7301 - val_loss: 0.6943 - val_acc: 0.7576
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0050802248.
2734/2734 [==============================] - 91s 33ms/step - loss: 0.7011 - acc: 0.7510 - val_loss: 0.6761 - val_acc: 0.7631
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0043681898.
2734/2734 [==============================] - 91s 33ms/step - loss: 0.6640 - acc: 0.7652 - val_loss: 0.6412 - val_acc: 0.7758
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0038312139.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.6353 - acc: 0.7760 - val_loss: 0.6207 - val_acc: 0.7889
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0034118051.
2734/2734 [==============================] - 91s 33ms/step - loss: 0.6162 - acc: 0.7837 - val_loss: 0.6166 - val_acc: 0.7876
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0030751624.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5976 - acc: 0.7896 - val_loss: 0.5884 - val_acc: 0.7977
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0027989865.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5822 - acc: 0.7960 - val_loss: 0.6038 - val_acc: 0.7949
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0025683286.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5726 - acc: 0.7989 - val_loss: 0.5967 - val_acc: 0.7957
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0023727924.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5646 - acc: 0.8012 - val_loss: 0.5880 - val_acc: 0.8010
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0022049235.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5542 - acc: 0.8055 - val_loss: 0.5837 - val_acc: 0.8058
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0020592378.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5491 - acc: 0.8068 - val_loss: 0.5764 - val_acc: 0.8062
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0019316106.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5416 - acc: 0.8097 - val_loss: 0.5756 - val_acc: 0.8093
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0018188804.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5375 - acc: 0.8110 - val_loss: 0.5762 - val_acc: 0.8038
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0017185825.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5303 - acc: 0.8135 - val_loss: 0.5834 - val_acc: 0.8039
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.001628768.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5257 - acc: 0.8151 - val_loss: 0.5700 - val_acc: 0.8067
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0015478748.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5233 - acc: 0.8156 - val_loss: 0.5638 - val_acc: 0.8097
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0014746366.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5201 - acc: 0.8175 - val_loss: 0.5654 - val_acc: 0.8099
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.0014080159.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5160 - acc: 0.8184 - val_loss: 0.5562 - val_acc: 0.8109
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0013471545.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5134 - acc: 0.8192 - val_loss: 0.5680 - val_acc: 0.8081
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.0012913365.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5087 - acc: 0.8209 - val_loss: 0.5654 - val_acc: 0.8138
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0012399601.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5083 - acc: 0.8218 - val_loss: 0.5523 - val_acc: 0.8124
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0011925153.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5048 - acc: 0.8225 - val_loss: 0.5496 - val_acc: 0.8165
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0011485675.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5044 - acc: 0.8230 - val_loss: 0.5490 - val_acc: 0.8126
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0011077437.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.5009 - acc: 0.8234 - val_loss: 0.5479 - val_acc: 0.8173
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0010697224.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4983 - acc: 0.8242 - val_loss: 0.5506 - val_acc: 0.8167
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0010342245.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4980 - acc: 0.8249 - val_loss: 0.5638 - val_acc: 0.8140
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0010010068.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4963 - acc: 0.8255 - val_loss: 0.5565 - val_acc: 0.8130
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0009698566.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4941 - acc: 0.8260 - val_loss: 0.5506 - val_acc: 0.8180
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0009405866.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4916 - acc: 0.8272 - val_loss: 0.5494 - val_acc: 0.8172
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0009130315.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4900 - acc: 0.8271 - val_loss: 0.5572 - val_acc: 0.8153
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.000887045.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4896 - acc: 0.8267 - val_loss: 0.5506 - val_acc: 0.8157
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0008624967.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4877 - acc: 0.8275 - val_loss: 0.5517 - val_acc: 0.8169
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0008392706.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4850 - acc: 0.8295 - val_loss: 0.5473 - val_acc: 0.8174
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0008172626.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4851 - acc: 0.8292 - val_loss: 0.5530 - val_acc: 0.8179
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0007963794.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4831 - acc: 0.8294 - val_loss: 0.5465 - val_acc: 0.8204
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0007765367.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4829 - acc: 0.8295 - val_loss: 0.5449 - val_acc: 0.8187
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0007576589.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4813 - acc: 0.8306 - val_loss: 0.5448 - val_acc: 0.8186
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0007396771.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4814 - acc: 0.8307 - val_loss: 0.5471 - val_acc: 0.8185
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0007225291.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4802 - acc: 0.8302 - val_loss: 0.5539 - val_acc: 0.8179
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0007061581.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4793 - acc: 0.8312 - val_loss: 0.5502 - val_acc: 0.8184
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0006905126.
2734/2734 [==============================] - 89s 33ms/step - loss: 0.4764 - acc: 0.8316 - val_loss: 0.5523 - val_acc: 0.8181
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0006755453.
2734/2734 [==============================] - 89s 33ms/step - loss: 0.4773 - acc: 0.8318 - val_loss: 0.5415 - val_acc: 0.8192
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0006612131.
2734/2734 [==============================] - 89s 33ms/step - loss: 0.4775 - acc: 0.8313 - val_loss: 0.5456 - val_acc: 0.8202
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0006474764.
2734/2734 [==============================] - 89s 33ms/step - loss: 0.4755 - acc: 0.8322 - val_loss: 0.5443 - val_acc: 0.8195
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0006342988.
2734/2734 [==============================] - 89s 33ms/step - loss: 0.4754 - acc: 0.8333 - val_loss: 0.5497 - val_acc: 0.8192
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.000621647.
2734/2734 [==============================] - 89s 33ms/step - loss: 0.4729 - acc: 0.8333 - val_loss: 0.5536 - val_acc: 0.8184
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0006094899.
2734/2734 [==============================] - 90s 33ms/step - loss: 0.4729 - acc: 0.8330 - val_loss: 0.5450 - val_acc: 0.8206
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0005977993.
2734/2734 [==============================] - 91s 33ms/step - loss: 0.4713 - acc: 0.8334 - val_loss: 0.5544 - val_acc: 0.8163
Model took 4513.34 seconds to train

Accuracy on test data is: 81.63
