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
=================================================================================================================================

Epoch 00001: LearningRateScheduler setting learning rate to 0.009942.
2734/2734 [==============================] - 104s 38ms/step - loss: 1.1827 - acc: 0.5677 - val_loss: 0.9040 - val_acc: 0.6777
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0075375284.
2734/2734 [==============================] - 97s 36ms/step - loss: 0.8524 - acc: 0.6957 - val_loss: 0.7570 - val_acc: 0.7295
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0060695971.
2734/2734 [==============================] - 97s 36ms/step - loss: 0.7584 - acc: 0.7301 - val_loss: 0.7010 - val_acc: 0.7584
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0050802248.
2734/2734 [==============================] - 97s 36ms/step - loss: 0.7035 - acc: 0.7502 - val_loss: 0.6872 - val_acc: 0.7596
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0043681898.
2734/2734 [==============================] - 97s 35ms/step - loss: 0.6649 - acc: 0.7639 - val_loss: 0.6388 - val_acc: 0.7770
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0038312139.
2734/2734 [==============================] - 97s 35ms/step - loss: 0.6366 - acc: 0.7739 - val_loss: 0.6164 - val_acc: 0.7897
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0034118051.
2734/2734 [==============================] - 97s 35ms/step - loss: 0.6156 - acc: 0.7816 - val_loss: 0.6057 - val_acc: 0.7927
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0030751624.
2734/2734 [==============================] - 97s 35ms/step - loss: 0.5992 - acc: 0.7890 - val_loss: 0.6280 - val_acc: 0.7828
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0027989865.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.5862 - acc: 0.7923 - val_loss: 0.5690 - val_acc: 0.8043
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0025683286.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.5726 - acc: 0.7975 - val_loss: 0.5834 - val_acc: 0.8022
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0023727924.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.5641 - acc: 0.8005 - val_loss: 0.5653 - val_acc: 0.8065
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0022049235.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.5544 - acc: 0.8045 - val_loss: 0.5772 - val_acc: 0.8030
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0020592378.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.5464 - acc: 0.8073 - val_loss: 0.5655 - val_acc: 0.8093
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0019316106.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.5425 - acc: 0.8088 - val_loss: 0.5639 - val_acc: 0.8080
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0018188804.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.5338 - acc: 0.8113 - val_loss: 0.5719 - val_acc: 0.8070
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0017185825.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.5293 - acc: 0.8134 - val_loss: 0.5713 - val_acc: 0.8078
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.001628768.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.5249 - acc: 0.8150 - val_loss: 0.5451 - val_acc: 0.8171
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0015478748.
2734/2734 [==============================] - 97s 35ms/step - loss: 0.5199 - acc: 0.8169 - val_loss: 0.5494 - val_acc: 0.8133
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0014746366.
2734/2734 [==============================] - 97s 35ms/step - loss: 0.5180 - acc: 0.8171 - val_loss: 0.5484 - val_acc: 0.8135
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.0014080159.
2734/2734 [==============================] - 97s 35ms/step - loss: 0.5118 - acc: 0.8192 - val_loss: 0.5535 - val_acc: 0.8097
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0013471545.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.5094 - acc: 0.8207 - val_loss: 0.5557 - val_acc: 0.8131
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.0012913365.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.5077 - acc: 0.8206 - val_loss: 0.5467 - val_acc: 0.8145
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0012399601.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.5028 - acc: 0.8222 - val_loss: 0.5476 - val_acc: 0.8154
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0011925153.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.5001 - acc: 0.8237 - val_loss: 0.5571 - val_acc: 0.8129
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0011485675.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.4977 - acc: 0.8245 - val_loss: 0.5474 - val_acc: 0.8152
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0011077437.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.4958 - acc: 0.8245 - val_loss: 0.5474 - val_acc: 0.8144
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0010697224.
2734/2734 [==============================] - 97s 36ms/step - loss: 0.4947 - acc: 0.8258 - val_loss: 0.5386 - val_acc: 0.8178
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0010342245.
2734/2734 [==============================] - 98s 36ms/step - loss: 0.4920 - acc: 0.8260 - val_loss: 0.5338 - val_acc: 0.8225
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0010010068.
2734/2734 [==============================] - 97s 36ms/step - loss: 0.4898 - acc: 0.8271 - val_loss: 0.5468 - val_acc: 0.8191
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0009698566.
2734/2734 [==============================] - 97s 36ms/step - loss: 0.4881 - acc: 0.8275 - val_loss: 0.5476 - val_acc: 0.8164
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0009405866.
2734/2734 [==============================] - 97s 36ms/step - loss: 0.4876 - acc: 0.8281 - val_loss: 0.5551 - val_acc: 0.8167
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0009130315.
2734/2734 [==============================] - 97s 35ms/step - loss: 0.4853 - acc: 0.8290 - val_loss: 0.5539 - val_acc: 0.8177
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.000887045.
2734/2734 [==============================] - 97s 35ms/step - loss: 0.4858 - acc: 0.8289 - val_loss: 0.5362 - val_acc: 0.8208
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0008624967.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.4796 - acc: 0.8309 - val_loss: 0.5413 - val_acc: 0.8174
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0008392706.
2734/2734 [==============================] - 95s 35ms/step - loss: 0.4796 - acc: 0.8303 - val_loss: 0.5415 - val_acc: 0.8188
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0008172626.
2734/2734 [==============================] - 95s 35ms/step - loss: 0.4804 - acc: 0.8305 - val_loss: 0.5329 - val_acc: 0.8219
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0007963794.
2734/2734 [==============================] - 95s 35ms/step - loss: 0.4769 - acc: 0.8313 - val_loss: 0.5342 - val_acc: 0.8201
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0007765367.
2734/2734 [==============================] - 99s 36ms/step - loss: 0.4756 - acc: 0.8315 - val_loss: 0.5429 - val_acc: 0.8169
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0007576589.
2734/2734 [==============================] - 99s 36ms/step - loss: 0.4763 - acc: 0.8323 - val_loss: 0.5340 - val_acc: 0.8211
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0007396771.
2734/2734 [==============================] - 98s 36ms/step - loss: 0.4744 - acc: 0.8319 - val_loss: 0.5303 - val_acc: 0.8244
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0007225291.
2734/2734 [==============================] - 102s 37ms/step - loss: 0.4749 - acc: 0.8329 - val_loss: 0.5370 - val_acc: 0.8199
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0007061581.
2734/2734 [==============================] - 97s 35ms/step - loss: 0.4704 - acc: 0.8336 - val_loss: 0.5308 - val_acc: 0.8219
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0006905126.
2734/2734 [==============================] - 95s 35ms/step - loss: 0.4717 - acc: 0.8333 - val_loss: 0.5326 - val_acc: 0.8222
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0006755453.
2734/2734 [==============================] - 99s 36ms/step - loss: 0.4708 - acc: 0.8335 - val_loss: 0.5344 - val_acc: 0.8243
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0006612131.
2734/2734 [==============================] - 99s 36ms/step - loss: 0.4686 - acc: 0.8337 - val_loss: 0.5491 - val_acc: 0.8189
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0006474764.
2734/2734 [==============================] - 99s 36ms/step - loss: 0.4683 - acc: 0.8345 - val_loss: 0.5261 - val_acc: 0.8230
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0006342988.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.4682 - acc: 0.8349 - val_loss: 0.5341 - val_acc: 0.8239
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.000621647.
2734/2734 [==============================] - 95s 35ms/step - loss: 0.4689 - acc: 0.8345 - val_loss: 0.5319 - val_acc: 0.8237
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0006094899.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.4655 - acc: 0.8352 - val_loss: 0.5339 - val_acc: 0.8224
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0005977993.
2734/2734 [==============================] - 96s 35ms/step - loss: 0.4662 - acc: 0.8355 - val_loss: 0.5235 - val_acc: 0.8269
Model took 4846.84 seconds to train

Accuracy on test data is: 82.69
