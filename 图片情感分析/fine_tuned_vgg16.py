from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from pathlib import Path
import numpy as np
import argparse
import os
import cv2

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True,
#	help="path to input dataset")
#ap.add_argument("-m", "--model", required=True,
#	help="path to output model")
#args = vars(ap.parse_args())


#----------------------------------------------------
# 重采样，扩大数据集；
#aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
#	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
#	horizontal_flip=True, fill_mode="nearest")
#----------------------------------------------------

def load_img(root):
	print('-----加载数据集-----')
	path = Path(root)
	img_paths = list(path.glob('./**/*.jpg'))
	img_paths = [str(name) for name in img_paths]
	class_names = [os.path.split(os.path.split(name)[0])[1] for name in img_paths]
	return img_paths, class_names

def read_img(path, classes):
	print('-----读取图片-----')
	img_arrs = [cv2.imread(p) for p in path]
	img_arrs = [cv2.resize(img_arr, (224, 224)) for img_arr in img_arrs]
	img_arrays = np.array(img_arrs).astype(np.float32)/ 255.0
	labels = LabelBinarizer().fit_transform(classes)
	classNames = [str(i) for i in np.unique(classes)]
	return img_arrays, labels, classNames

# grab the list of images, then extract
# the class label names from the image paths
#print("[INFO] loading images...")
#p = Path(args["dataset"])
#imagePaths = list(p.glob('./**/*.jpg'))#获得所有图片的路径；
#imagePaths = [str(names) for names in imagePaths]
#classNames=[os.path.split(os.path.split(names)[0])[1] for names in imagePaths]
#classNames = [str(x) for x in np.unique(classNames)]#类别名称；

# initialize the image preprocessors
#sp = SimplePreprocessor(224, 224)#将图片resize为224*224，以适应VGG16的输入；
#iap = ImageToArrayPreprocessor()#将转化图片数据格式；

# load the dataset from disk then scale the raw pixel intensities to
# the range [0, 1]
#sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
#(data, labels) = sdl.load(imagePaths, verbose=500)
#data = data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
# convert the labels from integers to vectors
#trainY = LabelBinarizer().fit_transform(trainY)#将标签转化为[[1,0,0], [0,1,0]]的形式;
#testY = LabelBinarizer().fit_transform(testY)

#batch_size= 32;
def generator(trainX, trainY):
	print('-----构造迭代器-----')
	while 1:
		for i in range(0, len(trainX), 32):
			X_batch = trainX[i:i + 12]
			Y_batch = trainY[i:i + 12]
			yield (X_batch, Y_batch)

#-------------------------------------------------------------------------------#
def build_model(img_arrays, labels, classNames):
	trainX, testX, trainY, testY = train_test_split(img_arrays, labels, test_size=0.2, random_state=42)
	#加载VGG16模型，在模型后加上一层新的全连接层；
	baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
	# 初始化网络，在VGG16的基础上加上一层全连接层，并接上softmax进行分类；
	headModel = baseModel.output
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(256, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(17, activation="softmax")(headModel)
# place the head FC model on top of the base model -- this will
# become the actual model we will train
	model = Model(inputs=baseModel.input, outputs=headModel)
# 对于VGG16模型中的卷积层和全连接层不进行训练；
	for layer in baseModel.layers:
		layer.trainable = False

	# 编译模型，设置学习率；
	print('-----模型编译中-----')
	opt = RMSprop(lr=0.001)
	model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
#至此，模型已搭建完毕；
#-------------------------------------------------------------------------------#
	# 对新的全连接层进行训练初始化；
	print("[INFO] training head...")
	model.fit_generator(generator(trainX, trainY),
						validation_data=(testX, testY), epochs=25,
						steps_per_epoch=len(trainX) // 32, verbose=1)

# 评估初始化后的网络；
	print('-----评估初始化后网络-----')
	predictions = model.predict(testX, batch_size=32)
	print(classification_report(testY.argmax(axis=1),
								predictions.argmax(axis=1), target_names=classNames))

# 当新的全连接层被初始化后，对网络进行微调，释放VGG16的最后一层卷积神经网络并
#重新进行训练，被初始化的全连接层没有进行训练；
	for layer in baseModel.layers[15:]:
		layer.trainable = True

# 对微调后的模型重新编译；
	print('-----重新编译初始化后的模型-----')
	opt = SGD(lr=0.001)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
				  metrics=["accuracy"])

# 再次训练已构建的模型，对基准模型的最后一层卷积层和新加入的全连接层重新训练以及初始化；
	print('-----微调模型构建完毕-----')
	model.fit_generator(generator(trainX, trainY),
						validation_data=(testX, testY), epochs=100,
						steps_per_epoch=len(trainX) // 32, verbose=1)

	# 评估微调的模型；
	print('----评估微调后的模型-----')
	predictions = model.predict(testX, batch_size=32)
	print(classification_report(testY.argmax(axis=1),
								predictions.argmax(axis=1), target_names=classNames))
	print('模型层数'+ str(len(model.layers)))
	print(model.summary())
	model.save(r'D:/fine_tuned_cnn.h5')

if __name__ == '__main__':
	img_paths, class_names = load_img(r'E:/kFineTuning-master/flowers17')
	img_arrays, labels, classNames = read_img(img_paths, class_names)
	build_model(img_arrays, labels, classNames)



