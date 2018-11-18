import os
from scipy import misc
import numpy as np

color = np.array([[0,0,0], 
		[128, 0, 0], 
		[0, 128, 0], 
		[128, 128, 0], 
		[0, 0, 128], 
		[128, 0, 128],
		[0, 128, 128], 
		[128, 128, 128], 
		[64, 0, 0], 
		[192, 0, 0], 
		[64, 128, 0], 
		[192, 128, 0], 
		[64, 0, 128], 
		[192, 0, 128], 
		[64, 128, 128], 
		[192, 128, 128], 
		[0, 64, 0], 
		[128, 64, 0], 
		[0, 192, 0], 
		[128, 192, 0], 
		[0, 64, 128]])

class Dataloader:
	def __init__(self):
		self.batch_size = 1
		with open("../VOC2012/ImageSets/Segmentation/train.txt", 'rt') as f:
			s = f.read()
			self.data_list = s.split()[:20]
		# self.data_list = self.data_list[:10]
		self.image_dir = "../VOC2012/JPEGImages/"
		self.label_dir = "../VOC2012/SegmentationClass/"
		self.num_class = 21
		self.h = np.zeros((255,255,255)).astype(np.int32)
		for i in range(21):
			self.h[color[i][0], color[i][1], color[i][2]] = i

	def generate(self):
		while True:
			inputs = []
			labels = []
			for i in range(self.batch_size):
				index = np.random.randint(len(self.data_list))
				image = misc.imread(os.path.join(self.image_dir, self.data_list[index]) + ".jpg").astype("float32")
				# image = cv2.resize(image, (300, 240), interpolation=cv2.INTER_NEAREST)
				# image = self.preprocess(image)
				label_img = misc.imread(os.path.join(self.label_dir, self.data_list[index]) + ".png")
				# label_img = cv2.resize(label_img, (300, 240), interpolation=cv2.INTER_NEAREST)
				label = np.zeros((image.shape[0], image.shape[1], self.num_class))
				for x in range(image.shape[0]):
					for y in range(image.shape[1]):
						ind = self.h[label_img[x, y, 0], label_img[x, y, 1], label_img[x, y, 2]]
						label[x, y, ind] = 1
				inputs.append(image)
				labels.append(label)
			yield np.array(inputs), np.array(labels)

	def preprocess(self, x):
		mean = [103.939, 116.779, 123.68]
		x[..., 0] -= mean[0]
		x[..., 1] -= mean[1]
		x[..., 2] -= mean[2]
		return x

	def save_images(self, name, labels):
		for i in range(labels.shape[0]):
			img = np.zeros((labels.shape[1], labels.shape[2], 3))
			for x in range(labels.shape[1]):
				for y in range(labels.shape[2]):
					k = np.argmax(labels[i,x,y,:])
					img[x, y, :] = color[k,:]
			misc.imsave(name, img)

	def get_images(self):
		for i in range(len(self.data_list)):
			image = misc.imread(os.path.join(self.image_dir, self.data_list[i]) + ".jpg").astype("float32")
			yield(np.array([image]))