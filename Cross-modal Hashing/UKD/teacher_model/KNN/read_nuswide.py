import os
import moxing as mox
mox.file.shift('os', 'mox')

import pdb
import numpy as np
import PIL
import scipy.io as scio
import h5py

def loading_data():
	#labels_path = 's3://bucket-7000/huhengtong/NUS-WIDE_DCMH/nus-wide-tc21-lall.mat'
	labels_path = 's3://bucket-7000/huhengtong/NUS-WIDE_DCMH/NUS-WIDE-tc10/nus-wide-tc10-lall.mat'
	labels = scio.loadmat(labels_path)['LAll']
	#labels = h5py.File(labels_path)['LAll']
	#print(labels.shape)

	#tags_path = 's3://bucket-7000/huhengtong/NUS-WIDE_DCMH/NUS-WIDE-tc10/nus-wide-tc10-yall.mat'
	tags_path = '/cache/nus-wide-tc10-yall.mat'
	#tags = scio.loadmat(tags_path)['YAll']

	tags = h5py.File(tags_path)
	print(tags.keys())

	tags = np.transpose(h5py.File(tags_path)['YAll'])
	#print(tags.shape)

	#imagePaths_path = 's3://bucket-7000/huhengtong/NUS-WIDE_DCMH/nus-wide-tc21-fall.mat'
	imagePaths_path = 's3://bucket-7000/huhengtong/NUS-WIDE_DCMH/NUS-WIDE-tc10/nus-wide-tc10-fall.mat'
	imagePaths = scio.loadmat(imagePaths_path)['FAll']
	#print(imagePaths.shape)

	labels_vsum, tags_vsum = labels.sum(axis=1), tags.sum(axis=1)
	indices = [i!=0 and j!=0 for (i,j) in zip(labels_vsum,tags_vsum)]
	labels_csum = labels.sum(axis=0)
	idx = np.argsort(labels_csum)
	labels_top10 = labels[:, idx[-11:-1]]
	print(labels_top10.shape)
	labels_top10_vsum = labels_top10.sum(axis=1)
	indices = [i != 0 for i in labels_top10_vsum]
	imagePaths_pruned = imagePaths[indices]
	labels_pruned = labels_top10[indices]
	tags_pruned = tags[indices]

	#print(imagePaths_pruned.shape, labels_pruned.shape, tags_pruned.shape)
	#images = read_images(imagePaths)
	#print(images.shape)
	#pdb.set_trace()
	#return imagePaths_pruned, tags_pruned, labels_pruned
return imagePaths, tags, labels

def read_images(imgs_path):
	prefix_path = 's3://bucket-7000/huhengtong/Flickr'
	imgs = []
	for each_img_name in imgs_path:
	each_img_name = each_img_name[0][0].strip('/')
	full_path_single_img = os.path.join(prefix_path, each_img_name)
	#print(full_path_single_img)
	#full_path_single_img = full_path_single_img.replace("\", "/").strip()
	with mox.file.File(full_path_single_img, 'rb') as f:
	with PIL.Image.open(f) as img:
	#print(img)
	new_height = max(224, img.size[0])
	new_width = max(224, img.size[1])
	img = img.convert('RGB').resize((new_height, new_width))
	croped_img = img.crop(((new_height-224)/2, (new_width-224)/2, (new_height-224)/2+224, (new_width-224)/2+224))
	#print(croped_img.size)
	#pdb.set_trace()
	imgs.append(np.asarray(croped_img).astype(np.float64))
return np.array(imgs)