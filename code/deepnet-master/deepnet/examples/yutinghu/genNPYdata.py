import numpy as np
import scipy.io as sio
import os

dataset='wikipedia'
#dataset='flickr30k'
#dataset='twitter100k'
root='../../../../../feature/'+dataset

path=dataset+'/data/'
if not os.path.exists(path):
	os.mkdir(path)

text=sio.loadmat(root+'/bow.mat')['text']
#text=sio.loadmat(root+'/text_word2vec_bow.mat')['text']
nb_sample=text.shape[0]
print 'text shape',text.shape

im_train=sio.loadmat(root+'/train_image.mat')['image']
im_val=sio.loadmat(root+'/val_image.mat')['image']
im_test=sio.loadmat(root+'/test_image.mat')['image']
print im_train.shape,im_val.shape,im_test.shape
np.save(path+'train_img_data',im_train)
np.save(path+'validation_img_data',im_val)
np.save(path+'test_img_data',im_test)

nb_train=im_train.shape[0]
nb_val=im_val.shape[0]
nb_test=im_test.shape[0]

txt_train=text[0:nb_train]
txt_val=text[nb_train:nb_train+nb_val]
txt_test=text[nb_train+nb_val:]
print txt_train.shape,txt_val.shape,txt_test.shape
np.save(path+'train_txt_data',txt_train)
np.save(path+'validation_txt_data',txt_val)
np.save(path+'test_txt_data',txt_test)

label=np.arange(int(nb_sample))
lab_train=label[0:nb_train]
lab_val=label[nb_train:nb_train+nb_val]
lab_test=label[nb_train+nb_val:]
np.save(path+'train_lab_data',lab_train)
np.save(path+'validation_lab_data',lab_val)
np.save(path+'test_lab_data',lab_test)


