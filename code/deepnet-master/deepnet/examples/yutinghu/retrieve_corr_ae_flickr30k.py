import numpy as np
import scipy.io as sio
import scipy.spatial
import os

print 'load data!'
model=['corr_ae_LAST','cross_corr_ae_LAST','full_corr_ae_LAST']
alpha=['0.8','0.2','0.8']
path='flickr30k/rbm_1024/data/ae_reps/'
test_index=np.load('flickr30k/data/2000_test_index.npy')
test_file_index=[np.floor(x/5) for x in test_index]
nb_test=test_index.shape[0]

savepath='../../../../../result/rank/flickr30k/'
if not os.path.exists(savepath):
	os.mkdir(savepath)
		
for i in np.arange(len(model)):
	m=model[i]
	a=alpha[i]
	print 'model',m,'alpha',a
	image=np.load(path+m+'/'+a+'/test/image_tied_hidden-00001-of-00001.npy')
	text=np.load(path+m+'/'+a+'/test/text_tied_hidden-00001-of-00001.npy')
	print 'test.shape:',text.shape,'image.shape',image.shape
	test_txt=text[test_index]
	test_im=image[test_index]
	index=np.arange(image.shape[0]/5)
	image=image[index*5]
	print 'image.shape',image.shape
	print 'test_txt.shape',test_txt.shape,'test_im.shape',test_im.shape
	dist_method='cosine'

	print 'im2txt'
	rank=np.zeros((nb_test,))
	t=text
	i=test_im
	print 'calculate dist'
        dist=scipy.spatial.distance.cdist(i,t,dist_method)
        print 'sort dist!'
        order=dist.argsort()
        for i in np.arange(nb_test):
		pos=t.shape[0]
		for j in np.arange(5):
			temp=order[i,:].tolist().index(test_file_index[i]*5+j)
			if temp<pos:
				pos=temp
                rank[i]=pos
        sio.savemat(savepath+m+'_'+a+'_im2txt_rank.mat',{'rank':rank})

	print 'txt2im'
	rank=np.zeros((nb_test,))
	t=test_txt
	i=image
	print 'calculate dist'
	dist=scipy.spatial.distance.cdist(t,i,dist_method)
	print 'sort dist!'
	order=dist.argsort()
	for i in np.arange(nb_test):
		rank[i]=order[i,:].tolist().index(test_file_index[i])
	sio.savemat(savepath+m+'_'+a+'_txt2im_rank.mat',{'rank':rank})
