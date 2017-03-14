import os
import numpy as np
import scipy.io as sio
nn=1024
#model='corr_ae_LAST'
model='cross_corr_ae_LAST'
#model='full_corr_ae_LAST'
alpha=0.3
prefix = os.path.join('rbm_'+str(nn)+'/data/ae_reps', model, str(alpha), 'test')
image = np.load(os.path.join(prefix, 'image_tied_hidden-00001-of-00001.npy'))
text = np.load(os.path.join(prefix, 'text_tied_hidden-00001-of-00001.npy'))
print image.shape,text.shape
sio.savemat('rbm_'+str(nn)+'_'+model+'_'+str(alpha)+'feature.mat',{'image':image,'text':text})
