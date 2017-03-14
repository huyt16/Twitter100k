import os,sys
import scipy.io
import numpy as np
from deepnet.fx_util import *
from deepnet.fxeval import *

mcmd = '/usr/local/MATLAB/R2012a/bin/matlab -nodesktop -nosplash -nodisplay -r '

def normdata(data):
  data = data.astype(np.float32)
  return data

def train():
  validation_label_data = np.load('data/validation_lab_data.npy')
  test_label_data = np.load('data/test_lab_data.npy')
  dic = {}
  for nn1 in [32,64,128,256,512]:
    for nn2 in [32,64,128,256,512,1024]:
      nn = str(nn1)+'_'+str(nn2)
      dic[nn] = {}
      for model in ['ae_LAST', 'cross_ae_LAST', 'full_ae_LAST']:
        dic[nn][model] = np.zeros((3,))
        prefix1 = os.path.join('rbm_'+str(nn1)+'/data/ae_reps', model)
        prefix2 = os.path.join('rbm_'+str(nn2)+'/data/ae_reps', model)

        train_image_data = np.load(os.path.join(prefix1, 'train', 'image_tied_hidden-00001-of-00001.npy'))
        validation_image_data = np.load(os.path.join(prefix1, 'validation', 'image_tied_hidden-00001-of-00001.npy'))
        test_image_data = np.load(os.path.join(prefix1, 'test', 'image_tied_hidden-00001-of-00001.npy'))
        train_text_data = np.load(os.path.join(prefix2, 'train', 'text_tied_hidden-00001-of-00001.npy'))
        validation_text_data = np.load(os.path.join(prefix2, 'validation', 'text_tied_hidden-00001-of-00001.npy'))
        test_text_data = np.load(os.path.join(prefix2, 'test', 'text_tied_hidden-00001-of-00001.npy'))
        
        # searchdim = test_text_data.shape[1] / 2
        
        searchdim = 20
        
        scipy.io.savemat('X.mat', {'X':train_image_data})
        scipy.io.savemat('Y.mat', {'Y':train_text_data})

        os.system(mcmd + '"cca()" > log')
        
        param = scipy.io.loadmat('param.mat')
        wx = param['Wx']
        wy = param['Wy']
        
        image = np.dot(validation_image_data, wx[:,::-1])
        text = np.dot(validation_text_data, wy[:,::-1])
        res = -1
        choose_k = 1
        
        for k in range(1,searchdim):
          area1 = calc(text[:,:k], image[:,:k], validation_label_data)
          area2 = calc(image[:,:k], text[:,:k], validation_label_data)
          area = area1+area2
          # print area
          if res < area:
            res = area
            choose_k = k
        image = np.dot(test_image_data, wx[:,::-1])
        text = np.dot(test_text_data, wy[:,::-1])
        print nn, choose_k
        dic[nn][model][0] = calc(image[:,:choose_k], text[:,:choose_k], test_label_data)
        dic[nn][model][1] = calc(text[:,:choose_k], image[:,:choose_k], test_label_data)
        dic[nn][model][2] = (dic[nn][model][0]+dic[nn][model][1]) / 2
      
  print dic
  fx_pickle('cca_res.pkl', dic)
  
def best():
  dic = fx_unpickle('cca_res.pkl')
  best_ae = -1
  best_cae = -1
  best_fae = -1
  for nn1 in [32,64,128,256,512]:
    for nn2 in [32,64,128,256,512,1024]:
      nn = str(nn1)+'_'+str(nn2)
      if best_ae < dic[nn]['ae_LAST'][2]:
        best_ae = dic[nn]['ae_LAST'][2]
        best_ae_nn = nn
      if best_cae < dic[nn]['cross_ae_LAST'][2]:
        best_cae = dic[nn]['cross_ae_LAST'][2]
        best_cae_nn = nn
      if best_fae < dic[nn]['full_ae_LAST'][2]:
        best_fae = dic[nn]['full_ae_LAST'][2]
        best_fae_nn = nn
  print best_ae_nn, dic[best_ae_nn]['ae_LAST']
  print best_cae_nn, dic[best_cae_nn]['cross_ae_LAST']
  print best_fae_nn, dic[best_fae_nn]['full_ae_LAST']
  
  return best_ae_nn, best_cae_nn, best_fae_nn
  
def calc(image, text, label):
  return fx_calc_map_label(image, text, label, k=50, dist_method='COS')
  

def best_dist():
  ae_nn, cae_nn, fae_nn = best()
  best_dist_per(ae_nn, 'ae_LAST')
  best_dist_per(cae_nn, 'cross_ae_LAST')
  best_dist_per(fae_nn, 'full_ae_LAST')
  
def best_dist_per(nn, model):
  validation_label_data = np.load('data/validation_lab_data.npy')
  test_label_data = np.load('data/test_lab_data.npy')
  
  save_name = os.path.join('results', model+'.pkl')
  [nn1, nn2] = nn.split('_')
  
  prefix1 = os.path.join('rbm_'+nn1+'/data/ae_reps', model)
  prefix2 = os.path.join('rbm_'+nn2+'/data/ae_reps', model)

  train_image_data = np.load(os.path.join(prefix1, 'train', 'image_tied_hidden-00001-of-00001.npy'))
  validation_image_data = np.load(os.path.join(prefix1, 'validation', 'image_tied_hidden-00001-of-00001.npy'))
  test_image_data = np.load(os.path.join(prefix1, 'test', 'image_tied_hidden-00001-of-00001.npy'))
  train_text_data = np.load(os.path.join(prefix2, 'train', 'text_tied_hidden-00001-of-00001.npy'))
  validation_text_data = np.load(os.path.join(prefix2, 'validation', 'text_tied_hidden-00001-of-00001.npy'))
  test_text_data = np.load(os.path.join(prefix2, 'test', 'text_tied_hidden-00001-of-00001.npy'))
  
  searchdim = 20
  
  scipy.io.savemat('X.mat', {'X':train_image_data})
  scipy.io.savemat('Y.mat', {'Y':train_text_data})

  os.system(mcmd + '"cca()" > log')
  
  param = scipy.io.loadmat('param.mat')
  wx = param['Wx']
  wy = param['Wy']
  
  image = np.dot(validation_image_data, wx[:,::-1])
  text = np.dot(validation_text_data, wy[:,::-1])
  res = -1
  choose_k = 1
  
  for k in range(1,searchdim):
    area1 = calc(text[:,:k], image[:,:k], validation_label_data)
    area2 = calc(image[:,:k], text[:,:k], validation_label_data)
    area = area1+area2
    # print area
    if res < area:
      res = area
      choose_k = k
  image = np.dot(test_image_data, wx[:,::-1])
  text = np.dot(test_text_data, wy[:,::-1])
  i2t_dist = -fx_cos_distant(image[:,:choose_k], text[:,:choose_k])
  i2t = i2t_dist.argsort()
  t2i_dist = -fx_cos_distant(text[:,:choose_k], image[:,:choose_k])
  t2i = t2i_dist.argsort()
  fx_pickle(save_name, {'i2t':i2t, 't2i':t2i})
  
def best_top():
  ae_nn, cae_nn, fae_nn = best()
  best_top_per(ae_nn, 'ae_LAST')
  best_top_per(cae_nn, 'cross_ae_LAST')
  best_top_per(fae_nn, 'full_ae_LAST')
  
def best_top_per(nn, model):
  validation_label_data = np.load('data/validation_lab_data.npy')
  test_label_data = np.load('data/test_lab_data.npy')
  
  [nn1, nn2] = nn.split('_')
  
  prefix1 = os.path.join('rbm_'+nn1+'/data/ae_reps', model)
  prefix2 = os.path.join('rbm_'+nn2+'/data/ae_reps', model)

  train_image_data = np.load(os.path.join(prefix1, 'train', 'image_tied_hidden-00001-of-00001.npy'))
  validation_image_data = np.load(os.path.join(prefix1, 'validation', 'image_tied_hidden-00001-of-00001.npy'))
  test_image_data = np.load(os.path.join(prefix1, 'test', 'image_tied_hidden-00001-of-00001.npy'))
  train_text_data = np.load(os.path.join(prefix2, 'train', 'text_tied_hidden-00001-of-00001.npy'))
  validation_text_data = np.load(os.path.join(prefix2, 'validation', 'text_tied_hidden-00001-of-00001.npy'))
  test_text_data = np.load(os.path.join(prefix2, 'test', 'text_tied_hidden-00001-of-00001.npy'))
  
  searchdim = 20
  
  scipy.io.savemat('X.mat', {'X':train_image_data})
  scipy.io.savemat('Y.mat', {'Y':train_text_data})

  os.system(mcmd + '"cca()" > log')
  
  param = scipy.io.loadmat('param.mat')
  wx = param['Wx']
  wy = param['Wy']
  
  image = np.dot(validation_image_data, wx[:,::-1])
  text = np.dot(validation_text_data, wy[:,::-1])
  res = -1
  choose_k = 1
  
  for k in range(1,searchdim):
    area1 = calc(text[:,:k], image[:,:k], validation_label_data)
    area2 = calc(image[:,:k], text[:,:k], validation_label_data)
    area = area1+area2
    # print area
    if res < area:
      res = area
      choose_k = k
  image = np.dot(test_image_data, wx[:,::-1])
  text = np.dot(test_text_data, wy[:,::-1])
  i2t = fx_calc_map_nolabel_top(image[:,:choose_k], text[:,:choose_k])
  t2i = fx_calc_map_nolabel_top(text[:,:choose_k], image[:,:choose_k])
  print i2t, t2i, (i2t+t2i) / 2
  
if __name__ == '__main__':
  # train()
  p = sys.argv[1]
  if p == 'train':
    train()
  elif p == 'best':
    best()
  elif p == 'best_dist':
    best_dist()
  elif p == 'best_top':
    best_top()