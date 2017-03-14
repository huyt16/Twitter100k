import os,sys
import numpy as np
from deepnet.fxeval import *
from deepnet.fx_util import *
import matplotlib.pyplot as plt

def train():  
  label = np.load('data/test_lab_data.npy')
  alpha_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
  dic = {}
  for nn in [1024]:
    dic[nn] = {}
    for model in ['corr_ae_LAST', 'cross_corr_ae_LAST', 'full_corr_ae_LAST']:
      dic[nn][model] = np.zeros((11,3))
      for i,alpha in enumerate(alpha_list):
        prefix = os.path.join('rbm_'+str(nn)+'/data/ae_reps', model, str(alpha), 'test')
        image = np.load(os.path.join(prefix, 'image_tied_hidden-00001-of-00001.npy'))
        text = np.load(os.path.join(prefix, 'text_tied_hidden-00001-of-00001.npy'))
        dic[nn][model][i,0] = fx_calc_map_label(image, text, label, k=50, dist_method='COS')
        dic[nn][model][i,1] = fx_calc_map_label(text, image, label, k=50, dist_method='COS')
        dic[nn][model][i,2] = (dic[nn][model][i,0]+dic[nn][model][i,1]) / 2
    
  print dic
  fx_pickle('corr_res.pkl', dic)
  
def best():
  dic = fx_unpickle('corr_res.pkl')
  # 0.2 2
  # 0.8 8
  for nn in [1024]:
 # for nn in [32,64,128,256,512,1024]:
    print nn
    print dic[nn]['corr_ae_LAST'][8], \
          dic[nn]['cross_corr_ae_LAST'][2], \
          dic[nn]['full_corr_ae_LAST'][8]

def best_dist_per(nn, alpha, model):
  save_name = os.path.join('results', model+'.pkl')
  
  prefix = os.path.join('rbm_'+str(nn)+'/data/ae_reps', model, str(alpha), 'test')
  image = np.load(os.path.join(prefix, 'image_tied_hidden-00001-of-00001.npy'))
  text = np.load(os.path.join(prefix, 'text_tied_hidden-00001-of-00001.npy'))
  i2t_dist = -fx_cos_distant(image, text)
  i2t = i2t_dist.argsort()
  t2i_dist = -fx_cos_distant(text, image)
  t2i = t2i_dist.argsort()
  fx_pickle(save_name, {'i2t':i2t, 't2i':t2i})
  
def best_dist():
  nn = 1024
  best_dist_per(nn, 0.8, 'corr_ae_LAST')
  best_dist_per(nn, 0.2, 'cross_corr_ae_LAST')
  best_dist_per(nn, 0.8, 'full_corr_ae_LAST')
  
def best_top_per(nn, alpha, model):
  prefix = os.path.join('rbm_'+str(nn)+'/data/ae_reps', model, str(alpha), 'test')
  image = np.load(os.path.join(prefix, 'image_tied_hidden-00001-of-00001.npy'))
  text = np.load(os.path.join(prefix, 'text_tied_hidden-00001-of-00001.npy'))
  i2t = fx_calc_map_nolabel_top(image, text)
  t2i = fx_calc_map_nolabel_top(text, image)
  print i2t, t2i, (i2t+t2i) / 2

def best_top():
  nn = 1024
  best_top_per(nn, 0.8, 'corr_ae_LAST')
  best_top_per(nn, 0.2, 'cross_corr_ae_LAST')
#  best_top_per(nn, 0.8, 'full_corr_ae_LAST')

def area_per(nn,alpha,model):
  prefix = os.path.join('rbm_'+str(nn)+'/data/ae_reps', model, str(alpha), 'test')
  image = np.load(os.path.join(prefix, 'image_tied_hidden-00001-of-00001.npy'))
  text = np.load(os.path.join(prefix, 'text_tied_hidden-00001-of-00001.npy'))
  i2t = fx_calc_map_nolabel(image, text)
  t2i = fx_calc_map_nolabel(text, image)
# print i2t, t2i, (i2t+t2i) / 2
  print i2t,t2i

def area():
  nn = 1024
  alpha_list = [0.01,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.99]
#  line_set=['b','g','k','c','r','b--','g--','k--','c--','r--','b-.']
#  model_list=['cross_corr_ae_LAST']
  model_list = ['corr_ae_LAST','cross_corr_ae_LAST','full_corr_ae_LAST']
#  model_list=['corr_ae_LAST']
  '''
  for alpha in alpha_list:
        print alpha
        for model in model_list:
                print model
		area_per(nn, alpha, model)
  '''
  for model in model_list:
        print model
        for alpha in alpha_list:
                print alpha
                area_per(nn, alpha, model)


def cmc():
  nn = 1024
 # alpha_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
#  model_list = ['corr_ae_LAST','cross_corr_ae_LAST']
#  model_list=['full_corr_ae_LAST']
  model_list=['corr_ae_LAST']
  alpha_list=[0.7]
#  alpha_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
  for alpha in alpha_list:
    	print alpha
	for model in model_list:
		print model
		t2i=cmc_per(nn, alpha, model)
		np.save('nn='+str(nn)+'_alpha='+str(alpha)+'_model='+model+'.npy',t2i)


def cmc_per(nn, alpha, model):
  prefix = os.path.join('rbm_'+str(nn)+'/data/ae_reps', model, str(alpha), 'test')
  image = np.load(os.path.join(prefix, 'image_tied_hidden-00001-of-00001.npy'))
  text = np.load(os.path.join(prefix, 'text_tied_hidden-00001-of-00001.npy'))
  #i2t = fx_calc_map_nolabel_cmc(image, text)
  t2i = fx_calc_map_nolabel_cmc(text, image)
  return t2i

  
if __name__ == '__main__':
  p = sys.argv[1]
  if p == 'train':
    train()
  elif p == 'best':
    best()
  elif p == 'best_dist':
    best_dist()
  elif p == 'best_top':
    best_top()
  elif p=='cmc':
    cmc()
  elif p=='area':
    area()
