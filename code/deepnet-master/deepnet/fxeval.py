import os, sys
import numpy as np
import math
import scipy.spatial
from operator import itemgetter, attrgetter
  
def fx_calc_map_label(image, text, label, k = 0, dist_method='L2'):
  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(k):
      if label[i] == label[order[j]]:
        r += 1
        p += (r / (j + 1))
    if r > 0:
      res += [p / r]
    else:
      res += [0]
  return np.mean(res)
  
def fx_calc_map_label_k_dist(dist, label, k = 0, dist_method='L2'):
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(k):
      if label[i] == label[order[j]]:
        r += 1
        p += (r / (j + 1))
    if r > 0:
      res += [p / r]
    else:
      res += [0]
  return np.mean(res)
  
def fx_calc_map_multilabel(image, text, label, k=0, n=1000, dist_method='L2'):
  image = image[:n,:]
  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  # print k
  res = []
  for i in range(numcases):
    order = ord[i].reshape(-1)

    tmp_label = (np.dot(label[order], label[i]) > 0)
    if tmp_label.sum() > 0:
      prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
      total_pos = float(tmp_label.sum())
      if total_pos > 0:
        res += [np.dot(tmp_label, prec) / total_pos]
      
  return np.mean(res)
  
def fx_calc_map_multilabel_k(image, text, label, k=0, dist_method='L2'):
  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i].reshape(-1)

    tmp_label = (np.dot(label[order], label[i]) > 0)
    if tmp_label.sum() > 0:
      prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
      total_pos = float(tmp_label.sum())
      if total_pos > 0:
        res += [np.dot(tmp_label, prec) / total_pos]
      
  return np.mean(res)
  
def fx_calc_map_nolabel(image, text, dist_method='COS'):
  if dist_method == 'L1':
    dist = scipy.spatial.distance.cdist(image, text, 'minkowski', 1)
  elif dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  order = dist.argsort()
  numcases = dist.shape[0]
  res = np.zeros((numcases, numcases))
  
  for i in range(numcases):
    res[i, order[i].tolist().index(i)] = 1
    
  res = (numcases - res.cumsum(axis=1).sum(axis=0)) / numcases
  area = 0.5 * (1 + res[0])
  for i in range(numcases-1):
    area += 0.5 * (res[i] + res[i+1])
  area /= numcases
  return area

def fx_calc_map_nolabel_cmc(image, text, dist_method='COS'):
  if dist_method == 'L1':
    dist = scipy.spatial.distance.cdist(image, text, 'minkowski', 1)
  elif dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  order = dist.argsort()
  numcases = dist.shape[0]
  res = np.zeros((numcases, numcases))
  
  for i in range(numcases):
    res[i, order[i].tolist().index(i)] = 1
  res = (numcases - res.cumsum(axis=1).sum(axis=0)) / numcases
  return res
  
def fx_calc_map_nolabel_top(image, text, per=0.2, top_k=0):
  numcases = image.shape[0]
  if per != 0:
    top_k = numcases*per
  if top_k == 0:
    print 'make_test error'
    return
  dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  order = dist.argsort()
  r = 0
  for i in xrange(numcases):
    if order[i,:].tolist().index(i) < top_k:
      r += 1
  return r / float(numcases)
  
def fx_calc_dcg_k(image, text, indexes, label, k, dist_method='COS'):
  dist = -(image*text).sum(axis=1)/np.sqrt((image**2).sum(axis=1))
  dist /= np.sqrt((text**2).sum(axis=1))
  
  # dist = ((image-text)**2).sum(axis=1)
  
  index2image = {}
  keys = set()
  for i,ind in enumerate(indexes):
    if ind not in keys:
      keys.add(ind)
      index2image[ind] = []
    index2image[ind] += [(dist[i], label[i])] 
  
  res = []
  for ind,dist_lab in index2image.items():
    dist_lab.sort(key=itemgetter(0))
    s = 0.0
    ct = 1
    for d,lab in dist_lab:
      s += (2.0**lab-1)/math.log(ct+1,2)
      if ct == k:
        break
      ct += 1
    s *= 0.01757
    res += [s]
  return np.mean(res)

def fx_calc_dcg_k_dist(dist, indexes, label, k):
  index2image = {}
  keys = set()
  for i,ind in enumerate(indexes):
    if ind not in keys:
      keys.add(ind)
      index2image[ind] = []
    index2image[ind] += [(dist[i], label[i])] 
  
  res = []
  for ind,dist_lab in index2image.items():
    dist_lab.sort(key=itemgetter(0))
    s = 0.0
    ct = 1
    for d,lab in dist_lab:
      s += (2.0**lab-1)/math.log(ct+1,2)
      if ct == k:
        break
      ct += 1
    s *= 0.01757
    res += [s]
  return np.mean(res)
  
def fx_calc_dcg_k_dist_onequery(dist_lab, k):
  dist_lab.sort(key=itemgetter(0))
  s = 0.0
  ct = 1
  for d,lab in dist_lab:
    s += (2.0**lab-1)/math.log(ct+1,2)
    if ct == k:
      break
    ct += 1
  s *= 0.01757
  return s