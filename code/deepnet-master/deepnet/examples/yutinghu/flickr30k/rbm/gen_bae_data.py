import os, sys
import numpy as np

def main():
  prefix = 'data/rbm_reps'
  image_path = os.path.join(prefix, 'image_rbm2_LAST/train/image_hidden2-00001-of-00001.npy')
  text_path = os.path.join(prefix, 'text_rbm2_LAST/train/text_hidden2-00001-of-00001.npy')
  
  image = np.load(image_path)
  text = np.load(text_path)
  
  numcases, dimensions = image.shape
  fake_data = np.zeros((numcases, dimensions))
  
  train_image = np.r_[image,fake_data,image]
  train_text = np.r_[fake_data,text,text]
  
  all_image = np.r_[image,image,image]
  all_text = np.r_[text,text,text]
  
  save_dir = os.path.join(prefix, 'bae_joint_layer2')
  if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)
    for dataset in ['train', 'validation', 'test', 'train_all']:
      os.mkdir(os.path.join(save_dir, dataset))
  
  image_name = 'image_hidden2-00001-of-00001.npy'
  text_name = 'text_hidden2-00001-of-00001.npy'
  np.save(os.path.join(save_dir, 'train', image_name), train_image)
  np.save(os.path.join(save_dir, 'train', text_name), train_text)
  
  np.save(os.path.join(save_dir, 'train_all', image_name), all_image)
  np.save(os.path.join(save_dir, 'train_all', text_name), all_text)
  
  for dataset in ['validation', 'test']:
    s = os.path.join(prefix, 'image_rbm2_LAST', dataset, image_name)
    d = os.path.join(save_dir, dataset, image_name)
    os.system('cp ' + s + ' ' + d)
    s = os.path.join(prefix, 'text_rbm2_LAST', dataset, text_name)
    d = os.path.join(save_dir, dataset, text_name)
    os.system('cp ' + s + ' ' + d)

if __name__ == '__main__':
  main()
  