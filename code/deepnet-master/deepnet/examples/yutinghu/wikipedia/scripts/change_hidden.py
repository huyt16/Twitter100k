from deepnet import util
from deepnet import deepnet_pb2
import sys, os
from google.protobuf import text_format
import glob
import copy

def change_model(proto, layers=None):
  model = util.ReadModel(proto)
  if layers is None:
    layers = ['image_hidden1', 'image_hidden2', 'image_hidden3',
              'text_hidden1', 'text_hidden2', 'text_hidden3',
              'image_layer', 'text_layer', 'joint_layer',
              'image_tied_hidden', 'text_tied_hidden',
              'image_hidden2_recon', 'text_hidden2_recon',
              'cross_image_hidden2_recon', 'cross_text_hidden2_recon']
  
  for layer in layers:
    try:
      layer_proto = next(lay for lay in model.layer if lay.name == layer)
      layer_proto.dimensions = dimensions
    except StopIteration:
        pass
  
  with open(proto, 'w') as f:
    text_format.PrintMessage(model, f)
  
def change_data(proto, datas=None):
  proto_cont = util.ReadData(proto)
  if datas is None:
    datas = []
    for m in ['image', 'text']:
      for i in [1,2,3]:
        for t in ['train', 'validation', 'test']:
          datas += [m+'_'+'hidden'+str(i)+'_'+t]
          datas += ['bae_'+m+'_'+'hidden'+str(i)+'_'+t]
          datas += ['bae_'+m+'_'+'hidden'+str(i)+'_'+t+'_all']
          datas += ['corr_'+m+'_hidden'+str(i)+'_'+t]
  for data in datas:
    try:
      data_proto = next(lay for lay in proto_cont.data if lay.name == data)
      data_proto.dimensions[0] = dimensions
    except StopIteration:
        pass
  with open(proto, 'w') as f:
    text_format.PrintMessage(proto_cont, f)
    
dimensions = int(sys.argv[1])
pre_dir = sys.argv[2]

for prefix in ['ae_models', 'rbm_models']:
  prefix = os.path.join(pre_dir, prefix)
  files = os.listdir(prefix)
  for model_file in files:
    proto = os.path.join(prefix, model_file)
    change_model(proto)
    
change_data(os.path.join(pre_dir, 'data.pbtxt'))
    