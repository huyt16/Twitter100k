from deepnet import util
from deepnet import deepnet_pb2
import sys, os
from google.protobuf import text_format
import glob
import copy

rep_tied_lambda = float(sys.argv[1])
loss_factor = 1 - rep_tied_lambda

pre_dir = sys.argv[2]

prefix = os.path.join(pre_dir, 'ae_models')
for model_file in ['corr_ae.pbtxt','cross_corr_ae.pbtxt','full_corr_ae.pbtxt']:

  proto = os.path.join(prefix, model_file)

  model = util.ReadModel(proto)
  
  try:
    text_tied_hidden_proto = next(lay for lay in model.layer if lay.name == 'text_tied_hidden')
    text_tied_hidden_proto.rep_tied_lambda = rep_tied_lambda
  except StopIteration:
      pass
  try:
    image_tied_hidden_proto = next(lay for lay in model.layer if lay.name == 'image_tied_hidden')
    image_tied_hidden_proto.rep_tied_lambda = rep_tied_lambda
  except StopIteration:
      pass
      
  try:
    text_hidden2_recon_proto = next(lay for lay in model.layer if lay.name == 'text_hidden2_recon')
    text_hidden2_recon_proto.loss_factor = loss_factor
  except StopIteration:
      pass
  try:
    image_hidden2_recon_proto = next(lay for lay in model.layer if lay.name == 'image_hidden2_recon')
    image_hidden2_recon_proto.loss_factor = loss_factor
  except StopIteration:
      pass
  
  try:
    cross_text_hidden2_recon_proto = next(lay for lay in model.layer if lay.name == 'cross_text_hidden2_recon')
    cross_text_hidden2_recon_proto.loss_factor = loss_factor
  except StopIteration:
      pass
  try:
    cross_image_hidden2_recon_proto = next(lay for lay in model.layer if lay.name == 'cross_image_hidden2_recon')
    cross_image_hidden2_recon_proto.loss_factor = loss_factor
  except StopIteration:
      pass
      
  with open(proto, 'w') as f:
    text_format.PrintMessage(model, f)