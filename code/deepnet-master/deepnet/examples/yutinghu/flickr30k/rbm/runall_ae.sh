#!/bin/bash

gpu_mem=4G
main_mem=30G

trainer=${deepnet}/trainer.py
extract_rep=${deepnet}/extract_rbm_representation.py
extract_rep_nn=${deepnet}/extract_neural_net_representation.py
model_output_dir=data/ae_models
data_output_dir=data/ae_reps

models_dir=ae_models
trainers_dir=ae_trainers

clobber=false

mkdir -p ${model_output_dir}
mkdir -p ${data_output_dir}

echo AEs
if ${clobber} || [ ! -e ${model_output_dir}/ae_LAST ]; then
  python ${trainer} ${models_dir}/ae.pbtxt \
    ${trainers_dir}/train_ae_layer.pbtxt eval.pbtxt || exit 1
  python ${extract_rep_nn} ${model_output_dir}/ae_LAST \
    ${trainers_dir}/train_ae_layer.pbtxt ${data_output_dir}/ae_LAST/${i} \
    None image_tied_hidden text_tied_hidden || exit 1
fi

echo Cross-AEs
if ${clobber} || [ ! -e ${model_output_dir}/cross_ae_LAST ]; then
  python ${trainer} ${models_dir}/cross_ae.pbtxt \
    ${trainers_dir}/train_ae_layer.pbtxt eval.pbtxt || exit 1
  python ${extract_rep_nn} ${model_output_dir}/cross_ae_LAST \
    ${trainers_dir}/train_ae_layer.pbtxt ${data_output_dir}/cross_ae_LAST/${i} \
    None image_tied_hidden text_tied_hidden || exit 1
fi

echo Full-AEs
if ${clobber} || [ ! -e ${model_output_dir}/full_ae_LAST ]; then
  python ${trainer} ${models_dir}/full_ae.pbtxt \
    ${trainers_dir}/train_ae_layer.pbtxt eval.pbtxt || exit 1
  python ${extract_rep_nn} ${model_output_dir}/full_ae_LAST \
    ${trainers_dir}/train_ae_layer.pbtxt ${data_output_dir}/full_ae_LAST/${i} \
    None image_tied_hidden text_tied_hidden || exit 1
fi
