#!/bin/bash

script_dir=../scripts
now_dir=$1

gpu_mem=4G
main_mem=30G

trainer=${deepnet}/trainer.py
extract_rep=${deepnet}/extract_rbm_representation.py
extract_rep_nn=${deepnet}/extract_neural_net_representation.py
model_output_dir=data/ae_models
data_output_dir=data/ae_reps

models_dir=ae_models
trainers_dir=ae_trainers

clobber=true

mkdir -p ${model_output_dir}
mkdir -p ${data_output_dir}


echo Corr-AEs
#for i in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99
for i in 0.8
do
  python ${script_dir}/change_tied_lambda.py ${i} ${now_dir} || exit 1
  if ${clobber} || [ ! -e ${model_output_dir}/corr_ae_LAST ]; then
    echo ${i}
    python ${trainer} ${models_dir}/corr_ae.pbtxt \
      ${trainers_dir}/train_ae_layer.pbtxt eval.pbtxt || exit 1
    python ${extract_rep_nn} ${model_output_dir}/corr_ae_BEST \
      ${trainers_dir}/train_ae_layer.pbtxt ${data_output_dir}/corr_ae_LAST/${i} \
      None image_tied_hidden text_tied_hidden || exit 1
  fi
done

echo Cross-Corr-AEs
#for i in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99
for i in 0.2
do
  python ${script_dir}/change_tied_lambda.py ${i} ${now_dir} || exit 1
  if ${clobber} || [ ! -e ${model_output_dir}/cross_corr_ae_LAST ]; then
    echo ${i}
    python ${trainer} ${models_dir}/cross_corr_ae.pbtxt \
      ${trainers_dir}/train_ae_layer.pbtxt eval.pbtxt || exit 1
    python ${extract_rep_nn} ${model_output_dir}/cross_corr_ae_BEST \
      ${trainers_dir}/train_ae_layer.pbtxt ${data_output_dir}/cross_corr_ae_LAST/${i} \
      None image_tied_hidden text_tied_hidden || exit 1
  fi
done

echo Full-Corr-AEs
#for i in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99
for i in 0.8
do
  python ${script_dir}/change_tied_lambda.py ${i} ${now_dir} || exit 1
  if ${clobber} || [ ! -e ${model_output_dir}/full_corr_ae_LAST ]; then
    echo ${i}
    python ${trainer} ${models_dir}/full_corr_ae.pbtxt \
      ${trainers_dir}/train_ae_layer.pbtxt eval.pbtxt || exit 1
    python ${extract_rep_nn} ${model_output_dir}/full_corr_ae_BEST \
      ${trainers_dir}/train_ae_layer.pbtxt ${data_output_dir}/full_corr_ae_LAST/${i} \
      None image_tied_hidden text_tied_hidden || exit 1
  fi
done

