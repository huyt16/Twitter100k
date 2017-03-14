#!/bin/bash

gpu_mem=4G
main_mem=30G

trainer=${deepnet}/trainer.py
extract_rep=${deepnet}/extract_rbm_representation.py
model_output_dir=data/rbm_models
data_output_dir=data/rbm_reps

models_dir=rbm_models
trainers_dir=rbm_trainers

clobber=false

mkdir -p ${model_output_dir}
mkdir -p ${data_output_dir}

if ${clobber} || [ ! -e ${model_output_dir}/joint_layer2_LAST ]; then
  echo "Training joint layer."
  python ${trainer} ${models_dir}/joint_layer2.pbtxt \
    ${trainers_dir}/train_CD_joint_layer.pbtxt eval.pbtxt || exit 1
fi

if ${clobber} || [ ! -e ${data_output_dir}/joint_layer2_generated_text/data.pbtxt ]; then
  echo "Inferring missing text"
  python ../scripts/sample_text.py ${model_output_dir}/joint_layer2_LAST \
    ${trainers_dir}/train_CD_joint_layer.pbtxt ${data_output_dir}/joint_layer2_generated_text \
    data.pbtxt ${gpu_mem} ${main_mem} || exit 1
fi

if ${clobber} || [ ! -e ${data_output_dir}/joint_layer2_generated_image/data.pbtxt ]; then
  echo "Inferring missing image"
  python ../scripts/sample_image.py ${model_output_dir}/joint_layer2_LAST \
    ${trainers_dir}/train_CD_joint_layer.pbtxt ${data_output_dir}/joint_layer2_generated_image \
    data.pbtxt ${gpu_mem} ${main_mem} || exit 1
fi
