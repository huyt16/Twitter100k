#!/bin/bash

# Location of the data. 
prefix=../data

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

if [ ! -e  ${prefix}/imgstats.npz ]
then
  echo Computing mean / variance
  python ${deepnet}/compute_data_stats.py ${prefix}/twitter100k.pbtxt \
    ${prefix}/imgstats.npz train_img_data || exit 1
fi

# IMAGE LAYER - 1.
if ${clobber} || [ ! -e ${model_output_dir}/image_rbm1_LAST ]; then
  echo "Training first layer image RBM."
  python ${trainer} ${models_dir}/image_rbm1.pbtxt \
    ${trainers_dir}/train_CD_image_layer1.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/image_rbm1_LAST \
    ${trainers_dir}/train_CD_image_layer1.pbtxt image_hidden1 \
    ${data_output_dir}/image_rbm1_LAST ${gpu_mem} ${cpu_mem} || exit 1
fi

# TEXT LAYER - 1.
if ${clobber} || [ ! -e ${model_output_dir}/text_rbm1_LAST ]; then
  echo "Training first layer text RBM."
  python ${trainer} ${models_dir}/text_rbm1.pbtxt \
    ${trainers_dir}/train_CD_text_layer1.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/text_rbm1_LAST \
    ${trainers_dir}/train_CD_text_layer1.pbtxt text_hidden1 \
    ${data_output_dir}/text_rbm1_LAST ${gpu_mem} ${cpu_mem} || exit 1
fi

# IMAGE LAYER - 2.
if ${clobber} || [ ! -e ${model_output_dir}/image_rbm2_LAST ]; then
  echo "Training second layer image RBM."
  python ${trainer} ${models_dir}/image_rbm2.pbtxt \
    ${trainers_dir}/train_CD_image_layer2.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/image_rbm2_LAST \
    ${trainers_dir}/train_CD_image_layer2.pbtxt image_hidden2 \
    ${data_output_dir}/image_rbm2_LAST ${gpu_mem} ${cpu_mem} || exit 1
fi

# TEXT LAYER - 2.
if ${clobber} || [ ! -e ${model_output_dir}/text_rbm2_LAST ]; then
  echo "Training second layer text RBM."
  python ${trainer} ${models_dir}/text_rbm2.pbtxt \
    ${trainers_dir}/train_CD_text_layer2.pbtxt eval.pbtxt || exit 1
  python ${extract_rep} ${model_output_dir}/text_rbm2_LAST \
    ${trainers_dir}/train_CD_text_layer2.pbtxt text_hidden2 \
    ${data_output_dir}/text_rbm2_LAST ${gpu_mem} ${cpu_mem} || exit 1
fi
