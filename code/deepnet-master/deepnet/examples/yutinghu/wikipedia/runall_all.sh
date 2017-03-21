root_dir=$(cd `dirname $0`; pwd)
echo ${root_dir}
src=${root_dir}/rbm

for i in 1024
do
  
  now_dir=${root_dir}/rbm_${i}
  
  echo ${i}
  cp -r ${src} ${now_dir} || exit 1
  
  python scripts/change_hidden.py ${i} ${now_dir} || exit 1
  cd ${now_dir}

#  echo "rbm"
#  sh runall_rbm.sh || exit 1
  
#  echo "ae"
#  sh runall_ae.sh || exit 1
  
  echo "corr_ae"
  sh runall_corr_ae.sh ${now_dir} || exit 1
  
#  echo "multimodal_rbm"
#  sh runall_multimodal_rbm.sh || exit 1
  
#  echo "bae"
#  sh runall_bae.sh || exit 1
  
  cd ${root_dir}
done



