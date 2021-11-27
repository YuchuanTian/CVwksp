models=( 'vit_base_patch16_224' 'vit_base_patch32_224' 'vit_large_patch16_224' 'vit_large_patch32_224' 'vit_small_patch16_224' 'vit_small_patch32_224' 'tnt_b_patch16_224' 'tnt_s_patch16_224' )
lrs=( 0.01 0.001 0.0001 )
for i in ${models[*]}
do
for j in ${lrs[*]}
do
python train.py --model ${i} --lr ${j} --gpu 3
done
done