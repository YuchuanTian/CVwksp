models=( 'vgg11' 'vgg11_bn' 'vgg13' 'vgg13_bn' 'vgg16' 'vgg16_bn' 'vgg19' 'vgg19_bn' )
lrs=( 0.01 0.001 0.0001 )
for i in ${models[*]}
do
for j in ${lrs[*]}
do
python train.py --model ${i} --lr ${j} --gpu 2
done
done