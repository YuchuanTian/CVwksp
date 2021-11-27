models=( 'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152' )
lrs=( 0.00001 0.000001 )
for i in ${models[*]}
do
for j in ${lrs[*]}
do
python train.py --model ${i} --lr ${j} --gpu 1
done
done