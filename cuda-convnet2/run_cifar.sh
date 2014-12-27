data_path=/home/zdb/play/bowl/bowl_data/bowl_batches
save_path=/home/zdb/play/bowl2/trained_model
model=bowl_models/AlexNet_models.cfg
params=bowl_models/AlexNet_params.cfg

python convnet.py --data-path $data_path --save-path $save_path --test-range 100-100 --train-range 0-8 --layer-def $model --layer-params $params --data-provider cifar --test-freq 18 --inner-size 224 --gpu 0 --mini 128 --epochs 90

#MODEL=`ls -t $save_path | head -1`
#python convnet.py --load-file $save_path/$MODEL --epochs 90

MODEL=`ls -t $save_path | head -1`
feature_path=/home/zdb/play/bowl/test_probs
#python shownet.py -f $save_path/$MODEL --feature-path=$feature_path --write-features=probs --test-range=1000-1042 --train-range=1-1

#python shownet.py --load-file $save_path/$MODEL --show-cost=logprob
