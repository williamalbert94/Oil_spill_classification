#!/bin/bash
dir_dataset_1="/share_gamma/william_temp/dataset/train"
dir_dataset_1_test="/share_gamma/william_temp/dataset/test"
dir_out="/share_gamma/william_temp/train_results/"
for type in "Xception" "Inception" "InceptionResNet"
do
	for size in 128 256 380 512
	do
		echo $type
		python train.py --model_name $type'_model_binary_' --path_train $dir_dataset_1'/'$size'/' --path_results $dir_out --batch 2 --size $size --classes 2 --epochs 50 --model $type --no_plot
		python inference.py --model_dir $dir_out'/models/'$type'_model_binary_.model' --path_test $dir_dataset_1_test'/'$size'/' --path_results "/share_gamma/william_temp/train_results" --batch 2 --size $size --save_report
		echo "finish"
	done 

done
