#!/bin/bash
dir_dataset_1="C:\Users\william\Desktop\dataset\train"
dir_dataset_1_test="C:\Users\william\Desktop\dataset\test"
dir_out="./"
for type in  "Custom" #"Inception" "Xception" #"InceptionResNet" 
do
	for size in  380 #512 #256  128 
	do
		echo $type
		python train.py --model_name $type'_'$size'_model_binary' --path_train $dir_dataset_1'/'$size'/' --path_results $dir_out --batch 2 --size $size --classes 2 --epochs 125 --model $type --no_plot
		python inference.py --model_dir $dir_out'/models/'$type'_'$size'_model_binary.model' --path_test $dir_dataset_1_test'/'$size'/' --path_results "/share_gamma/william_temp/train_results" --batch 2 --size $size --save_report
		echo "finish"
	done 

done
