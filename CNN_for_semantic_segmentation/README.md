## Train code

```bash
run Train_Seg.py --id_gpu 1 --size 256 --stride 0.06 --classes 2 --csv_path "./Patchs_example" --model_name "Unet" --model_weigths_dir "./Models/example1.h5" --epochs 50 --Flip_x True --Flip_y True --batch 16
```
## Inference code

```bash
inference.py --id_gpu 0 --size 256  --classes 2 --model_name "Unet" --model_weigths_dir "./Models/example1.h5" --out_path_inf '../inf/unet' --save_plot True --limiar 0.07 
```

## SAR Images of scenes:

https://drive.google.com/drive/folders/1rAjSq4FmnA8nQJ2TUNL9c4lk-T_nTNay?usp=sharing
