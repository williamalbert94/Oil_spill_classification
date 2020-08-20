# Oil_spill_classification
 Oill spill study codes in gulf of mexico. It contains codes related to CNN architectures, based on classification and semantic segmentation.  Classification: VGG16, ResNet, Inception, Xception, Inception-ResNet. Segmentation: SegNet, Unet, Deeplab v3.
 
 # Run training code 
 		python train.py --model_name 'example1' --path_train './train/' --path_results ./results/ --batch 2 --size 256 --classes 2 --epochs 50 --model Xception --no_plot

Parameters:
model_name : name to save model
path_train : path of train data
path_results : path to save model and train graph
batch : batch size for generator
size : size of input of model and out of generator
classes : number of classes (this case 2 or 5)
epochs : epochs for training
model : Model to use (Xcetion Inception InceptionResNet VGG16 ResNet50 MobileNetV2)
no_plot : Use to save training graph

 # Run inference code
 
 		python inference.py --model_dir ./path/example1.model' --path_test './test/' --path_results ./results" --batch 2 --size 256 --save_report
   
Parameters:
model_dir : full path model to use
path_test : path of test data
path_results : path to save confusion matrix and report metrics
batch : batch size for generator
size : size of input of model and out of generator
save_report : save cvs to report metrics
