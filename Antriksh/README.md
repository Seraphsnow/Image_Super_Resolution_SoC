# Image_Super_Resolution_SoC
to download the dataset-https://www.dropbox.com/s/curldmdf11iqakd/91-image_x3.h5?dl=0
to test the code run the command -python test.py --weights-file "./outputs/srcnn_x3.pth" --image-file "./ppt3.bmp" --scale 3 
the weights are stored in the outputs folder
to train the code run-train.py --train-file ".\91-image_x2.h5"  --eval-file ".\Set5_x2.h5"  --outputs-dir ".\outputs" --scale 3  --lr 1e-4  --batch-size 16  --num-epochs 400 --num-workers 8  --seed 123
