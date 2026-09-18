# CMALA-Net
Cross-Modal Alignment with LoRA Adaptation Network for breast ultrasound + pathology subtyping and biomarker prediction.


├── data/
│   ├── us/          #  us_0001.png
│   └── path/        #  path_0001.png
├── train.csv
├── val.csv
├── test.csv       # Center B 测试集




python train.py --train_csv data/train.csv --val_csv data/val.csv --batch_size 8 --epochs 100 --lr 1e-4 --patience 50      

 python evaluate.py --checkpoint checkpoints/cmalanet_best.pth --test_csv D:\pythonProject\CMALA-Net\data\test.csv --us_dir data/ --path_dir data/ 
--dinov2_path ./models/dinov2-large
