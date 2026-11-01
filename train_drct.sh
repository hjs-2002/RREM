
nohup python train_test_DRCT.py --root_path /home/law/Datasets/Dataset_hjs/Dataset_hjs/dataset_DRCT/MSCOCO/train2017/,/home/law/Datasets/Dataset_hjs/Dataset_hjs/dataset_DRCT/images/stable-diffusion-inpainting/train2017/ \
                            --fake_root_path /home/law/Datasets/Dataset_hjs/Dataset_hjs/dataset_DRCT/images/stable-diffusion-v1-4/train2017/,/home/law/Datasets/Dataset_hjs/Dataset_hjs/dataset_DRCT/fake_rec_images/stable-diffusion-v1-4/train2017/ \
                            --dataset_name DRCT-2M \
                            --model_name RREM \
                            --embedding_size 1024 \
                            --input_size 224 \
                            --batch_size 256 \
                            --fake_indexes 2 \
                            --num_epochs 3 \
                            --alpha 0.5 \
                            --device_id 1 \
                            --lr 0.00002 \
                            --is_crop \
                            --num_workers 12 \
                            --is_amp \
                            --is_warmup \
                            --save_flag _drct_amp_crop > rrem_drct_amp_crop.log 2>&1




