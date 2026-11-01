
nohup python train_test_gen.py --root_path /home/law/Datasets/Dataset_hjs/DR/GenImage \
                            --fake_root_path /home/law/Datasets/Dataset_hjs/DR/GenImage\
                            --dataset_name GenImage \
                            --model_name RREM \
                            --embedding_size 1024 \
                            --input_size 224 \
                            --batch_size 256 \
                            --alpha 0.3 \
                            --fake_indexes 1 \
                            --num_epochs 4 \
                            --device_id 0\
                            --lr 0.00002 \
                            --is_amp \
                            --is_warmup \
                            --is_crop \
                            --num_workers 12 \
                            --inpainting_dir inpainting \
                            --save_flag _drct_amp_crop > rrem_gen_amp_crop.log 2>&1


