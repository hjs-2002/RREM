#!/bin/bash

MODEL_NAME=RREM
DEVICE_ID=0
EMBEDDING_SIZE=1024
ROOT_PATH=/home/law/HDD/Project_hjs/dataset_DRCT/MSCOCO/val2017,/home/law/HDD/Project_hjs/dataset_DRCT/images/stable-diffusion-inpainting/val2017 
FAKE_ROOT_PATH=/home/law/HDD/Project_hjs/dataset_DRCT/images,/home/law/HDD/Project_hjs/dataset_DRCT/fake_rec_images_all_final
DATASET_NAME=DRCT-2M
INPUT_SIZE=224
BATCH_SIZE=32

# --- 2. 设置权重目录 ---

WEIGHTS_DIR='RREM_drct/DRCT-2M/2/RREM_224/weights'

SAVE_TXT="./RREM_drct/results_final/results.txt"

mkdir -p $(dirname ${SAVE_TXT})

# --- 4. 【核心修改】遍历每一个 .pth 文件 ---
echo "开始批量特征空间可视化..."
echo "权重目录: ${WEIGHTS_DIR}"
echo "-----------------------------------------------------"

find "${WEIGHTS_DIR}" -name "best.pth" | while read MODEL_PATH; 
do
    # 提取权重文件名
    WEIGHT_NAME=$(basename "${MODEL_PATH}" .pth)
    
    echo ""
    echo "========= 开始处理权重: ${WEIGHT_NAME} =========="
    FAKE_INDEX="all_models" 

   
    LOG_FILE="./log_RREM_drct/plot_log_${WEIGHT_NAME}.log"
    mkdir -p ./log_RREM_drct
    echo "正在为权重 ${WEIGHT_NAME} 生成全模型特征图..."
    echo "日志: ${LOG_FILE}"

    nohup python test_DRCT_auto_thread.py --root_path ${ROOT_PATH} --fake_root_path ${FAKE_ROOT_PATH} --model_name ${MODEL_NAME} \
          --input_size ${INPUT_SIZE} --batch_size ${BATCH_SIZE} --device_id ${DEVICE_ID} --is_test --auto_threshold \
          --model_path ${MODEL_PATH} --is_crop --fake_indexes ${FAKE_INDEX}  \
          --dataset_name ${DATASET_NAME} --save_txt ${SAVE_TXT} --embedding_size ${EMBEDDING_SIZE} > ${LOG_FILE} 2>&1
    echo "权重 ${WEIGHT_NAME} 处理完成。"
    
done

echo "-----------------------------------------------------"
echo "所有任务结束。"
