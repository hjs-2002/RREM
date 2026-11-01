#!/bin/bash

# --- 1. 配置基础参数 ---

MODEL_NAME=RREM

DEVICE_ID=1
EMBEDDING_SIZE=1024
ROOT_PATH=/home/law/Datasets/Dataset_hjs/DR/GenImage
DATASET_NAME=GenImage
INPUT_SIZE=224
BATCH_SIZE=32

# --- 2. 设置权重目录 ---

WEIGHTS_DIR="RREM_gen/GenImage/1/InCTRL_3_point_pair_224_drct_amp_crop/weights"

SAVE_TXT="./RREM_gen/results_final/results.txt"

mkdir -p $(dirname ${SAVE_TXT})

# --- 4. 【核心修改】遍历每一个 .pth 文件 ---
echo "开始批量特征空间可视化..."
echo "权重目录: ${WEIGHTS_DIR}"
echo "-----------------------------------------------------"

# 使用 find 命令查找所有 .pth 文件，并循环处理
find "${WEIGHTS_DIR}" -name "best.pth" | while read MODEL_PATH; 
do
    # 提取权重文件名
    WEIGHT_NAME=$(basename "${MODEL_PATH}" .pth)
    
    echo ""
    echo "========= 开始处理权重: ${WEIGHT_NAME} =========="

    FAKE_INDEX="all_models" 

    # 日志文件
    LOG_FILE="./RREM_gen/plot_log_${WEIGHT_NAME}.log"
    mkdir -p ./RREM_gen
    echo "正在为权重 ${WEIGHT_NAME} 生成全模型特征图..."
    echo "日志: ${LOG_FILE}"
    nohup python train_test_gen_con_mul_point_pair.py --root_path ${ROOT_PATH} --fake_root_path '' --model_name ${MODEL_NAME} \
          --input_size ${INPUT_SIZE} --batch_size ${BATCH_SIZE} --device_id ${DEVICE_ID} --is_test \
          --model_path ${MODEL_PATH} --is_crop --fake_indexes ${FAKE_INDEX} --inpainting_dir inpainting  \
          --dataset_name ${DATASET_NAME} --save_txt ${SAVE_TXT} --embedding_size ${EMBEDDING_SIZE} > ${LOG_FILE} 2>&1
    echo "权重 ${WEIGHT_NAME} 处理完成。"
    
done

echo "-----------------------------------------------------"
echo "所有任务结束。"
