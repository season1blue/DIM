#!bin/bash

export DIR_SEARCH="../../data/wikipedia/search_top100.json"
export PATH_ANS="../../data/wikipedia/qids_ordered.json"
export PATH_NEG_CONFIG="../../data/wikipedia/neg.json"
export DIR_PREPRO="../../data/wikipedia"
export DATASET="wiki"
export IMG_PATH="../../data/ImgData"
export GT_TYPE="brief"  #brief
export FEATURE_EXTRACTOR="clip"
export EPOCHS=300  #300
export LOGGING_STEPS=565 #565
export SAVE_STEPS=500 #1000  # 0 represent not save
export IMG_FEAT_SIZE=512
export TEXT_FEAT_SIZE=512
export LR=5e-5
export OUTPUT_SIZE=512



python3 ../nel_model/train.py \
--data_dir "../../data/wikipedia" \
--path_ans_list "../../data/wikipedia/qids_ordered.json" \
--dir_img_feat "../../data/wikipedia" \
--dir_neg_feat "../../data/wikipedia" \
--logging_steps $LOGGING_STEPS \
--save_steps $SAVE_STEPS \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--neg_sample_num 1 \
--model_type bert \
--num_train_epochs $EPOCHS \
--overwrite_output_dir \
--strip_accents \
--path_candidates $DIR_SEARCH \
--path_neg_config $PATH_NEG_CONFIG \
--num_attn_layers 2 \
--loss_scale 16 \
--loss_margin 0.5 \
--loss_function "triplet" \
--similarity "cos" \
--feat_cate "w" \
--learning_rate $LR \
--dropout 0.4 \
--weight_decay 0.001 \
--hidden_size 512 \
--nheaders 8 \
--ff_size 2048 \
--output_size $OUTPUT_SIZE \
--do_train \
--evaluate_during_training \
--gpu_id 0 \
--seed 114514 \
--max_sent_length 32 \
--img_feat_size $IMG_FEAT_SIZE \
--text_feat_size $TEXT_FEAT_SIZE \
--dataset $DATASET \
--img_path $IMG_PATH \
--gt_type $GT_TYPE \
--feature_extrator $FEATURE_EXTRACTOR \
# --overwrite_cache 