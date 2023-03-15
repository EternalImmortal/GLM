DATA_ROOT=/data/text2music/cnn-dailymail
CHECKPOINT_PATH=/data/text2music/glm-10b-chinese.zip
SAVE_PATH=/data/text2music/glm-checkpoint
DATESTR=$(date +"%m-%d-%H-%M")

source $1    # Model
source $2    # Task

if [ -z $N_GPU ];then
  N_GPU=4
fi

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
DISTRIBUTED_ARGS="--nproc_per_node ${N_GPU} --nnodes 1 --node_rank 0 --master_addr localhost --master_port $MASTER_PORT"

DATESTR=$(date +"%m-%d-%H-%M")
EXPERIMENT_NAME=${EXPERIMENT_NAME}  #-${DATESTR}

TOKENIZERS_PARALLELISM=false

mkdir logs
python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_glm.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${SAVE_PATH} \
       --checkpoint-activations \
       --epochs 1 \
       --batch-size 16 \
       --lr 0.001 \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       $TASK_ARGS \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}.txt

#mkdir logs
#python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_glm.py \
#       --finetune \
#       --experiment-name ${EXPERIMENT_NAME} \
#       --task ${TASK_NAME} \
#       --data-dir ${DATA_PATH} \
#       --save ${SAVE_PATH} \
#       --checkpoint-activations \
#       --epochs ${EPOCH_SINGLE} \
#       --batch-size ${BATCH_SINGLE} \
#       --lr ${LR_SINGLE} \
#       $MODEL_ARGS \
#       $TRAIN_ARGS \
#       $COMMON_ARGS \
#       $TASK_ARGS \
#       2>&1 | tee logs/log-${EXPERIMENT_NAME}.txt