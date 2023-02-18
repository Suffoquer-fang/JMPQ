python ../run_train.py \
    --doc_maxlen 180 \
    --queries ./data/dataset/queries.train.tsv \
    --qrels ./data/dataset/qrels.train.tsv \
    --training_nprobe 32 \
    --train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --index_path path/to/faiss/index \
    --model_path path/to/colbert/model \
    --preprocess_dir ./preprocess/ \
    --centroid_lr 5e-6 \
    --learning_rate 5e-6 \
    --num_train_epochs 5 
