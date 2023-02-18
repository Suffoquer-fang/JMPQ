# JMPQ

### Usage
Encode the corpus and compress the embeddings into faiss index with the [ColBERTv1 repo](https://github.com/stanford-futuredata/ColBERT/tree/colbertv1)
```shell
CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=6 \
python -m torch.distributed.launch --nproc_per_node=4 -m \
colbert.index --amp --doc_maxlen 180 --mask-punctuation --bsize 256 \
--checkpoint /root/to/experiments/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn \
--collection /path/to/MSMARCO/collection.tsv \
--index_root /root/to/indexes/ --index_name MSMARCO.L2.32x200k \
--root /root/to/experiments/ --experiment MSMARCO-psg
```
Faiss indexing
```
python -m colbert.index_faiss \
--index_root /root/to/indexes/ --index_name MSMARCO.L2.32x200k \
--partitions 32768 --sample 0.3 \
--root /root/to/experiments/ --experiment MSMARCO-psg
```

JMPQ Preprocess
```
python preprocess.py \
    --doclens_path MSMARCO.L2.32x200k \
    --preprocess_dir ./preprocess/ 
```

JMPQ Training
```
python run_train.py \
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
```

JMPQ Retrieval
```
python run_inference.py \
    --queries ./data/dataset/msmarco-test2019-queries.tsv \
    --inference_nprobe 32 \
    --faiss_depth 1024 \
    --index_path path/to/faiss/index \
    --model_path path/to/colbert/model \
    --doclens_path path/to/doclens \
    --preprocess_dir ./preprocess/ \
    --output_path ./output.trec
```





