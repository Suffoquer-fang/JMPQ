python ../run_inference.py \
    --queries ./data/dataset/msmarco-test2019-queries.tsv \
    --inference_nprobe 32 \
    --faiss_depth 1024 \
    --index_path path/to/faiss/index \
    --model_path path/to/colbert/model \
    --doclens_path path/to/doclens \
    --preprocess_dir ./preprocess/ \
    --output_path ./output.trec
