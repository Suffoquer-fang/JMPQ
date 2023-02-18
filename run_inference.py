from model import load_model

import faiss
import torch
import argparse
import os
import time
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
    RobertaConfig)
from tqdm import trange, tqdm
from utils import set_seed, load_doclens, load_queries, load_preprocess, build_centroids, build_emb2pqcodes
from torch.utils.data import Dataset
from collections import defaultdict
import torch.nn as nn



def get_score(Q, pids, cen, coarse_embeds, pid2offset, emb2ivf, emb2pqcodes, all_doclens):
    s = time.time()
    term_per_query, qdim = Q.size(0), Q.size(1)
    M = cen.shape[0]
    d = cen.shape[2]

    doclens = all_doclens[pids]
    doc_offsets = pid2offset[pids]

    max_len = max(doclens)
    embs = torch.arange(max_len).expand(len(pids), max_len)

    max_tensor = max_len - doclens
    max_tensor = max_tensor.unsqueeze(1).expand(len(pids), max_len)
    
    embs = embs - max_tensor
    embs = torch.clamp(embs, 0) + doc_offsets.unsqueeze(1).expand(len(pids), max_len)

    embs, inv = torch.unique_consecutive(embs, return_inverse=True)
    tot_num = len(embs)

    center_ids = emb2ivf[embs].reshape(-1)
    pq_code_ids = emb2pqcodes[embs].reshape(-1)

    center_embeds = coarse_embeds[center_ids].reshape(tot_num, -1)
    center_scores = Q @ center_embeds.permute(1, 0)

    Q = Q.reshape(term_per_query, M, 1, -1)
    centroid_scores = (Q * cen.unsqueeze(0)).sum(-1).reshape(term_per_query, M, -1, 1)
    
    first_indices = torch.arange(M)
    first_indices = first_indices.expand(tot_num, M).reshape(-1)
    
    second_indices = pq_code_ids
    
    scores = centroid_scores[:, first_indices, second_indices].reshape(term_per_query, tot_num, -1).sum(2) + center_scores
    scores = scores[:, inv]
    score_q = scores.reshape(term_per_query, len(pids), -1)
    
    score_q = score_q.permute(1, 0, 2)
    score_q = score_q.max(2).values.sum(1).cpu()

    print('rebuild time', (time.time() - s) * 1000.0, 'ms')
    
    return score_q


def inference(args):
    print("#> Building queries..")
    args.queries = load_queries(args.queries)

    # build model
    print("#> Building the model..")
    query_encoder = load_model(args)
    query_encoder.colbert.eval()

    # build index
    print("#> Building the index..")
    index = faiss.read_index(args.index_path)
    index.nprobe = args.inference_nprobe

    emb2pid, pid2offset = load_preprocess(args)
    centroids, coarse_embeds = build_centroids(index, False, False)
    emb2ivf, emb2pqcodes = build_emb2pqcodes(index, len(emb2pid))

    all_doclens = load_doclens(args.doclens_path, flatten=True)
    all_doclens = torch.tensor(all_doclens)

    # inference
    print("#> Inference..")
    qids_in_order = list(args.queries.keys())

    start_time = time.time()
    all_time = 0.0


    f = open(args.output_path, 'w')
    with torch.no_grad():
        for idx, qid in enumerate(qids_in_order):
            s = time.time()
            query = args.queries[qid]
            Q = query_encoder.queryFromText([query])
            num_queries, embeddings_per_query, dim = Q.size()
            Q_faiss = Q.view(num_queries * embeddings_per_query, dim).cpu().contiguous()
            Q_faiss = Q_faiss.cpu().contiguous().float().detach().numpy()
            dist, embedding_ids = index.search(Q_faiss, args.faiss_depth)

            embedding_ids = embedding_ids.reshape(-1)
                
            all_pids = emb2pid[embedding_ids]
            all_pids = torch.unique(all_pids).cpu().numpy()

            score_q = get_score(Q[0], all_pids, centroids, coarse_embeds, pid2offset, emb2ivf, emb2pqcodes, all_doclens)
            scores_sorter = torch.tensor(score_q).cpu().sort(descending=True)
            pids, scores = torch.tensor(all_pids).cpu()[scores_sorter.indices].tolist(), scores_sorter.values.tolist()
            
            print('doing', idx, query, ' time', (time.time() - s) * 1000.0, 'ms')
            all_time += time.time() - s
            
            pids = pids[:1000]
            scores = scores[:1000]
            for rank, pid in enumerate(pids):
                f.write(str(qid) + ' Q0 ' + str(pids[rank]) + ' ' + str(rank+1) + ' ' + str(scores[rank]) + ' jpq\n')
    
    print('total_time:', (time.time() - start_time) * 1000.0 / len(qids_in_order))
    print('all_time:', all_time * 1000.0 / len(qids_in_order))
        
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--query_maxlen", type=int, default=32)
    parser.add_argument("--doc_maxlen", type=int, default=180)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--index_path", type=str, default="ivfpq.32768.faiss")
    parser.add_argument("--doclens_path", type=str, default="")
    parser.add_argument("--preprocess_dir", type=str, default="./preprocess_data/")
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--inference_nprobe", type=int, default=32)
    parser.add_argument("--faiss_depth", type=int, default=1024)
    parser.add_argument("--output_path", type=str, default="./output.trec")

    args = parser.parse_args()

    inference(args)



    