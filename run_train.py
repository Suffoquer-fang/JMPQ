from model import load_model

import faiss
import torch
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
    RobertaConfig)
from tqdm import trange, tqdm
from utils import set_seed, load_doclens, load_queries, load_preprocess, build_centroids, build_emb2pqcodes
from torch.utils.data import Dataset
from collections import defaultdict
import torch.nn as nn



class TrainQueryDataset(Dataset):
    def __init__(self, rel_file):
        self.data_points = []
        self.reldict = defaultdict(list)
        for line in tqdm(open(rel_file), desc=os.path.split(rel_file)[1]):
            qid, _, pid, _ = line.split()
            qid, pid = int(qid), int(pid)
            # self.data_points.append((qid, pid))
            self.data_points.append(qid)
            self.reldict[qid].append((pid))

    def __getitem__(self, item):
        ret_val = {}
        qid = self.data_points[item]
        ret_val['qid'] = qid
        ret_val['rel_pids'] = self.reldict[qid]
        return ret_val

    def __len__(self):
        return len(self.data_points)

def get_collate_function():
    def collate_function(batch):
        qids = [x['qid'] for x in batch]
        rel_pids = [x['rel_pids'] for x in batch]
        return qids, rel_pids
    return collate_function


def rebuild_passage_embedding(pids, cen, coarse_embeds, pid2offset, emb2ivf, emb2pqcodes, all_doclens):
    M = cen.shape[0]
    d = cen.shape[2]
    
    doc_offsets = pid2offset[pids]

    doclens = all_doclens[pids]

    max_len = max(doclens)
    embs = torch.arange(max_len).expand(len(pids), max_len)

    max_tensor = max_len - doclens
    max_tensor = max_tensor.unsqueeze(1).expand(len(pids), max_len)
    
    embs = embs - max_tensor
    embs = torch.clamp(embs, 0) + doc_offsets.unsqueeze(1).expand(len(pids), max_len)

    max_doclen = embs.shape[1]
    tot_num = len(pids) * embs.shape[1]
    embs = embs.reshape(-1)

    center_ids = emb2ivf[embs].reshape(-1)
    pq_code_ids = emb2pqcodes[embs].reshape(-1)

    center_embeds = coarse_embeds[center_ids].reshape(tot_num, -1)

    first_indices = torch.arange(M)
    first_indices = first_indices.expand(tot_num, M).reshape(-1)
    
    second_indices = pq_code_ids
    
    embeddings = cen[first_indices, second_indices].reshape(tot_num, -1) + center_embeds
    
    embeddings = embeddings.reshape(len(pids), max_doclen, -1).cuda()
    
    return embeddings


def compute_loss(batch_Q, batch_neighbors, batch_rel_pids, centroids, coarse_embeds, pid2offset, emb2ivf, emb2pqcodes, all_doclens, negative_num=255):
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    train_batch_size = len(batch_Q)
    scores = []
    training_mrr = 0.0

    for Q, neighbors, rel_pids in zip(batch_Q, batch_neighbors, batch_rel_pids):
        rel_pids = torch.LongTensor(rel_pids)
        neighbors = torch.tensor(neighbors)

        target_labels = (neighbors[:, None]==rel_pids).any(-1)
        retrieved_rel_pids = neighbors[target_labels]
        
        retrieved_not_rel_pids = neighbors[~target_labels]
        
        positive_pids = np.random.choice(rel_pids, 1)
        negative_pids = np.random.choice(retrieved_not_rel_pids, negative_num)
        all_pids = np.hstack([positive_pids, negative_pids])

        

        D = rebuild_passage_embedding(all_pids, centroids, coarse_embeds, pid2offset, emb2ivf, emb2pqcodes, all_doclens)
        score = (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        scores.append(score)

        # if len(retrieved_rel_pids) > 0:
        #     scores_sorter = torch.tensor(score).cpu().sort(descending=True)
        #     pids = torch.tensor(all_pids).cpu()[scores_sorter.indices]

        #     mrr_target_labels = (pids[:, None]==rel_pids).any(-1)
        #     position_matrix = (1+torch.arange(len(mrr_target_labels)))
        #     first_rel_pos = position_matrix[mrr_target_labels][0].item()
        #     if first_rel_pos <= 10:
        #         training_mrr += 1/first_rel_pos 
        
    scores = torch.vstack(scores)
    labels = torch.zeros(scores.shape[0], dtype=torch.long).cuda()

    loss = loss_fn(scores, labels)
    mrr = training_mrr / train_batch_size

    return loss, mrr


def train(args):
    #build queries
    print("#> Building queries..")
    args.queries = load_queries(args.queries)

    #build dataset
    print("#> Building the dataset..")
    train_dataset = TrainQueryDataset(args.qrels)

    train_sampler = RandomSampler(train_dataset)
    collate_function = get_collate_function()

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_function)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    # build model
    print("#> Building the model..")
    query_encoder = load_model(args)
    query_encoder.colbert.train()

    # build index
    print("#> Building the index..")
    index = faiss.read_index(args.index_path)
    index.nprobe = args.training_nprobe

    emb2pid, pid2offset = load_preprocess(args)
    centroids, coarse_embeds = build_centroids(index)
    emb2ivf, emb2pqcodes = build_emb2pqcodes(index, len(emb2pid))

    # build doclens
    print("#> Building the doclens..")
    all_doclens = load_doclens(args.doclens_path, flatten=True)
    all_doclens = torch.tensor(all_doclens)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in query_encoder.colbert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in query_encoder.colbert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [centroids], 'weight_decay': args.centroid_weight_decay, 'lr': args.centroid_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_mrr, logging_mrr = 0.0, 0.0

    optimizer.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)

    for epoch_idx, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            qids, rel_pids = batch 

            queries = [args.queries[qid] for qid in qids]

            Q = query_encoder.queryFromText(queries)
            num_queries, embeddings_per_query, dim = Q.size()

            Q_faiss = Q.view(num_queries * embeddings_per_query, dim).cpu().contiguous()
            Q_faiss = Q_faiss.cpu().contiguous().float().detach().numpy()
            dist, embedding_ids = index.search(Q_faiss, 256)

            embedding_ids = embedding_ids.reshape(-1)
            all_pids = emb2pid[embedding_ids]
            all_pids = all_pids.reshape(num_queries, -1)
            embedding_ids = embedding_ids.reshape(num_queries, -1)

            batch_neighbors = []

            for pids in all_pids:
                pids = pids.tolist()
                pids = list(set(pids))
                batch_neighbors.append(pids)
            
            loss, mrr = compute_loss(Q, batch_neighbors, rel_pids, centroids, coarse_embeds, pid2offset, emb2ivf, emb2pqcodes, all_doclens, 127)

            loss /= args.gradient_accumulation_steps
            loss.backward()

            torch.nn.utils.clip_grad_norm_(query_encoder.colbert.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_mrr += mrr 

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                #copy faiss index 
                faiss.copy_array_to_vector(
                    centroids.detach().cpu().numpy().ravel(), 
                    index.pq.centroids)

                coarse_quantizer = faiss.downcast_index(index.quantizer)
                temp = faiss.vector_to_array(coarse_quantizer.xb)
                coarse_embeds_tmp = coarse_embeds.reshape(temp.shape)

                faiss.copy_array_to_vector(
                    coarse_embeds_tmp.detach().cpu().numpy().ravel(), 
                    coarse_quantizer.xb)

                # logging
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    cur_mrr = (tr_mrr - logging_mrr) / (args.logging_steps * args.gradient_accumulation_steps)
                    cur_loss = (tr_loss - logging_loss) / (args.logging_steps * args.gradient_accumulation_steps)

                    logging_mrr = tr_mrr
                    logging_loss = tr_loss

                    print("Step: {}/{}".format(global_step, t_total))
                    print("Loss: {}".format(cur_loss))
                    print("MRR: {}".format(cur_mrr))

                # if global_step % 2000 == 0:
                #     query_encoder.colbert.save_pretrained(os.path.join(args.model_output_dir, f"checkpoint-step-{global_step}"))
                #     faiss.write_index(index, os.path.join(args.index_output_dir, f"step-{global_step}.faiss"))



        query_encoder.colbert.save_pretrained(os.path.join(args.model_output_dir, f"checkpoint-epoch-{epoch_idx}"))
        faiss.write_index(index, os.path.join(args.index_output_dir, f"epoch-{epoch_idx}.faiss"))

            
                    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--query_maxlen", type=int, default=32)
    parser.add_argument("--doc_maxlen", type=int, default=180)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--index_path", type=str, default="")
    parser.add_argument("--doclens_path", type=str, default="")
    parser.add_argument("--model_output_dir", type=str, default="./output/model/")
    parser.add_argument("--index_output_dir", type=str, default="./output/index/")
    parser.add_argument("--preprocess_dir", type=str, default="./preprocess_data/")
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--qrels", type=str, required=True)
    parser.add_argument("--training_nprobe", type=int, default=32)


    
    parser.add_argument("--centroid_lr", type=float, required=True)
    parser.add_argument("--centroid_weight_decay", type=float, default=0)

    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--logging_steps", type=int, default=100)

    parser.add_argument("--learning_rate", default=5e-6, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)


    args = parser.parse_args()

    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)
    
    if not os.path.exists(args.index_output_dir):
        os.makedirs(args.index_output_dir)

    train(args)




    