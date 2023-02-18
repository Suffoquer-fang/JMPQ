
import os 
import json, ujson
import torch 
from tqdm import tqdm
import random 
import numpy as np 
from collections import OrderedDict
import faiss

def set_seed(args):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


   
def load_doclens(directory, flatten=True):
    parts = list(range(0, 64))
    doclens_filenames = [os.path.join(directory, 'doclens.{}.json'.format(filename)) for filename in parts]
    all_doclens = [ujson.load(open(filename)) for filename in doclens_filenames]

    if flatten:
        all_doclens = [x for sub_doclens in all_doclens for x in sub_doclens]

    return all_doclens

def load_queries(queries_path):
    queries = OrderedDict()

    print("#> Loading the queries from", queries_path, "...")

    with open(queries_path) as f:
        for line in f:
            qid, query, *_ = line.strip().split('\t')
            qid = int(qid)

            assert (qid not in queries), ("Query QID", qid, "is repeated!")
            queries[qid] = query

    print("#> Got", len(queries), "queries. All QIDs are unique.\n")

    return queries


def load_preprocess(args):
    print("#> Loading the preprocessed data..")
    emb2pid = torch.load(os.path.join(args.preprocess_dir, 'emb2pid.pt'))
    pid2offset = torch.load(os.path.join(args.preprocess_dir, 'pid2offset.pt'))
    return emb2pid, pid2offset


def build_centroids(index, train_centroid=True, train_coarse_embeds=False):
    pq = index.pq
    cen = faiss.vector_to_array(pq.centroids)
    cen = cen.reshape(pq.M, pq.ksub, pq.dsub)

    coarse_quantizer = faiss.downcast_index(index.quantizer)
    coarse_embeds = faiss.vector_to_array(coarse_quantizer.xb)
    coarse_embeds = coarse_embeds.reshape(-1, index.pq.M * index.pq.dsub)

    cen = torch.FloatTensor(cen).cuda()
    cen.requires_grad = train_centroid

    coarse_embeds = torch.FloatTensor(coarse_embeds).cuda()
    coarse_embeds.requires_grad = train_coarse_embeds

    return cen, coarse_embeds

def build_emb2pqcodes(index, num_embeddings):
    print("#> Building the emb2pqcodes mapping..")
    invlists = faiss.extract_index_ivf(index).invlists

    token_id2ivf_center_id = np.zeros(num_embeddings, dtype='int64')
    token_id2pq_codes = np.zeros((num_embeddings, invlists.code_size), dtype='int64')
    
    list_num = invlists.nlist
    # list_num = 1000
    for idx in tqdm(range(list_num)):
        ls = invlists.list_size(idx)
        list_ids = np.zeros(ls, dtype='int64')
        x = invlists.get_ids(idx)
        faiss.memcpy(faiss.swig_ptr(list_ids), x, list_ids.nbytes)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(idx), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)
        token_id2ivf_center_id[list_ids] = idx
        token_id2pq_codes[list_ids] = pq_codes[torch.arange(ls)]

    emb2ivf = token_id2ivf_center_id
    emb2pqcodes = token_id2pq_codes

    return emb2ivf, emb2pqcodes