import torch 
from tqdm import tqdm 
import argparse
import os 
import numpy as np 

from utils import load_doclens

def build_emb2pid(args):
    print("#> Building the emb2pid mapping..")
    all_doclens = load_doclens(args.doclens_path, flatten=True)
    total_num_embeddings = sum(all_doclens)
    emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

    offset_doclens = 0

    for pid, dlength in tqdm(enumerate(all_doclens)):
        emb2pid[offset_doclens: offset_doclens + dlength] = pid
        offset_doclens += dlength
    
    print("len(emb2pid) =", len(emb2pid))

    torch.save(emb2pid, os.path.join(args.preprocess_dir, 'emb2pid.pt'))

    return emb2pid


def build_pid2offset(args):
    print("#> Building the pid2offset mapping..")
    all_doclens = load_doclens(args.doclens_path, flatten=True)

    pid2offset = torch.zeros(len(all_doclens), dtype=torch.int)
    offset_doclens = 0
    for pid, dlength in tqdm(enumerate(all_doclens)):
        pid2offset[pid] = offset_doclens
        offset_doclens += dlength

    print("len(pid2offset) =", len(pid2offset))
    torch.save(pid2offset, os.path.join(args.preprocess_dir, 'pid2offset.pt'))

    return pid2offset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--preprocess_dir", type=str, default="preprocess")
    parser.add_argument("--doclens_path", type=str, default="")
    
    args = parser.parse_args()


    if not os.path.exists(args.preprocess_dir):
        os.makedirs(args.preprocess_dir)

    build_emb2pid(args)
    build_pid2offset(args)
    
    