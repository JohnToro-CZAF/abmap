import torch
import abmap
import argparse
import os
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--chain_type", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--plm_name", type=str, required=True) # beplerberger, esm2, esm1b, protbert
    parser.add_argument("--machine", type=str, required=True)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    sequence = args.sequence
    chain_type = args.chain_type
    output_dir = args.output_dir
    plm_name = args.plm_name
    machine = args.machine
    k = args.k

    # -------- Load AbMAP 
    # Using Bepler-Berger as foundational model (best for structure prediction)
    if plm_name == 'beplerberger':
        abmap_H = abmap.load_abmap(pretrained_path='./pretrained_models/AbMAP_beplerberger_H.pt', plm_name='beplerberger')
        abmap_L = abmap.load_abmap(pretrained_path='./pretrained_models/AbMAP_beplerberger_L.pt', plm_name='beplerberger')
    elif plm_name == 'esm2':
        abmap_H = abmap.load_abmap(pretrained_path='./pretrained_models/AbMAP_esm2_H.pt', plm_name='esm2')
        abmap_L = abmap.load_abmap(pretrained_path='./pretrained_models/AbMAP_esm2_L.pt', plm_name='esm2')
    elif plm_name == 'esm1b':
        abmap_H = abmap.load_abmap(pretrained_path='./pretrained_models/AbMAP_esm1b_H.pt', plm_name='esm1b')
        abmap_L = abmap.load_abmap(pretrained_path='./pretrained_models/AbMAP_esm1b_L.pt', plm_name='esm1b')
    elif plm_name == 'protbert':
        abmap_H = abmap.load_abmap(pretrained_path='./pretrained_models/AbMAP_protbert_H.pt', plm_name='protbert')
        abmap_L = abmap.load_abmap(pretrained_path='./pretrained_models/AbMAP_protbert_L.pt', plm_name='protbert')

    # ----- Get embedding for one sequence (steps 1-3)
    demo_seq = sequence

    # Contrastive augmentation (PLM, mutagenesis, CDR focus)
    if machine == 'cpu':
        embed_device = torch.device('cpu')
    elif machine == 'gpu':
        embed_device = torch.device('cuda')

    x = abmap.ProteinEmbedding(demo_seq, chain_type=chain_type, embed_device=embed_device) # push sequence through foundational model
    x.create_cdr_specific_embedding(embed_type=plm_name, k=k) # decrease k to speed up, would recommend k >= 6

    # # Pass the augmented embedding to AbMAP to get final embedding
    with torch.no_grad():
        if chain_type == 'H':
            embed_var = abmap_H.embed(x.cdr_embedding.unsqueeze(0), embed_type='variable') # residue-level embeddings
            embed_fl = abmap_H.embed(x.cdr_embedding.unsqueeze(0), embed_type='fixed') # fixed-length
        elif chain_type == 'L':
            embed_var = abmap_L.embed(x.cdr_embedding.unsqueeze(0), embed_type='variable') # residue-level embeddings
            embed_fl = abmap_L.embed(x.cdr_embedding.unsqueeze(0), embed_type='fixed') # fixed-length
        
        with open(os.path.join(output_dir, f'{chain_type}_AbMAP.pkl'), 'wb') as f:
            pickle.dump(embed_fl.cpu(), f)