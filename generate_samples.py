''' SAMPLING PIPELINE '''

## IMPORTING UTILITIES
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import glob
import os
import torchvision
from torchvision.utils import make_grid
from torch.optim import Adam, AdamW
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import argparse
import subprocess
import json
import numpy as np
import pickle
from biotite.sequence.io.fasta import FastaFile, get_sequences
from pathlib import Path

from utilities.model import *
from utilities.noise_scheduling import *
import utilities.angles_and_coords as ac
from utilities.ss_annot_pipeline import *
from utilities.nerf_fd import *


def wrap_angle(x):
    pi = torch.tensor(np.pi, dtype=x.dtype, device=x.device)
    return ((x+pi) % (2*pi)) - pi

def wrapped_smooth_l1_loss(pred, target, mask, beta=0.1):
    """
    pred, target: [batch, 6, 64] – should be angle residuals
    beta: smooth L1 transition point (standard beta=1.0)
    """
    diff = wrap_angle(pred - target)
    abs_diff = torch.abs(diff)
    loss = torch.where(
        abs_diff < beta,
        0.5 * (diff ** 2) / beta,
        abs_diff - 0.5 * beta
    )
    masked_loss = loss * mask  # mask is 1.0 where valid, 0.0 where invalid
    return masked_loss.sum() / mask.sum()  # Average over valid positions only



def sample_from_model(model_path, noise_schedule,config,num_samples=1, seq_len=64, channels=6, device='cuda',
                      scaling=False, center=False, movie_folder=None, save_interval=10):
    """
    Sample from a trained diffusion model and optionally save intermediate timesteps.

    Args:
        model_path (str): Path to trained model checkpoint (.pth)
        num_samples (int): Number of samples to generate
        seq_len (int): Sequence length (e.g., 64 residues)
        channels (int): Number of angular channels (e.g., 6: phi, psi, omega, θ1-3)
        device (str): 'cuda' or 'cpu'
        scaling (bool): If True, multiply outputs by π (inverse of scaling used during training)
        center (bool): If True, add back the mean vector `mu`
        movie_folder (str): Folder to save intermediate x_t arrays for visualization
        save_interval (int): Save every N timesteps (default: 10)
    """
    # ---- Load configs ----
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']

    scheduler = NoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        schedule=noise_schedule,
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end'],
        device=device
    )

    # ---- Create movie output directory if needed ----
    if movie_folder is not None:
        os.makedirs(movie_folder, exist_ok=True)

    # ---- Load model ----
    model = UNet(model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ---- Initialize starting noise ----
    xt = torch.randn(num_samples, channels, seq_len, device=device)

    # ---- Sampling loop ----
    all_timesteps = []  # Will hold intermediate results if saving enabled

    with torch.no_grad():
        for t in tqdm(reversed(range(diffusion_config['num_timesteps'])), desc="Sampling"):
            t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)
            noise_pred = model(xt, t_tensor)
            xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, t)

            # Save intermediate states
            if movie_folder is not None and (t % save_interval == 0 or t == 0):
                samples_xt = xt.detach().cpu().numpy()
                if scaling:
                    samples_xt = samples_xt * np.pi
                if center:
                    samples_xt = samples_xt + mu[:, None, None]

                samples_xt = ((samples_xt + np.pi) % (2 * np.pi)) - np.pi
                
                np.save(
                    os.path.join(movie_folder, f"x_t_{t:04d}.npy"),
                    samples_xt
                )

            # Optional: store in-memory (comment out if memory is limited)
            # all_timesteps.append(xt.detach().cpu().numpy())

    # ---- Final denoised sample ----
    samples = xt.cpu().numpy()  # [num_samples, 6, 64]

    # ---- Undo transformations if needed ----
    if scaling:
        samples = samples * np.pi
    if center:
        samples = samples + mu[:, None, None]  # mu defined outside, shape [6]

    # ---- Wrap to [-π, π] ----
    samples_w = ((samples + np.pi) % (2 * np.pi)) - np.pi

    return samples_w, samples  # optionally return all_timesteps if stored





def get_args():
    parser = argparse.ArgumentParser(description="Run FUN sampling and generation")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="default_fd2/ddpm_foldingdiff1.pth",
        help="Path to model checkpoint (.pth file)"
    )

    parser.add_argument(
        '--noise_schedule',
        type=str,
        default='linear',
        help='choose the noise schedule based on the model in use'
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to generate"
    )

    parser.add_argument(
        "--movie_folder",
        type=str,
        default="fd2_outputs/movies",
        help="Output folder for movie frames"
    )

    parser.add_argument(
        "--out_folder",
        type=str,
        default="fd2_outputs/final",
        help="Final output folder for generated structures"
    )

    parser.add_argument(
        "--n_sequences",
        type=int,
        default=2,
        help="Number of sequences to generate per sample"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="ProteinMPNN Sampling temperature (controls randomness)"
    )

    parser.add_argument(
        '--config_file',
        type=str,
        default='config.json',
        help='Config file to determine all the training parameters'
    )

    return parser.parse_args()




if __name__ == '__main__':

    # STEP-1 :: GENERATING SAMPLES FROM THE MODEL

    args = get_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)
    
    ckpt_path = args.ckpt_path
    noise_schedule = args.noise_schedule
    num_samples = args.num_samples
    movie_folder = args.movie_folder
    out_folder = args.out_folder
    n_sequences = args.n_sequences
    temperature = args.temperature


    print('Generating Samples !')

    os.makedirs(movie_folder, exist_ok=True)
    generated_angles, raw_angles = sample_from_model(
        ckpt_path,
        noise_schedule,
        config,
        num_samples=num_samples,
        scaling=True,
        center=False,
        movie_folder=movie_folder,
        save_interval=5  # saves every 5 timesteps
    )

    # Save final samples
    os.makedirs(out_folder, exist_ok=True)

    pbar = tqdm(range(num_samples))

    for i in pbar:
        sample_name = f'generated_sample_{i}'
        
        np.save(os.path.join(out_folder,f'{sample_name}.npy'),
                generated_angles[i].transpose(1, 0))  # [64, 6]
    


    # STEP-2 :: ANNOTATE GENERATED STRUCTURE USING NERF/PSEA
        pbar.set_description(f'Annotating :: {sample_name}')
        an = np.load(os.path.join(out_folder,f'{sample_name}.npy'))

        out_coords = os.path.join(out_folder,f'{sample_name}_coords.npy')
        out_pdb = os.path.join(out_folder,f'{sample_name}.pdb')
        out_sse = os.path.join(out_folder,f'{sample_name}_sse.npy')

        phi, psi, omega = an[:,0],an[:,1],an[:,2]

        builder = NERFBuilder(phi, psi, omega)
        cd = builder.cartesian_coords
        # print(builder.cartesian_coords)
        # print(builder.cartesian_coords.shape)
        
        np.save(out_coords,cd)
        write_backbone_pdb(cd,out_pdb)
        
        source = PDBFile.read(out_pdb)
        # expect single model 
        assert source.get_model_count() == 1
        atom_array = source.get_structure()[0]

        # Compute SSE (note: annotate_sse expects an AtomArray only)
        sse = struc.annotate_sse(atom_array)
        # print(sse)
        np.save(out_sse,np.array(sse))

        annot = add_secondary_structure_to_pdb(out_pdb,'annot.pdb')
        # print(annot)



    # STEP-3 :: RUNNING PROTEIN-MPNN
        pbar.set_description(f'Running ProteinMPNN :: {sample_name}')
        PROTEINMPNN_SCRIPT = os.path.expanduser("ProteinMPNN/protein_mpnn_run.py")   
        assert os.path.isfile(PROTEINMPNN_SCRIPT), f"Expected {PROTEINMPNN_SCRIPT} to exist"

        mpnn_dir = os.path.join(out_folder,'mpnn_seqs/')
        os.makedirs(mpnn_dir,exist_ok=True)

        cmd = f'python {PROTEINMPNN_SCRIPT} --pdb_path_chains A --out_folder {Path(mpnn_dir).as_posix()} --num_seq_per_target {n_sequences} --seed 1234 --batch_size {n_sequences} --pdb_path {Path(out_pdb).as_posix()} --sampling_temp "{temperature}" --ca_only --path_to_model_weights ProteinMPNN/ca_model_weights/'
        #print(cmd)
        retval = subprocess.call(
            cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        #retval = subprocess.call(cmd, shell=True)
        #print(f"PMPNN exited with code: {retval}")

        
        #print(f'PMPNN : {retval}')
        

    # STEP-4 :: RECONSTRUCTING COMPLETE ALL RESIDUE STRUCTURE
        pbar.set_description(f'Reconstructing :: {sample_name}')
        fasta_path = os.path.join(mpnn_dir,f'seqs/{sample_name}.fa')

        # Read the FASTA file
        fasta_file = FastaFile.read(fasta_path)

        # Extract all sequences as Sequence objects
        sequences = get_sequences(fasta_file)

        # Print sequences and their IDs
        #print(list(sequences.items()))
        for i,(name, seq) in enumerate(list(sequences.items())):
            if i==1:
                ac.add_sidechains_to_backbone('annot.pdb',seq,out_pdb)

                annot = add_secondary_structure_to_pdb(out_pdb,out_pdb)
                #print(f'OUPUTPATH : {annot},{i}')
            else:
                continue
            
        







