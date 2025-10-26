import pymol
import sys
import glob
import os
from tqdm import tqdm



def pdb2png(
    pdb_fname: str, png_fname: str, add_ss: bool = False, pse: bool = False
) -> str:
    """Convert the pdb file into a png, returns output filename"""
    # https://gist.github.com/bougui505/11401240
    assert png_fname.endswith(".png")
    pymol.cmd.load(pdb_fname)
    if add_ss:
        annot_ss_psea(pdb_fname)
    pymol.cmd.show("cartoon")
    pymol.cmd.spectrum("count", palette="rainbow")
    pymol.cmd.set("ray_opaque_background", 0)
    pymol.cmd.png(png_fname, ray=1, dpi=800)
    if pse:
        pymol.cmd.save(filename=png_fname.replace(".png", ".pse"))
    pymol.cmd.delete("*")  # So we dont' draw multiple images at once
    return png_fname

if __name__ == '__main__':
    pdb_folder = sys.argv[1]
    png_folder = sys.argv[2]

    os.makedirs(png_folder,exist_ok=True)

    pdbs = glob.glob(os.path.join(pdb_folder,'*.pdb'))
    pbar = tqdm(pdbs)

    for pdb in pbar:
        pdb_fname = pdb
        base = os.path.basename(pdb).split('.')[0]
        png_fname = os.path.join(png_folder,f'{base}.png')
        pbar.set_description(f'Processing :: {base}')

        pdb2png(pdb_fname,png_fname)