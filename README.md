![crop1](https://github.com/user-attachments/assets/bb8144a2-550d-4a6b-9b76-5d543a4af385)


<h1 align="center"><b>FUN : FoldUNet</b></h1>
<p align="center">
  <img src="https://img.shields.io/github/stars/sudhishgupta/DeepArt?style=flat-square" alt="GitHub stars"/>
  <img src="https://img.shields.io/github/forks/sudhishgupta/DeepArt?style=flat-square" alt="GitHub forks"/>
  <img src="https://img.shields.io/github/issues/sudhishgupta/DeepArt?style=flat-square" alt="GitHub issues"/>
  <img src="https://img.shields.io/github/license/sudhishgupta/DeepArt?style=flat-square" alt="License"/>
</p>
<h3 align="center">Denoising Dihedrals</h3>


![FUN_4pdb_diffusion_movie-compress1-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/ba62ad44-b681-4682-b6dc-238637394e9c)


<h2 align='left'>📋 About</h2>
<p align='left'>FUN is a deep learning framework which employs a U-Net architecture to implement standard diffusion strategy over dihedral and internal angles of 3D protein structures.</p>

<h2 align='left'>📁 Data</h2>
<h3 align='left'>🧺 Collection</h3>
<p align='left'>Data of protein pdb structures was collected from the non-redundant CATH database</p>
<p align='left'>The PDB structures of protein were then parsed, and the required angles were calculated using an in-house script.</p>
<h3 align='left'>🏭 Preprocessing</h3>
<p align='left'>To maintain a fixed length inputs, the protein structures were chopped off into non-overlapping peptides of 64aa</p>
<p align='left'>Angles calculated for each of the peptide were in the range [-pi, pi] radians. These were brought down to [-1,1] by dividing the angles by pi</p>
<p align='left'>In total ~65k instances were generated for model training</p>


<h2 align='left'>👟 Training</h2>
<p align='left'>Edit the config.json file for editing parameters related to model, noise schedule and saving checkpoints.</p>
<p align='left'>Make sure to use the same configs for generating samples from a custom-trained model</p>

```
python train2.py [-h] [--train_data TRAIN_DATA] [--config_file CONFIG_FILE]
```
```
options:
  -h, --help            show this help message and exit
  --train_data TRAIN_DATA
                        Path to the training data (.npy arrays)
  --config_file CONFIG_FILE
                        Config file to determine all the training parameters
```

<h2 align='left'>🛎️ Sampling</h2>
<img width="1836" height="1032" alt="image" src="https://github.com/user-attachments/assets/a29bc612-afbc-4995-99c4-5d1e8a836acb" />

```
python generate_samples.py [-h] [--ckpt_path CKPT_PATH] [--noise_schedule NOISE_SCHEDULE] [--num_samples NUM_SAMPLES]
                           [--movie_folder MOVIE_FOLDER] [--out_folder OUT_FOLDER] [--n_sequences N_SEQUENCES]
                           [--temperature TEMPERATURE] [--config_file CONFIG_FILE]

```
```
options:
  -h, --help            show this help message and exit
  --ckpt_path CKPT_PATH
                        Path to model checkpoint (.pth file)
  --noise_schedule NOISE_SCHEDULE
                        choose the noise schedule based on the model in use
  --num_samples NUM_SAMPLES
                        Number of samples to generate
  --movie_folder MOVIE_FOLDER
                        Output folder for movie frames
  --out_folder OUT_FOLDER
                        Final output folder for generated structures
  --n_sequences N_SEQUENCES
                        Number of sequences to generate per sample
  --temperature TEMPERATURE
                        ProteinMPNN Sampling temperature (controls randomness)
  --config_file CONFIG_FILE
                        Config file to determine all the training parameters

```

<p align='left'>




<h2 align='left'>🏗️ Architecture</h2>
<img width="1920" height="1080" alt="Screenshot (16)" src="https://github.com/user-attachments/assets/7505a540-394f-436d-9a5d-17fe0efdc25a" />




<p align='left'></p>

![angles_t0000](https://github.com/user-attachments/assets/2e8dab12-51ea-4220-ab72-5758350dfe95)


## Denoising Diffusion Probabilistic Models (DDPM)

Denoising Diffusion Probabilistic Models (DDPMs) define a **forward diffusion process**
that gradually adds Gaussian noise to a data sample, and a **reverse denoising process**
that learns to recover the original data by iteratively removing the noise.

---

### 1. Forward Diffusion Process

Given a real data sample $\ \mathbf{x}_0 \sim q(\mathbf{x}_0) \$,  
the forward process produces a sequence of increasingly noisy samples:

$$
q(\mathbf{x}_{1:T} | \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t | \mathbf{x}_{t-1})
$$

where

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = 
\mathcal{N}\left(\mathbf{x}_t; \sqrt{1 - \beta_t} \ \mathbf{x}_{t-1} \, \beta_t \mathbf{I}\right)
$$

Here, $\( \beta_t \)$ controls the noise level at each timestep \( t \).

---

### 2. Linear Noise Scheduler

A **linear scheduler** defines $\( \beta_t )\$ that increases linearly from 
$\ \beta_{\text{start}} \ to \ \beta_{\text{end}} \$:

$$
\beta_t = \beta_{\text{start}} + 
\frac{t-1}{T-1} (\beta_{\text{end}} - \beta_{\text{start}}), \quad t = 1, \dots, T
$$

We define:

$$
\alpha_t = 1 - \beta_t, \quad 
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
$$

Then, the closed-form expression for the noisy sample at timestep $\( t \)$ is:

$$
q(\mathbf{x}_t | \mathbf{x}_0) = 
\mathcal{N}\left(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0, \, (1 - \bar{\alpha}_t)\mathbf{I}\right)
$$

---

### 3. Reverse (Denoising) Process

The goal is to learn the reverse conditional distribution:

$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = 
\mathcal{N}\left(
\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \, 
\Sigma_\theta(\mathbf{x}_t, t)
\right)
$$

where $\( \boldsymbol{\mu}_\theta \)$ is predicted by a neural network (e.g., U-Net) parameterized by $\( \theta \)$.

### 4. Training Objective

Training minimizes a simplified noise-prediction loss:

$$
L_{\text{simple}}(\theta) =
\mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}
\left[
\left\|
\boldsymbol{\epsilon} -
\boldsymbol{\epsilon}_\theta
\left(
\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}, \, t
\right)
\right\|_2^2
\right]
$$

where $\( \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}) \)$ is the noise added at each step.

---

### 5. Sampling (Reverse Process)

Starting from $\( \mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I}) \)$, we iteratively apply:

$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \, \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}
$$


where $\ \mathbf{z} \sim \mathcal{N}(0, \mathbf{I}) \ if \ t > 1 \$,
and $\ \sigma_t = \sqrt{\beta_t} \$.

---

### 6. Summary

- **Forward:** progressively corrupts data with Gaussian noise  
- **Reverse:** denoises step-by-step using a learned network  
- **Linear scheduler:** smoothly increases $\( \beta_t )\$  
- **Loss:** MSE between true and predicted noise  

DDPMs thus learn to model complex data distributions through iterative denoising.
