# ImprovNet: Generating Controllable Musical Improvisations with Iterative Corruption Refinement

🌐 [**Demo Website**](https://keshavbhandari.github.io/portfolio/improvnet.html)

📄 [**ArXiv Paper**](https://arxiv.org/abs/your-paper-id) 

🚀 [**Run in Colab**](https://colab.research.google.com/drive/1wywPizbCJOJoODzogEXY59wyBalw6Hay?usp=sharing)  

![Corruption Refinement Training](assets/trainingphase.png)
![Iterative Generation](assets/generationphase.png)
---

## Overview

ImprovNet is a transformer-based model designed to generate **expressive and controllable musical improvisations** using a self-supervised corruption-refinement strategy. It enables:

- **Cross-genre and intra-genre improvisation**  
- **Musical harmonization in different styles**  
- **Short prompt continuation and musical infilling**  
- **User control over the degree of improvisation and structure preservation**  

ImprovNet outperforms existing models like **Anticipatory Music Transformer (AMT)** in continuation and infilling tasks. It also achieves **highly recognizable jazz-style improvisations**, with 79% of listeners correctly identifying genre transformations.  

For more details, refer to the paper:  
**ImprovNet: Generating Controllable Musical Improvisations with Iterative Corruption Refinement.**  

---

## Setup Instructions

### 1. Clone the repository and install dependencies
```bash
!git clone https://github.com/keshavbhandari/improvnet.git
%cd improvnet
%pip install -r requirements.txt
```

### 2. Download data and model artifacts
```bash
import gdown

# Download dataset and model artifacts
artifacts_url = 'https://drive.google.com/uc?id=11H3y2sFUFldf6nS5pSpk8B-bIDHtFH4K'
artifacts_out = '/content/improvnet/artifacts.zip'
gdown.download(artifacts_url, artifacts_out, quiet=False)

# Unzip files
!unzip -q /content/improvnet/artifacts.zip -d /content/improvnet/
```

### 3. Generate an improvisation
sss

```bash
!python generate.py --config configs/configs_os.yaml
```

### 4. Harmonize a monophonic melody
sss

```bash
!python generate.py --config configs/configs_os.yaml
```

### 5. Generate a short prompt continuation
sss

```bash
!python generate.py --config configs/configs_os.yaml
```

### 6. Generate a short infilling
sss

```bash
!python generate.py --config configs/configs_os.yaml
```

### 7. Recreate experiments by training models from scratch
To train individual models, use the following commands:

- Train the phrase refiner model:
  ```bash
  !python phrase_refiner/train.py --config configs/configs_os.yaml
  ```
- Train the phrase generator model:
  ```bash
  !python phrase_generator/train.py --config configs/configs_os.yaml
  ```
- Train the phrase selector model:
  ```bash
  !python phrase_selector/train.py --config configs/configs_os.yaml
  ```
- Train the structure derivation model:
  ```bash
  !python structure_derivation/train.py --config configs/configs_os.yaml
  ```

## Citation

If you use this repository in your work, please cite:

```plaintext
@article{bhandari2025improvnet,
  author    = {Bhandari, Keshav and Chang, S. and Lu, T. and Enus, F. R. and Bradshaw, L. and Herremans, D. and Colton, S.},
  title     = {ImprovNet: Generating Controllable Musical Improvisations with Iterative Corruption Refinement},
  journal   = {arXiv preprint},
  year      = {2025},
  archivePrefix = {arXiv},
  eprint    = {2501.XXXXX}, % Replace with actual arXiv identifier
  primaryClass = {cs.SD} % Adjust field if necessary (e.g., cs.AI, cs.LG)
}
