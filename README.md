# DIAYN-JAX

This repo contains a pure Jax implementation of ["Diversity is All You Need (DIAYN)" by Eysenbach et al.](https://arxiv.org/abs/1802.06070)

## Installation

A conda environment.yaml file is provided for convenience:

```bash
conda env create --file environment.yaml
```

## Usage

Execute the code via the main script:

```bash
python diayn.py 
```

Learning 50 skills over 1e6 steps in the `halfcheetah` evironment completes in ~20 minutes:

```bash
===========================================================================
  DIAYN Training on halfcheetah
  Skills: 50 | Steps: 1000000 | Envs: 32 | SAC Ratio: 32 | Disc Ratio: 1
===========================================================================
  Obs dim: 17 | Act dim: 6
  Compiling training functions...
  Running 31250 iterations (1000000 env steps)...
  Step   50016 | Ep   64 | Reward/Step: -1.851 | Disc Loss: 1.0301 | Critic Loss: 1.3916
  Step  100000 | Ep   96 | Reward/Step: +1.305 | Disc Loss: 1.2446 | Critic Loss: 1.9008
  Step  149984 | Ep  160 | Reward/Step: -1.075 | Disc Loss: 1.3884 | Critic Loss: 2.1853
  Step  199968 | Ep  192 | Reward/Step: +2.338 | Disc Loss: 1.3817 | Critic Loss: 1.9982
  Step  249952 | Ep  256 | Reward/Step: +1.627 | Disc Loss: 1.3585 | Critic Loss: 1.7274
  Step  299936 | Ep  288 | Reward/Step: +3.220 | Disc Loss: 1.3056 | Critic Loss: 1.6477
  Step  349920 | Ep  352 | Reward/Step: +2.302 | Disc Loss: 1.2751 | Critic Loss: 1.6065
  Step  399904 | Ep  384 | Reward/Step: +2.916 | Disc Loss: 1.1727 | Critic Loss: 1.3884
  Step  449888 | Ep  448 | Reward/Step: +3.041 | Disc Loss: 1.1666 | Critic Loss: 1.4669
  Step  499872 | Ep  512 | Reward/Step: +2.504 | Disc Loss: 1.0744 | Critic Loss: 1.2764
  Step  549856 | Ep  544 | Reward/Step: +3.487 | Disc Loss: 1.0259 | Critic Loss: 1.1967
  Step  599840 | Ep  608 | Reward/Step: +3.266 | Disc Loss: 0.9832 | Critic Loss: 1.1200
  Step  649824 | Ep  640 | Reward/Step: +3.572 | Disc Loss: 0.9364 | Critic Loss: 1.0683
  Step  699808 | Ep  704 | Reward/Step: +3.473 | Disc Loss: 0.9071 | Critic Loss: 1.1089
  Step  749792 | Ep  736 | Reward/Step: +3.513 | Disc Loss: 0.8714 | Critic Loss: 0.9272
  Step  799776 | Ep  800 | Reward/Step: +3.535 | Disc Loss: 0.8461 | Critic Loss: 1.0353
  Step  849760 | Ep  864 | Reward/Step: +3.349 | Disc Loss: 0.8147 | Critic Loss: 0.9275
  Step  899744 | Ep  896 | Reward/Step: +3.357 | Disc Loss: 0.7889 | Critic Loss: 0.8599
  Step  949728 | Ep  960 | Reward/Step: +3.636 | Disc Loss: 0.7773 | Critic Loss: 0.9453
  Step  999712 | Ep  992 | Reward/Step: +3.499 | Disc Loss: 0.7551 | Critic Loss: 0.8358

===========================================================================
  Training complete!
  Total steps: 1000000 | Episodes: 992
  Avg Reward/Step: +2.4532
  Final Disc Loss: 1.0424 | Critic Loss: 1.3250
===========================================================================

  Evaluating 50 learned skills...

  Running 250 evaluation episodes in parallel...
  Skill  0: Env Reward =   -449.8 | Disc Accuracy = 99.80%
  Skill  1: Env Reward =   -488.9 | Disc Accuracy = 99.54%
  Skill  2: Env Reward =  -1241.0 | Disc Accuracy = 96.90%
  Skill  3: Env Reward =    937.4 | Disc Accuracy = 34.98%
  Skill  4: Env Reward =   -887.8 | Disc Accuracy = 98.58%
  Skill  5: Env Reward =   -475.6 | Disc Accuracy = 97.82%
  Skill  6: Env Reward =  -1506.5 | Disc Accuracy = 98.24%
  Skill  7: Env Reward =   -917.5 | Disc Accuracy = 98.70%
  Skill  8: Env Reward =  -1188.9 | Disc Accuracy = 0.42%
  Skill  9: Env Reward =    236.5 | Disc Accuracy = 98.16%
  Skill 10: Env Reward =   -617.3 | Disc Accuracy = 98.74%
  Skill 11: Env Reward =   -766.7 | Disc Accuracy = 99.40%
  Skill 12: Env Reward =   -795.8 | Disc Accuracy = 99.24%
  Skill 13: Env Reward =  -1190.8 | Disc Accuracy = 27.18%
  Skill 14: Env Reward =   1248.6 | Disc Accuracy = 53.22%
  Skill 15: Env Reward =  -1169.5 | Disc Accuracy = 4.04%
  Skill 16: Env Reward =  -1251.7 | Disc Accuracy = 99.02%
  Skill 17: Env Reward =   -281.2 | Disc Accuracy = 96.26%
  Skill 18: Env Reward =  -1168.9 | Disc Accuracy = 0.90%
  Skill 19: Env Reward =     -7.2 | Disc Accuracy = 98.74%
  Skill 20: Env Reward =  -1100.2 | Disc Accuracy = 99.14%
  Skill 21: Env Reward =  -1206.9 | Disc Accuracy = 95.80%
  Skill 22: Env Reward =   1573.7 | Disc Accuracy = 16.30%
  Skill 23: Env Reward =    956.2 | Disc Accuracy = 91.16%
  Skill 24: Env Reward =   1156.5 | Disc Accuracy = 98.80%
  Skill 25: Env Reward =  -1074.0 | Disc Accuracy = 99.18%
  Skill 26: Env Reward =   1103.4 | Disc Accuracy = 98.92%
  Skill 27: Env Reward =    930.1 | Disc Accuracy = 3.72%
  Skill 28: Env Reward =    999.5 | Disc Accuracy = 96.54%
  Skill 29: Env Reward =   -468.9 | Disc Accuracy = 96.38%
  Skill 30: Env Reward =    274.2 | Disc Accuracy = 98.54%
  Skill 31: Env Reward =    702.6 | Disc Accuracy = 98.60%
  Skill 32: Env Reward =    202.3 | Disc Accuracy = 99.22%
  Skill 33: Env Reward =   -415.9 | Disc Accuracy = 99.18%
  Skill 34: Env Reward =  -1581.9 | Disc Accuracy = 96.40%
  Skill 35: Env Reward =   -245.1 | Disc Accuracy = 99.26%
  Skill 36: Env Reward =   1879.3 | Disc Accuracy = 61.68%
  Skill 37: Env Reward =  -2315.3 | Disc Accuracy = 98.50%
  Skill 38: Env Reward =   1508.8 | Disc Accuracy = 70.82%
  Skill 39: Env Reward =   -533.5 | Disc Accuracy = 98.72%
  Skill 40: Env Reward =  -4811.4 | Disc Accuracy = 97.92%
  Skill 41: Env Reward =   -639.1 | Disc Accuracy = 99.40%
  Skill 42: Env Reward =  -1212.6 | Disc Accuracy = 5.72%
  Skill 43: Env Reward =  -1206.1 | Disc Accuracy = 3.48%
  Skill 44: Env Reward =  -1205.3 | Disc Accuracy = 90.50%
  Skill 45: Env Reward =  -1295.2 | Disc Accuracy = 97.80%
  Skill 46: Env Reward =   -517.4 | Disc Accuracy = 98.64%
  Skill 47: Env Reward =  -3694.1 | Disc Accuracy = 98.76%
  Skill 48: Env Reward =    941.5 | Disc Accuracy = 2.86%
  Skill 49: Env Reward =   -455.5 | Disc Accuracy = 98.78%

  Overall Discriminator Accuracy: 78.21%
  (Random baseline: 2.00%)

  Saved trained policies to policies/halfcheetah/
```
