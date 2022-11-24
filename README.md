# A Maximum Log-Likelihood Method for Imbalanced Few-Shot Learning Tasks

##  Introduction
This repository provides the algorithm demonstration for the paper A Maximum Log-Likelihood Method for Imbalanced Few-Shot Learning Tasks.

### Getting Started
Install dependencies (I prefer a conda environment)
- `conda create -n MLL_FSL python=3.8.13`
- `conda activate MLL_FSL`
- `pip install -r requirements.txt`
- `python -m ipykernel install --user --name MLL_FSL`
- `python setup.py develop` (or `python setup.py install` if you don't want to do development)

Download preprocessed features and trained models
-  This data is over 20GB and we are currently in the process of finding a public drive to host the data.

## Reproducing the results
- `cd ./scripts/`
- `bash run_all_inductive.sh` to run all inductive results
- `bash run_all_transductive.sh` to run all transductive results

# Contact
For further questions or details, reach out to Samuel Hess (shess@email.arizona.edu)

# Acknowledgements
Special thanks to the authors of many prior works that have shared their code, including:
- [SimpleShot](https://github.com/mileyan/simple_shot)

- [FEAT](https://github.com/Sha-Lab/FEAT)

- [TIM](https://github.com/mboudiaf/TIM)

- [&alpha;TIM](https://github.com/oveilleux/Realistic_Transductive_Few_Shot)
