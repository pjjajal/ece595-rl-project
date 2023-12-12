# ece595-rl-project


## Environment Setup
The underlying environment (TextWorld) requires either MacOS or Linux. If you attempt to install this on anything else it will not work. This application was developed on WSL, attempt this in WSL for reproducibility.
### Requirements:
- conda > 23.X

### Setup
To setup the environment run the following: 
```bash
# This creates the environment 
conda env create -f environment.yml
# Activate
conda activate rl-project
```

## Generating Games
Games are stored in the `games` directory. To generate the treasure-hunt games run `./generate_thunt.sh`

## Scripts
### Running LlamaV2 on treasure level 1 (generated with ./generate_thunt.sh)
```bash
python run.py --model llama --game games/thunt_1.z8
```

### Attempting to finetune LlamaV2 on treasure level 1 (generated with ./generate_thunt.sh)
```bash
python run.py --model llama --game games/thunt_1.z8
```