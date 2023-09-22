# Personalized Federated Learning with Domain Generalization

This is the official PyTorch implemention of our paper **Personalized Federated Learning with Domain Generalization** 

## Usage
### Setup
**pip**

See the `requirements.txt` for environment configuration. 
```bash
pip install -r requirements.txt
```
**conda**

We recommend using conda to quick setup the environment. Please use the following commands.
```bash
conda env create -f environment.yaml
conda activate pFedDR
```
### Dataset & Training Model
**Digits**

**office-caltech10**

**DomainNet**
- find three datasets [here](https://github.com/med-air/FedBN), put under `data/` directory 
   
### Train
Please using following commands to train a model with pfedDR.
- **--unseen_client** specify unseen client, option: 0-4 for digits| 0-3 for office | 0-5 for DomainNet 
```bash
cd generalated
# Digits experiment
python digits_pfedDR.py --log --unseen_client 0

# office-caltech-10 experiment
python office_pfedDR.py --log --unseen_client 0

# DomainNet experiment
python domainnet_pFedDR.py --log --unseen_client 0
```
Please using following example commands to train a model with others methods.
- **--unseen_client** specify unseen client, option: 0-4 for digits| 0-3 for office | 0-5 for DomainNet 
```bash
cd generalated
# fedavg for Digits experiment
python digits_fls.py --log --mode fedavg --unseen_client 0

# fedadg for Digits experiment
python digits_FedSR.py --log --mode fedadg --unseen_client 0

# feder for Digits experiment
python digits_fedper.py --log --unseen_client 0
```
