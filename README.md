# Multimodal Contrastive Learning with Graph Networks and Large Language Models for miRNA-Disease Association Prediction
![image](workflow.png)
## Software Requirements
python
### Python Dependencies
```bash
pytorch
sklearn
```
## Installation Guide
### download this repo
```bash
git clone https://github.com/ZHANG-GY/synergyMD.git
```
### install env
```bash
conda create -n synergymd python=3.9
conda activate synergymd
pip install -r requirements.txt
```
## Dataset
Known miRNA-disease associations were obtained from Human MicroRNA Disease Database (HMDD) v4.0.  Disease-gene associations were derived from DisGeNET, and miRNA-target gene associations from mirTarbase. miRNA sequence information was gained from the miRBase. After integration, our dataset comprises 21,213 experimentally confirmed associations between 1,119 miRNAs and 1,456 diseases. For auxiliary networks, there are 346,585 miRNA-gene associations (1,602 miRNAs and 15,311 genes) and 749,270 disease-gene associations (7,732 diseases and 21,038 genes). miRNA sequences are available for 1,917 miRNAs. 

The data folder includes biological entity names and multi-modal features (Sequence-based & graph-based).
```text
/data
├── miRNA graph names.pkl          # Names of 1,602 miRNAs
├── miRNA graph features.npy       # Graph features of 1602 miRNAs from GCN
├── miRNA sequence names.pkl       # Names of 1917 miRNAs
├── miRNA sequence features.npy    # Sequence features of 1917 miRNAs from RNABERT
├──disease graph names.pkl         # IDs of 7732 diseases (MeSH)
├── disease graph features.npy     # Graph features of 7732 diseases from GCN
├── disease text names.pkl         # IDs of 7053 diseases (MeSH)
├── disease text features.npy      # Text features of 7053 diseases from BioLinkBERT
└──miRNA-disease associations.npy  # 42626 miRNA-disease associations from HMDD v4.0 (including positive samples and negative samples)
```
## Model Implementations & Evaluation
To rigorously evaluate **SynergyMD**, we implement three data partitioning strategies. Each resides in its respective sub-folder under `/models`:
### 1. Random Split (`/models/random_split`)
- **Purpose**: Evaluates the model's overall predictive power on known associations.
- **Files**: `synergyMD_random.py`, `synergyMD_ST_random.py`.

### 2. Disease-based Split (`/models/disease_split`)
- **Purpose**: Simulates "Cold-start" scenarios for diseases. It ensures that diseases in the test set are completely unseen during training, testing the model's generalization to novel diseases.
- **Files**: `synergyMD_disease.py`, `synergyMD_ST_disease.py`.

### 3. miRNA-based Split (`/models/mirna_split`)
- **Purpose**: Evaluates the model's ability to predict associations for newly discovered miRNAs that have no prior association data in the training set.
- **Files**: `synergyMD_mirna.py`, `synergyMD_ST_mirna.py`.

### Usage Example
You can run different experimental scenarios directly from the project root directory.
Note: Each script is pre-configured with the best hyperparameters based on our validation tests.
For example, you can run `synergyMD_random.py` to evaluate the model's ability on randomly shuffled miRNA-disease associations for ten random seeds.
```bash
python models/random_split/synergyMD_random.py.py
```
