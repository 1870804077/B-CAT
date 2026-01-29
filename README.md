# B-CAT: Breaking the Barrier Between Communication Rounds and Accuracy for Private Transformer Inference

## Installing B-CAT
The following commands run successfully on Ubuntu 22.04 with Python 3.10.12.We recommend using conda to manage your Python environment.
### 0. Set up Conda Environment (Recommended)
```bash
conda create -n b-cat python=3.10.12 -y
conda activate b-cat
```
### 1. Install Dependencies
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install wheel==0.40.0
```
### 2. Install B-CAT
```bash
git clone https://anonymous.4open.science/r/B-CAT-Anonymous/
cd B-CAT-Anonymous
pip install .
```

### 3. Install Transformers (for Hugging Face Integration)
```bash
git clone -b 'v4.45.0' --depth 1 https://github.com/huggingface/transformers
pip install ./transformers
```

## Running Experiments
We have a set of sub-directories in the `examples` directory for reproducible experimental results. Additional dependencies for the experiments are included in the `requirements.txt` file in each subdirectory under the folder. Please refer to the `README.md` file in the sub-directories for instructions on how to set up and run the experiments.

`unit-test` - Costs of private all of our protocols.

## License
B-CAT is MIT licensed, as found in the LICENSE file.
