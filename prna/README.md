# Prepare data

```
cd ~/ # home dir
wget https://physionet.org/static/published-projects/challenge-2020/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2.zip
unzip classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2.zip
# filename too long, rename to something short
mv classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2 physionet_data
mkdir all_data
# copy the files to single `all_data` dir
mv physionet_data/training/cpsc_2018/g{1..7}/* all_data/
mv physionet_data/training/cpsc_2018_extra/g{1..4}/* all_data/
mv physionet_data/training/georgia/g{1..5}/* all_data/
mv physionet_data/training/georgia/g{6..11}/* all_data/
mv physionet_data/training/st_petersburg_incart/g1/* all_data/
mv physionet_data/training/ptb/g1/* all_data/
mv physionet_data/training/ptb-xl/g{1..5}/* all_data/
mv physionet_data/training/ptb-xl/g{6..10}/* all_data/
mv physionet_data/training/ptb-xl/g{11..15}/* all_data/
mv physionet_data/training/ptb-xl/g{16..20}/* all_data/
mv physionet_data/training/ptb-xl/g{21..22}/* all_data/
```

# Install dependencies

```
conda create -n prna python==3.10
conda activate prna
pip install -r requirements.txt
```

# Train

To run training:
```
python train_model.py ~/physionet_data/all_data ~/prna_train_out
```
The authors set training global variables in `utils.py`, to eval only, set `do_train = False`.

# View logs

```
tensorboard --logdir ~/prna_train_out
```

# infer .hea files

```
python driver.py ~/prna_train_out folder_containing_hea_files output_folder
```