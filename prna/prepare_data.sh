cd ~/ # home dir
wget https://physionet.org/static/published-projects/challenge-2020/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2.zip
unzip classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2.zip
# filename too long, rename to something short
mv classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2 physionet_data
cd physionet_data
mkdir all_data
# copy the files to single `data` dir
mv training/cpsc_2018/g{1..7}/* all_data/
mv training/georgia/g{1..5}/* all_data/
mv training/georgia/g{6..10}/* all_data/
mv training/st_petersburg_incart/g1/* all_data/
mv training/ptb/g1/* all_data/
mv training/ptb-xl/g{1..5}/* all_data/
mv training/ptb-xl/g{6..10}/* all_data/
mv training/ptb-xl/g{11..15}/* all_data/
mv training/ptb-xl/g{16..20}/* all_data/
mv training/ptb-xl/g{21..22}/* all_data/