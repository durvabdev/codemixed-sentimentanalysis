
# Setup Instructions  

Follow these steps to set up the required dependencies and download the models.  

## Install Requirements  

```sh
pip install -r requirements.txt
```

## Download fastText Models  

```sh
git clone https://github.com/facebookresearch/fastText.git
cd fastText
mkdir vectormodels
cd vectormodels

../download_model.py en
../download_model.py bn
../download_model.py hi
```

## Run the Script  

```sh
cd ~
python run.py -d fasttext
python run.py -d bpemb
```
