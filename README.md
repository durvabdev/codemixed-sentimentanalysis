# Setup Instructions
Follow these steps to set up the required dependencies and download the models.

# Install Requirements
pip install -r requirements.txt

# Get fastText models
git clone https://github.com/facebookresearch/fastText.git

cd fastText

mkdir vectormodels

cd vectormodels

./fastText/download_model.py en

./fastText/download_model.py bn

./fastText/download_model.py hi

# Switch to home directory
cd ~
python run.py -d fasttext
python run.py -d bpemb
