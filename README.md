# codemixed-sentimentanalysis

# Setup Instructions
Follow these steps to set up the required dependencies and download the models.

# Install Requirements
pip install -r requirements.txt

# Get fastText models
git clone https://github.com/facebookresearch/fastText.git
cd fastText
./download_model.py en
./download_model.py bn
./download_mode.py hi

# switch to home directory
to run 
python run.py -d fasttext
python run.py -d bpemb
