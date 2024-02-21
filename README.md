# Machine-Translation

This is a Github repository to train Named Entity Recognition models. 

## Usage

Poetry is needed to run this project. 
To change the dataset and model among other things, go to python/ner/config.py 
Steps to follow
1. Clone the project
2. Go to the python folder and do poetry lock -> poetry install
3. Run cli.py train <output directory>

Alternatively you can build a docker container from the provided dockerfile and run the code on the container.

## Model

Google's t5-large model is used in this project.

## Dataset

The Machine translation model is trained on the wmt17 german to english dataset. The repository could be used with most machine translation datasets with small changes. 

