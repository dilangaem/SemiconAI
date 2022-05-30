# semiconAI
This project contains a metal/non-metal classifier developed using random forest algorithms for computationally discovering new semiconductor materials. Our work is reported in detail in the following preprint.

#### [Generative Design of Stable Semiconductor Materials Using Deep Learning And DFT](https://chemrxiv.org/engage/chemrxiv/article-details/61d08f7275c57229dbff6255)

cite:  "Siriwardane E, Zhao Y, Perera I, Hu J. Generative Design of Stable Semiconductor Materials Using Deep Learning And DFT. ChemRxiv. Cambridge: Cambridge Open Engage; 2022; This content is a preprint and has not been peer-reviewed."

**Under construction**

## Prerequisites
- python 3.7
- pandas 1.3.0
- numpy 1.21.0
- sklearn 1.0.0
- scipy 1.5.1

It is highly recomending to build a python environment to use our code

## Training the model

The data files must be in the DATA folder. You can specify a crystal system in the train.py file, if you want a specific system for your project. To train the model, run the following command.  <br />  <br />
python train.py

### Data File Structure
Please use the following format to create the training and predicting data files with .csv extension.
|Chemical Formula | Target | Crystal System|
|-----------------|--------|---------------|
| | 1: non-metal, 0: metal| |

## Predicting New Metals or Non-metals


