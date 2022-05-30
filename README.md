# semiconAI
This project contains a metal/non-metal classifier developed using random forest algorithm. We combine this classifier with the CubicGAN model for computationally discovering new semiconductor materials. Our work is reported in detail in the following preprint.

#### [Generative Design of Stable Semiconductor Materials Using Deep Learning and DFT](https://chemrxiv.org/engage/chemrxiv/article-details/61d08f7275c57229dbff6255)

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

The data files must be in the DATA folder. In the DATA folder, we provide a file with quanternery materials' data, which was used in the paper. You can specify a crystal system in the train.py file, if you want a specific system for your project. <br />  <br />

For example, to train the model with cubic crystal systems, run the following command.  <br />
python train.py --file_name data_file.csv --sym cubic --train_size 0.1

To train the model with all the crystal systems, run the following command. <br />
python train.py --file_name data_file.csv --sym all --train_size 0.1


### Data File Structure
Please use the following format to create the training and predicting data files with .csv extension.
|Chemical Formula | Target | Crystal System|
|-----------------|--------|---------------|
| | 1: non-metal, 0: metal| |

## Predicting New Metals or Non-metals
Mention all the chemical formulas and their crystal systems in a data file with the above format. In order to keep the file strucutre, you can state 1 or 0 in the Target column. For clarity, a sample file named predict_data.csv is in DATA folder. As an example, to predict metals/non-metals, run the following command. <br />  <br />
python predict.py --file_name predict_data.csv --model model-2022_05_29_12_19_44.sav

