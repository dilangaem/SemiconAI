# SemiconAI

**Under Construction** <br />


This repository contains a metal/nonmetal classifier developed using random forest algorithm. We combine this classifier with the CubicGAN model for computationally discovering new semiconductor materials. Our work is reported in detail in the following preprint.

#### [Generative Design of Stable Semiconductor Materials Using Deep Learning and DFT](https://www.nature.com/articles/s41524-022-00850-3)

How to cite:

- Siriwardane, E.M.D., Zhao, Y., Perera, I. et al. Generative design of stable semiconductor materials using deep learning and density functional theory. npj Comput Mater 8, 164 (2022). https://doi.org/10.1038/s41524-022-00850-3
- Zhao, Y., Al-Fahdi, M., Hu, M., Siriwardane, E. M. D., Song, Y., Nasiri, A., Hu, J., High-Throughput Discovery of Novel Cubic Crystal Materials Using Deep Generative Neural Networks. Adv. Sci. 2021, 8, 2100566. https://doi.org/10.1002/advs.202100566

<img src='https://github.com/dilangaem/semiconAI/blob/main/semiconAI.jpg'>

## Prerequisites
- python 3.7
- pandas 1.3.0
- numpy 1.21.0
- sklearn 1.0.0
- scipy 1.5.1

It is highly recommended to build a python environment to use our code.

## Training the Model

The data files must be in the DATA folder. In the DATA folder, we provide a file with quaternary materials' data (data_file.csv), which was used in the paper.  <br />  <br />

For example, to train the model for quaternary cubic materials, run the following command.  <br />
```bash
python train.py --file_name data_file.csv --sym cubic --test_size 0.1
```

To train the model for quaternary materials with all the crystal systems, run the following command. <br />
```bash
python train.py --file_name data_file.csv --sym all --test_size 0.1
```

### Data File Structure
Please use the following format to create the training and predicting data files with .csv extension.
|Chemical Formula | Target | Crystal System|
|-----------------|--------|---------------|
| | 1: non-metal, 0: metal| |

## Predicting New Metals or Non-metals 
Mention all the chemical formulas and their crystal systems in a .csv data file with the above format. In order to keep the file strucutre, you can state 1 or 0 in the Target column. For clarity, a sample file named predict_data.csv is in DATA folder. We also provided a trained model in the TRAINED forlder for quaternary cubic materials. <br /> <br />

As an example, to predict metals/non-metals, run the following command. <br /> 
```bash
python predict.py --file_name predict_data.csv --model_name model-2022_05_29_12_19_44.sav
```

