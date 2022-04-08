# AutoML using Metadata Language Embeddings


| Folder | Description             | 
| ---------- | --------------------------------------------------------------- |
|data | autokaggle, kaggle metadata |
|models | metadata embedding, oboe embedding, autosklearn embedding, metric neural network |
|code | execution engine, embeddings, metric neural network, data augmentation |

## Setup

### Prerequisites
- Linux or OSX
- NVIDIA GPU or simply running on Colab

### Getting Started
- Install python 3+ and Tensorflow==2.0.0-beta1
- Clone this repo:
```bash
git clone https://github.com/idrori/automl-embedding.git
cd automl-embedding
```
- Install dependencies:
```bash
pip install -r code/requirements.txt
```

This code also requires the AlphaD3M, OBOE, TPOT and AutoSklearn.
- Locate `AlphaD3M`
```bash
cd external/alphaautoml
```

- Locate `OBOE` or clone from Oboe repo:
```bash
cd external
git clone https://github.com/udellgroup/oboe.git
cd ..
```

- Install `TPOT`:
```bash
pip install TPOT
```

- Install `AutoSklearn`:
```bash
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
pip install auto-sklearn
```

- Download the dataset using the kaggle API (dataset for each competition can be found in the corresponding row in `data/final_autokaggle.xlsx`. e.g., [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)):
    
```bash
kaggle competitions download -c titanic
```

## Dataset
All dataset can be fetched directly through the kaggle dataset. You can easily find the link to the dataset through the entry in `data/final_autokaggle.xlsx`.

Here are some examples of the dataset:

| dataset | detail |
| --- | --- |
| Titanic: Machine Learning from Disaster | 891 images from [Titanic training dataset](https://www.kaggle.com/c/titanic/data). (34KB) |
| kobe bryant shot selection | 30.7K images from the [Kobe training set](https://www.cityscapes-dataset.com/). (721KB) |
| Forest Cover Type Prediction | 15.1K images from the [Forest training set](https://www.kaggle.com/c/forest-cover-type-prediction). (13MB) |
| Indian Liver Patient Records | 583 images from [Patient dataset](https://www.kaggle.com/uciml/indian-liver-patient-records). (23KB) |
| Did it rain in Seattle? (1948-2017) | 25K images from [Seattle dataset](https://www.kaggle.com/rtatman/did-it-rain-in-seattle-19482017). (744KB) |

## Citation
If you use this code for your research, please cite our paper <a href="https://arxiv.org/pdf/">AutoML using Metadata Language Embeddings</a>:

```
@article{drori2019automlmetadata,
  title = {AutoML using Metadata Language Embeddings},
  author = {Drori, Iddo and Liu, Lu and Nian, Yi and Koorathota, Sharath and Li, Jie and Moretti, Antonio Khalil and Freire, Juliana and Udell, Madeleine},
  journal = {NeurIPS Workshop on Meta-Learning},
  year = {2019}
}
```

