# NLP_Attack
University Project About NLP, in particular Attacks to change the behavoiur.
> DATASET
1. Amazon Food Review (Classification)      https://www.kaggle.com/snap/amazon-fine-food-reviews
2. GMB Corpus (NER)                         https://www.kaggle.com/shoumikgoswami/annotated-gmb-corpus
3. ATE_ABSITA (Classification)              http://www.di.uniba.it/~swap/ate_absita/dataset.html
4. DDI (Medical)                            https://github.com/ncbi-nlp/BLUE_Benchmark/releases/tag/0.1

# Table of Model trained and augmentations

Modello  | Modello base allenato | Sinonimi | Embedding | Studio 0 | Studio 1 | Studio 2 | Studio 3 |
:------: | :-------------------: | :------: | :-------: | :------: | :------: | :------: | :------: |
DDI      |  SÌ (optuna)          | SÌ       |  SÌ       | SÌ       | SÌ       | SÌ       | SÌ       |
ATE      |  SI (optuna)          | SÌ       |  SÌ       | SÌ       | SÌ       | SÌ       | SÌ       |
GMB      |  SI weights in h5     | SÌ       |  SÌ       | SÌ       | SÌ       | SÌ       | SÌ       |
AMAZON   |  SI weights in h5     | SÌ       |  SÌ       | SÌ       | SÌ       | SÌ       | NO       |

### Definizione Studi 
**Studio 0**: MODELLO allenato con ORIGINALE guardo come si comporta </br>

**Studio 1**: MODELLO ri-allenato con AUGMENTATION (SINONIMI) guardo come si comporta </br>

**Studio 2**: MODELLO ri-allenato con AUGMENTATION (EMBEDDING) guardo come si comporta </br>

**Studio 3**: MODELLO ri-allenato con AUGMENTATION (SINONIMI,EMBEDDING) guardo come si comporta </br>

# Tabelle riguardanti i risultati
Per delle tabelle comparative sui risultati dei vari studi cercare [qui](models/).
