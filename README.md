# NLP_Attack
University Project About NLP, in particular Attacks to change his behavoiur.
> DATASET
1. Amazon Food Review (Classification)      https://www.kaggle.com/snap/amazon-fine-food-reviews
2. GMB Corpus (NER)                         https://www.kaggle.com/shoumikgoswami/annotated-gmb-corpus
3. ATE_ABSITA (Classification)              http://www.di.uniba.it/~swap/ate_absita/dataset.html
4. DDI (Medical)                            https://github.com/ncbi-nlp/BLUE_Benchmark/releases/tag/0.1

# Table of Model trained and augmentations

Modello  | Modello base allenato | Sinonimi | Embedding | Studio 0 | Studio 1.1 | Studio 1.2 |  Studio 1.3 |  Studio 1.4 | Studio 2.1 | Studio 2.2 |  Studio 2.3 |  Studio 2.4 | Studio 3.1 | Studio 3.2 |  Studio 3.3 |  Studio 3.4 |
:------: | :-------------------: | :------: | :-------: | :------: | :--------: | :--------: | :---------: | :---------: | :--------: | :--------: | :---------: | :--------: | :--------: | :--------: | :---------: | :---------: |
DDI      |  SI (optuna)          | SI       |  SI       | SI       | NO         | NO         | NO          | NO          | NO         | NO         | NO          | NO          | NO         | NO         | NO          | NO          |
ATE      |  SI (optuna)          | SI       |  SI       | SI       | NO         | NO         | NO          | NO          | NO         | NO         | NO          | NO          | NO         | NO         | NO          | NO          |
GMB      |  SI weights in h5     | SI       |  SI       | SI       | NO         | NO         | NO          | NO          | NO         | NO         | NO          | NO          | NO         | NO         | NO          | NO          |
AMAZON   |  SI weights in h5     | SI       |  SI       | SI       | NO         | NO         | NO          | NO          | NO         | NO         | NO          | NO          | NO         | NO         | NO          | NO          |

### Definizione Studi 
**Studio 0**: MODELLO base analizzare come si comporta dandogli il dataset SINONIMI e EMBEDDING come validation set, valutare il cambio di precisione. </br> 
</br> 
**Studio 1.1**: MODELLO ri-allenato con AUGMENTATION (SINONIMI) guardo come si comporta mettendo come validation: dataset originale </br>
**Studio 1.2**: MODELLO ri-allenato con AUGMENTATION (SINONIMI) guardo come si comporta mettendo come validation: dataset con sinonimi e basta </br>
**Studio 1.3**: MODELLO ri-allenato con AUGMENTATION (SINONIMI) guardo come si comporta mettendo come validation: dataset con emebedding e bsata </br>
**Studio 1.4**: MODELLO ri-allenato con AUGMENTATION (SINONIMI) guardo come si comporta mettendo come validation: dataset con tutti e 3 </br>
</br>
**Studio 2.1**: MODELLO ri-allenato con AUGMENTATION (EMBEDDING) guardo come si comporta mettendo come validation: dataset originale </br>
**Studio 2.2**: MODELLO ri-allenato con AUGMENTATION (EMBEDDING) guardo come si comporta mettendo come validation: dataset con sinonimi e basta </br>
**Studio 2.3**: MODELLO ri-allenato con AUGMENTATION (EMBEDDING) guardo come si comporta mettendo come validation: dataset con emebedding e bsata </br>
**Studio 2.4**: MODELLO ri-allenato con AUGMENTATION (EMBEDDING) guardo come si comporta mettendo come validation: dataset con tutti e 3 </br>
</br>
**Studio 3.1**: MODELLO ri-allenato con AUGMENTATION (SINONIMI,EMBEDDING) guardo come si comporta mettendo come validation: dataset originale </br>
**Studio 3.2**: MODELLO ri-allenato con AUGMENTATION (SINONIMI,EMBEDDING) guardo come si comporta mettendo come validation: dataset con sinonimi e basta </br>
**Studio 3.3**: MODELLO ri-allenato con AUGMENTATION (SINONIMI,EMBEDDING) guardo come si comporta mettendo come validation: dataset con emebedding e bsata </br>
**Studio 3.4**: MODELLO ri-allenato con AUGMENTATION (SINONIMI,EMBEDDING) guardo come si comporta mettendo come validation: dataset con tutti e 3 </br>
