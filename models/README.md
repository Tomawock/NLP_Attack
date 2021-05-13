# Riassunto dei risultati

Si riporta in forma tabulare, dataset per dataset, i risultati sui vari dataset per tutti gli studi.

Con **Testset aug** s'intende il testset relativo all'augmentation usata nello studio. <br/>
Con **Testset org** s'intende il testset relativo al dataset senza alcuna augmentation, in tal senso originario. <br/>
Notare come nello Studio 0 i due coincidano.

## DDI

Testset    | Studio 0 | Studio 1 | Studio 2 | Studio 3 |
:---------:|:--------:|:--------:|:--------:|:--------:|
Testset aug|  0.8410  |  0.8338  |  0.8290  |  0.8207  |
Testset org|  0.8410  |  0.8349  |  0.8315  |  0.8164  |
Sinonimi   |  0.8332  |  0.8327  |  0.8301  |  0.8200  |
Embedding  |  0.8368  |  0.8294  |  0.8266  |  0.7971  |

## GMB

Testset    | Studio 0 | Studio 1 | Studio 2 | Studio 3 |
:---------:|:--------:|:--------:|:--------:|:--------:|
Testset aug|  0.9399  |  0.9350  |  0.9371  |  0.9019  |
Testset org|  0.9399  |  0.9408  |  0.9409  |  0.9407  |
Sinonimi   |  0.9272  |  0.9292  |  0.9290  |  0.9292  |
Embedding  |  0.9312  |  0.9327  |  0.9333  |  0.9331  |


## AMAZON

Studio 1_005 ha un dataset con probabilità di swap 0.05, studio 1 dello 0.5. Stesso per i sinonimi.

Testset    | Studio 0 | Studio 1 | Studio 1_005 | Studio 2 |
:---------:|:--------:|:--------:|:------------:|:--------:|
Testset aug|  0.9212  |  0.8639  |    0.9116    |  0.9466  |
Testset org|  0.9212  |  0.9227  |    0.9451    |  0.9451  |
Sinonimi   |  0.8062  |  0.9007  |    0.8238    |  0.8139  |
Sinonimi005|  0.9038  |  0.9123  |    0.9435    |  0.9312  |
Embedding  |  0.9071  |  0.9147  |    0.9362    |  0.9476  |


## ATE_ABSITA

### Split 50%, senza validation

Testset    | Studio 0 | Studio 1 | Studio 2 | Studio 3 |
:---------:|:--------:|:--------:|:--------:|:--------:|
Testset aug|  0.7142  |  0.6675  |  0.6908  |  0.7117  |
Testset org|  0.7142  |  0.6675  |  0.6925  |  0.7308  |
Sinonimi   |  0.6433  |  0.6675  |  0.6517  |  0.6925  |
Embedding  |  0.6650  |  0.6458  |  0.6892  |  0.7117  |

### Split 5%, con validation

Testset    | Studio 0 | Studio 1 | Studio 2 | Studio 3 |
:---------:|:--------:|:--------:|:--------:|:--------:|
Testset aug|  0.7442  |  0.7696  |  0.7633  |  0.7697  |
Testset org|  0.7442  |  0.7700  |  0.7592  |  0.7725  |
Sinonimi   |  0.7400  |  0.7692  |  0.7542  |  0.7667  |
Embedding  |  0.7442  |  0.7683  |  0.7675  |  0.7700  |
