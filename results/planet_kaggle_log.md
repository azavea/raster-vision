# Planet Kaggle Experiment Log

## Increased F2 👍

####  6_8_17/longer
Trained for approx twice as long. Switched to using longer epochs that approx run through entire dataset.

test: 0.92325

#### 6_7_17/jpg
Switched to using JPG files

test: 0.91549

#### 5_17_17/baseline
Baseline resnet model with decision threshold at 0.2 using TIFF files

test: 0.87193

## Decreased F2 👎

#### 6_8_17/longer_pretraining
Used a pretrained model and reduced learning rate which curiously allowed for massive overfitting. Within a few epochs, the validation loss reaches a minimum, while the training loss keeps going down the whole time, reaching a much lower value than without pretraining.

validation: 0.9108 vs 0.9213 (without pretraining)
