# AlertSounds

## Motivation

Environmental sound classification can potentially __help defense infrastructure detect dangerous flying targets in addition to conventional detection methods__. This project aims to build a proof of concept sound classifier that can distinguish typical urban environmental sounds from similarly sounding air targets.

## Dependencies

The Librosa sound processing library must be installed:

pip install librosa

## Trained NN

The custom attention CNN can be downloaded from the following link:

https://drive.google.com/file/d/1gjdaXr-stnMYKLnFm-CL6kP44Yutz22-/view?usp=sharing

## SoundPredictor

The SoundPredictor takes two string or Path arguments as paths to the sound file/directory and to the NN .zip model file, respectively:

Example (for tfcnn):
```
from ESCPredictor import SoundPredictor, tfcnn

predictor = SoundPredictor(path_to_model = path_to_model, mode = 'multiclass')
predictor.load_sounds_segments(path = path_to_sounds)
predictor.predict_segments(3)
r = predictor.get_result(print_result=True)

```

# Predict and show results

The predict() function evaluates the top-n (default n = 3) class probabilities and top-n class values for each loaded sound file:

```
predictor.predict()

```
predictor.get_result(print_result=True)

```
Out:
```
------------------------------
File mis2023_s4 is missle
------------------------------
File mis2023_s1 is missle
------------------------------
File mis2023_s8 is missle
------------------------------
File hel2023_s1 is children_playing
------------------------------
File mis2023_s7 is missle
------------------------------
File mis2023_s2 is helicopter
------------------------------
File mis2023_s6 is missle
------------------------------
File mis2023_s5 is missle

```

Additionally, you can use .result field of SoundPredictor to get access to predicted segment values.  

# Important notes

Please use short (< 5 s) .wav file as inputs. 5-fold mean accuracy and f1 score were 0.92 and 0.91 respectively.