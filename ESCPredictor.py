#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torchvision import models, transforms
from torchvision.datasets import DatasetFolder
import random
import shutil
from pathlib import Path
from torch.nn import Softmax
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as FT
import soundfile as sf
from typing import Any, Callable, Optional
from custom_filters import logspec

# set seeds
def fullseed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

IMG_EXTENSIONS = '.csv'

# load .csv features
def sound_loader(path):
    S_2d = np.loadtxt(path, delimiter=',')
    S_2d = S_2d.astype('float32')
    try:
        S_3d = S_2d.reshape((128, 64, 3))
    except:
        raise TypeError
    return S_3d

# time-grequency cnn class
class tfcnn(nn.Module):
    def __init__(self, num_classes):
        super(tfcnn, self).__init__()
        channels = 3
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=32 * channels,
                kernel_size=(5, 3),
                stride=2,
                padding=(2, 1),
                bias=False,
                groups=channels,
            ),
            nn.BatchNorm2d(32 * channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32 * channels,
                out_channels=32 * channels,
                kernel_size=(5, 3),
                stride=2,
                padding=(2, 1),
                bias=False,
            ),
            nn.BatchNorm2d(32 * channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32 * channels,
                out_channels=64 * channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(64 * channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64 * channels,
                out_channels=64 * channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(64 * channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64 * channels,
                out_channels=128 * channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(128 * channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128 * channels,
                out_channels=128 * channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(128 * channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.fc_1 = nn.Linear(1024 * channels, 256 * channels, bias=False)
        self.relu_1 = nn.ReLU(inplace=True)
        self.dropout_1 = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(256 * channels, num_classes)

    # Progresses data across layers
    def forward(self, x):
        out = self.block_1(x)
        out = self.block_2(out)
        out = self.block_3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc_1(out)
        out = self.relu_1(out)
        out = self.dropout_1(out)
        out = self.fc_2(out)
        return out

# custom dataset
class CSVDataset(DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = sound_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.sounds = self.samples

# main class
class SoundPredictor:
    def __init__(
        self,
        mode='binary',
        path_to_model=None,
        path_to_sounds=None,
        path_to_images=None,
                 ):

        self.mode = mode
        self.sounds_fnames = []
        self.sounds = []
        self.predicted_values = None
        self.predicted_probabilities = None
        self.path_to_images = path_to_images
        self._model = None
        self.class_names = None
        self.path_to_sounds = path_to_sounds
        self._dataloader = None
        self.result = None
        self.file_result = None
        self.images_fnames = None
        if path_to_model:
            self._load_model(path_to_model)
        if path_to_sounds:
            self._load_sounds(path_to_sounds)
            self._gen_attention_mels()
            self._init_dataloader()

    # loads model from path
    def _load_model(self, path):
        if self.mode == 'multiclass':
            self.class_names = [
                'air_conditioner',
                'car_horn',
                'children_playing',
                'dog_bark',
                'drilling',
                'engine_idling',
                'gun_shot',
                'helicopter',
                'jackhammer',
                'missile',
                'motorcycle',
                'shahed',
                'siren',
                'street_music',
                'vroom',
            ]
        elif self.mode == 'binary':
            self.class_names = ['danger', 'safe']
        if not isinstance(path, Path):
            try:
                path = Path(path)
            except:
                raise ValueError('Adjust path to model first')
            else:
                try:
                    print('Trying to load state_dict...')
                    model = models.resnet18()
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, len(self.class_names))
                    model.load_state_dict(torch.load(path))
                    self._model = model

                except:
                    print('Cannot use state_dict. Trying to load full model...')
                    model = torch.load(path)
                    self._model = model
                    print('Full model loaded')

    # loads single or multiple sound files from path
    def _load_sounds(self, path=None, to_append=False):
        if not path:
            raise ValueError('Nothing to load, please set path to sound files')
        if not isinstance(path, Path):
            try:
                path = Path(path)
            except:
                raise TypeError('path has to be str or Path')
        if not to_append:
            self.sounds = []
            self.sounds_fnames = []
        self.path_to_sounds = path
        fnames = [f for f in os.listdir(path)]
        for f in fnames:
            if f[-4:] == '.wav':
                if os.path.isfile(str(path) + '//' + f):
                    y, sr = self._return_sound(str(path) + '//' + f)
                    if len(y.shape) > 1:
                        y = librosa.to_mono(y)
                    if sr != 22050:
                        y = librosa.resample(y, sr, 22050)
                    self.sounds.append(y)
                    self.sounds_fnames.append(f)
# load single sound file
    def _return_sound(self, path):
        try:
            y, sr = librosa.load(path)
        except:
            y, sr = sf.read(path)
            y = y.T
        return y, sr

    def load_sounds(self, path, to_append=False):
        self._load_sounds(path, to_append=to_append)
        self._gen_melspec_images()
        self._init_dataloader()

    def load_sounds_segments(self, path, to_append=False):
        self._load_sounds(path, to_append=to_append)
        self._gen_attention_mels()
        self._init_dataloader()

    # generates melspec features and saves them to '/val/uknown'
    def _gen_melspec_images(self):
        if len(self.sounds) == 0:
            raise ValueError('No sounds were loaded')
        if not self.path_to_images:
            path_to_images = Path(str(self.path_to_sounds) + '/test')
            path_to_images_sub = Path(str(path_to_images) + '/unknown')
            if os.path.exists(path_to_images):
                shutil.rmtree(path_to_images)
            os.mkdir(path_to_images)
            os.mkdir(path_to_images_sub)
            self.path_to_images = path_to_images
        self.path_to_images = path_to_images
        for i in range(len(self.sounds)):
            plt.interactive(False)
            y = self.sounds[i]
            fname = self.sounds_fnames[i]
            if not np.isfinite(y).all():
                y = np.nan_to_num(
                    y,
                    nan=0.005 * np.abs((np.random.randn(1)[0])),
                    posinf=0.005 * np.abs((np.random.randn(1)[0])),
                    neginf=0.005 * np.abs((np.random.randn(1)[0])),
                )
            fig = plt.figure(figsize=[0.72, 0.72])
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            S = librosa.feature.melspectrogram(
                y=y, sr=22050, n_fft=2048, hop_length=512, fmax=22050 // 2, center=False
            )
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='log')
            filename = Path(str(path_to_images_sub) + '/' + fname[:-4] + '.jpg')
            plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            fig.clf()
            plt.close(fig)
            plt.close('all')
            del filename, fname, y, fig, ax, S
# the next three defs are from draft version and aren't useful
    def _gen_frequency_vector(self, S_log_harmonic, kernel):
        frequency_vector = FT.normalize(S_log_harmonic, dim=1)
        sftmx = Softmax(dim=0)
        while frequency_vector.shape[2] >= 3:
            frequency_vector = FT.normalize(
                FT.conv2d(frequency_vector, kernel, stride=1), dim=1
            )
        frequency_vector = sftmx(frequency_vector[0])
        return frequency_vector[:, -1].reshape(frequency_vector.shape[0], 1)

    def _gen_time_vector(self, S_log_percusive, kernel):
        time_vector = FT.normalize(S_log_percusive, dim=2)
        sftmx = Softmax(dim=0)
        while time_vector.shape[1] >= 5:
            time_vector = FT.normalize(FT.conv2d(time_vector, kernel, stride=1), dim=2)
        time_vector = sftmx(time_vector[0])
        return time_vector[-1].reshape(1, time_vector.shape[1])

    def gen_attention_mel(self, y_original, segment):
        n_fft, hop_length, n_mels, sr = 1024, 512, 128, 22050
        if len(y_original) > 88200:
            y_original = y_original[:88200]
        S_original = librosa.feature.melspectrogram(
            y=y_original,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=sr // 2,
        )
        if S_original.shape[1] > segment:
            inds = [i for i in range(0, S_original.shape[1], int(segment / 2))] + [
                S_original.shape[1]
            ]
            S_segments = []
            for i in range(2, len(inds)):
                S_segments.append(S_original[:, inds[i - 2] : inds[i]])
        else:
            S_segments = [S_original]
        S_av_segments = []
        gauss_h = torch.tensor([[[[1, 1, 1]]]], dtype=torch.float32)
        gauss_p = torch.tensor([[[[1], [2], [4], [2], [1]]]], dtype=torch.float32)
        for S in S_segments:
            if S.shape[1] < 5:
                continue
            S_harmonic, S_percusive = librosa.decompose.hpss(S)
            S_log_harmonic = to_tensor(S_harmonic)
            S_log_percusive = to_tensor(S_percusive)
            frequency_vector = self._gen_frequency_vector(S_log_harmonic, gauss_h)
            time_vector = self._gen_time_vector(S_log_percusive, gauss_p)
            S_time = FT.normalize(S_log_percusive, dim=2) * time_vector
            S_frequency = FT.normalize(S_log_harmonic, dim=2) * frequency_vector
            S_av = S_time + S_frequency
            S_av = S_av.detach().numpy()[0]
            if S_av.shape[1] == segment:
                S_av_segments.append(S_av)
            elif S_av.shape[1] > 55:
                S_pad = np.zeros((n_mels, segment - S_av.shape[1]), 'uint8')
                S_av = np.hstack([S_av, S_pad])
                S_av_segments.append(S_av)
            else:
                continue
        return S_av_segments

# feature extraction
    def gen_attention_mel2(self, y_original, segment):
        n_fft, hop_length, n_mels, sr = 1024, 512, 128, 22050
        S_mel = librosa.amplitude_to_db(
            librosa.feature.melspectrogram(
                y=y_original,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmax=sr // 2,
                power=1,
            )
        )
        S_time = logspec(y=y_original, sr=sr, n_fft=1024, part='real')
        S_phase = logspec(y=y_original, sr=sr, n_fft=1024, part='phase')
        if S_mel.shape[1] > segment:
            inds = [i for i in range(0, S_mel.shape[1], int(segment / 2))] + [
                S_mel.shape[1]
            ]
            S_mel_segments = []
            S_time_segments = []
            S_phase_segments = []
            for i in range(2, len(inds)):
                length = S_mel[:, inds[i - 2] : inds[i]].shape[1]
                if length == segment:
                    S_mel_segments.append(S_mel[:, inds[i - 2] : inds[i]])
                    S_time_segments.append(S_time[:, inds[i - 2] : inds[i]])
                    S_phase_segments.append(S_phase[:, inds[i - 2] : inds[i]])
                elif length > 55:
                    S_pad = np.zeros((n_mels, segment - length), 'uint8')
                    S_mel_segments.append(
                        np.hstack([S_mel[:, inds[i - 2] : inds[i]], S_pad])
                    )
                    S_time_segments.append(
                        np.hstack([S_time[:, inds[i - 2] : inds[i]], S_pad])
                    )
                    S_phase_segments.append(
                        np.hstack([S_phase[:, inds[i - 2] : inds[i]], S_pad])
                    )
        elif S_mel.shape[1] == segment:
            S_mel_segments = [S_mel]
            S_time_segments = [S_time]
            S_phase_segments = [S_phase]
        else:
            length = S_mel.shape[1]
            S_pad = np.zeros((n_mels, segment - length), 'uint8')
            S_mel_segments = [np.hstack([S_mel, S_pad])]
            S_time_segments = [np.hstack([S_time, S_pad])]
            S_phase_segments = [np.hstack([S_phase, S_pad])]
        return S_mel_segments, S_time_segments, S_phase_segments

    def _gen_attention_mels(self, attention=2):
        if len(self.sounds) == 0:
            raise ValueError('No sounds were loaded')
        if not self.path_to_images:
            path_to_images = Path(str(self.path_to_sounds) + '/test')
            path_to_images_sub = Path(str(path_to_images) + '/unknown')
            if os.path.exists(path_to_images):
                shutil.rmtree(path_to_images)
            os.mkdir(path_to_images)
            os.mkdir(path_to_images_sub)
            self.path_to_images = path_to_images
        segment = 64
        self.path_to_images = path_to_images
        for i in range(len(self.sounds)):
            y = self.sounds[i]
            fname = self.sounds_fnames[i]
            if not np.isfinite(y).all():
                y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
            S_av_segments = self.gen_attention_mel(y, segment)
            j = 0
            if attention == 1:
                S_av_segments = self.gen_attention_mel(y, segment)
                for S in S_av_segments:
                    S = S.astype('float16')
                    np.savetxt(
                        path_to_images_sub.joinpath(
                            'segment_' + str(j) + '_' + fname[:-4] + '.csv'
                        ),
                        S,
                        delimiter=",",
                        fmt='%.1e',
                    )
                    j += 1
            if attention == 2:
                S_mel, S_time, S_phase = self.gen_attention_mel2(y, segment)
                for i in range(len(S_mel)):
                    S = np.hstack([S_mel[i], S_time[i], S_phase[i]]).astype('float16')
                    np.savetxt(
                        path_to_images_sub.joinpath(
                            'segment_' + str(i) + '_' + fname[:-4] + '.csv'
                        ),
                        S,
                        delimiter=",",
                        fmt='%.1e',
                    )
                    j += 1

    def _init_dataloader(self):
        if len(self.sounds) == 0:
            raise ValueError('Load sounds first')
        fnames = [f for f in os.listdir(self.path_to_images.joinpath('unknown'))]
        data_transforms = transforms.Compose([transforms.ToTensor()])
        test = CSVDataset(os.path.join(self.path_to_images), data_transforms)
        test_loader = torch.utils.data.DataLoader(
            test,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            worker_init_fn=fullseed(42),
        )
        self._dataloader = test_loader
        self.images_fnames = [
            self._dataloader.dataset.sounds[i][0] for i in range(len(fnames))
        ]

    def predict(self, n=3):
        seed = 42
        fullseed(seed)
        self._model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        top_ps = np.empty((0, 3), 'float32')
        top_classes = np.empty((0, 3), 'uint8')
        for inputs, labels in self._dataloader:
            print(len(inputs))
            inputs = inputs.to(device)
            with torch.no_grad():
                out = self._model(inputs)
                prob = nnf.softmax(out)
                top_p, top_class = prob.topk(n)
                top_ps = np.vstack([top_ps, top_p.numpy()])
                top_classes = np.vstack([top_classes, top_class.numpy()])
        self.predicted_probabilities = top_ps
        self.predicted_values = top_classes
        to_show = list(zip(self.predicted_values, self.predicted_probabilities))
        to_show = dict(zip(self.images_fnames, to_show))
        self.result = to_show
# predict value of each segment
    def predict_segments(self, n):
        seed = 42
        fullseed(seed)
        self._model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.mode == 'binary':
            running_preds_i = np.empty(0)
            running_preds = np.empty(0)
            i = -1
            for inputs, labels in self._dataloader:
                inputs = inputs.to(device)
                with torch.no_grad():
                    i += 1
                    out = self._model(inputs)
                    preds = torch.sigmoid(out)
                    preds_i = torch.where(preds < 0.51, 0, 1)
                    if i > 0:
                        running_preds_i = np.vstack([running_preds_i, preds_i])
                        running_preds = np.vstack([running_preds, preds])
                    if i == 0:
                        running_preds_i = np.vstack([preds_i])
                        running_preds = np.vstack([preds])
            self.predicted_probabilities = running_preds
            self.predicted_values = running_preds_i
        elif self.mode == 'multiclass':
            top_ps = np.empty((0, n), 'float32')
            top_classes = np.empty((0, n), 'uint8')
            sftmx = nn.Softmax()
            for inputs, labels in self._dataloader:
                inputs = inputs.to(device)
                with torch.no_grad():
                    out = self._model(inputs)
                    prob = sftmx(out)
                    top_p, top_class = prob.topk(n)
                    top_ps = np.vstack([top_ps, top_p.numpy()])
                    top_classes = np.vstack([top_classes, top_class.numpy()])
            self.predicted_probabilities = top_ps
            self.predicted_values = top_classes
        to_show = list(zip(self.predicted_values, self.predicted_probabilities))
        to_show = dict(zip(self.images_fnames, to_show))
        self.result = to_show
# get classification result for file
    def get_result(self, print_result=False, index=-1):
        if self.result is None:
            raise ValueError("Run predict method first")
        if index > len(self.sounds_fnames) or index == -1:
            index = [i for i in range(len(self.images_fnames))]
        result = []
        for i, (key, value) in enumerate(self.result.items()):
            if i not in index:
                continue
            fname = key
            fresult = dict()
            for j in range(len(value[0])):
                if self.mode == 'multiclass':
                    res = {
                        self.class_names[value[0][j]]: np.around(100 * value[1][j], 1)
                    }
                elif self.mode == 'binary':
                    res = {self.class_names[value[0][j]]: np.around(value[1][j], 2)}
                fresult.update(res)
            result.append({fname: fresult})
        fnames = [f[:-4] for f in self.sounds_fnames]
        file_results = dict()
        if self.mode == 'binary':
            for fname in fnames:
                segments = 0
                dangers = 0
                for i in range(len(result)):
                    if fname in list(result[i].keys())[0]:
                        segments += 1
                        danger = list(list(result[i].values())[0].keys())[0] == 'danger'
                        dangers += int(danger)
                if segments == 0:
                    continue
                if dangers / segments >= 0.5:
                    file_result = 'danger'
                else:
                    file_result = 'safe'
                file_results.update({fname: file_result})
        if self.mode == 'multiclass':
            for fname in fnames:
                vote_list = np.zeros(len(self.class_names), dtype='uint8')
                segments = 0
                for i in range(len(result)):
                    if fname in list(result[i].keys())[0]:
                        segments += 1
                        cl = list(list(result[i].values())[0])[0]
                        class_ind = self.class_names.index(cl)
                        vote_list[class_ind] += 1
                if segments == 0:
                    continue
                file_results.update({fname: self.class_names[np.argmax(vote_list)]})
        self.file_result = file_results
        if print_result:
            for key, value in file_results.items():
                print('-' * 30)
                print('File', key, 'is', value)
        return file_results