import torch
import fairseq
import torchaudio
import numpy as np
from torch import nn
from os import walk
from scipy import spatial

model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(['wav2vec_large.pt'])
model = model[0]
model.eval()
for (dirpath, dirnames, filenames) in walk('fairseq/cat'):
  for i in range(len(filenames)):
    for j in range(i+1,len(filenames)):
      print(filenames[i],filenames[j])
      waveform1, sample_rate1 = torchaudio.load('fairseq/cat/'+filenames[i])
      z1 = model.feature_extractor(waveform1)
      c1 = model.feature_aggregator(z1)
      feature1 = z1.cpu().detach().numpy()
      print("feature aggregator : "+filenames[i])
      print(feature1.shape)
      waveform2, sample_rate1 = torchaudio.load('fairseq/cat/'+filenames[j])
      z2 = model.feature_extractor(waveform2)
      c2 = model.feature_aggregator(z2)
      feature2 = z2.cpu().detach().numpy()
      print("feature aggregator : "+filenames[j])
      print(feature2.shape)
      result = 1 - spatial.distance.cosine(feature1.flatten(), feature2.flatten())
