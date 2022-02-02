import torch
import fairseq
import torchaudio
import numpy as np
from torch import nn

model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(['wav2vec_large.pt'])
model = model[0]
model.eval()
waveform, sample_rate = torchaudio.load('fairseq/cat/badri-1.wav')
wav_input_16khz = torch.randn(1,10000)
z = model.feature_extractor(wav_input_16khz)
c = model.feature_aggregator(z)
print("feature extractor : ")
print(z.cpu().detach().numpy().shape)
print(z.cpu().detach().numpy())
print("feature aggregator : ")
print(c.cpu().detach().numpy().shape)
print(c.cpu().detach().numpy())
