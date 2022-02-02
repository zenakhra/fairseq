import torch
import fairseq
from torch import nn

model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(['wav2vec_large.pt'])
model = model[0]
model.eval()

wav_input_16khz = torch.randn(1,10000)
z = model.feature_extractor(wav_input_16khz)
c = model.feature_aggregator(z)