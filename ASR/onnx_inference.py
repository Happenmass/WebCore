from funasr_onnx import Paraformer
from pathlib import Path

model_dir = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx"
model = Paraformer(model_dir, batch_size=1, quantize=True)

wav_path = ['test.wav']

result = model(wav_path)
print(result)