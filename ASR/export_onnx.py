from funasr import AutoModel

model = AutoModel(model="paraformer")

res = model.export(quantize=True)