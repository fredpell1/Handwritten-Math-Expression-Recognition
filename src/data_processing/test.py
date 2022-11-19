import loadData
import pandas as pd
from torchvision.io import read_image

labels = 'data/iso_GT.csv'
# images = 'data/data_png/data_png_trainingSymbols/'
#
# dataset = loadData.HMEDataset(annotations_file=labels, img_dir=images)
#
# img, lbl = dataset.__getitem__(0)
# print(img)

df = pd.read_csv(labels, sep=',')

print(df.to_string())