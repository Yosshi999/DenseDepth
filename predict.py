import os
import glob
import argparse
import matplotlib
from PIL import Image
from pathlib import Path
import numpy as np
import scipy
from tqdm import tqdm
from skimage.transform import resize

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning (for KITTI data)')
parser.add_argument('--model', default='kitti.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
parser.add_argument('--out', default='results', type=str, help='output folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
fnames = glob.glob(args.input)
inputs = []
for fn in fnames:
    im = Image.open(fn).resize((1248, 384))
    x = np.clip(np.asarray(im, dtype=float) / 255, 0, 1)
    inputs.append(x)
inputs = np.stack(inputs, axis=0)
print(inputs.shape)
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
print("original...")
outputsO = predict(model, inputs, maxDepth=8000)


inputs = np.ascontiguousarray(inputs[:,:,::-1,:])
print("flip lr...")
outputsH = predict(model, inputs, maxDepth=8000)

outputs = (outputsO + outputsH[:,:,::-1,:]) / 2
for fn, output in tqdm(zip(fnames, outputs), total=len(fnames)):
    h,w,_ = output.shape
    figure = resize(output, (375, 1242)).astype(np.float32)
    figure.tofile(str(Path(args.out) / "{}.bin".format(Path(fn).stem)))

