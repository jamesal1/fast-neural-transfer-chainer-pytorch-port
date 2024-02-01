from __future__ import print_function
import numpy as np
import argparse
from PIL import Image, ImageFilter
import time

import chainer
from chainer import cuda, Variable, serializers

import chainer_model 
import model
import numpy as np
from chainer import serializers
import time
import torch


parser = argparse.ArgumentParser(description='Real-time style transfer image generator')
parser.add_argument('--input', default="tubingen.jpg")
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', default='models/style.model', type=str)
parser.add_argument('--out', '-o', default='out_torch.jpg', type=str)
parser.add_argument('--median_filter', default=3, type=int)
parser.add_argument('--padding', default=50, type=int)
parser.add_argument('--keep_colors', action='store_true')
parser.set_defaults(keep_colors=False)
args = parser.parse_args()

def original_colors(original, stylized):
    h, s, v = original.convert('HSV').split()
    hs, ss, vs = stylized.convert('HSV').split()
    return Image.merge('HSV', (h, s, vs)).convert('RGB')


def transfer_weights(chainer_net, pytorch_net):
    def convert_array(chainer_var):
        return torch.from_numpy(chainer.as_array(chainer_var))
    def transfer_conv(chainer_conv, pytorch_conv, batch_norm):
        pytorch_conv.block[1].weight.data = convert_array(chainer_conv.W)
        pytorch_conv.block[1].bias.data = convert_array(chainer_conv.b)
        if pytorch_conv.norm is not None:
            pytorch_conv.norm.weight.data = convert_array(batch_norm.gamma)
            pytorch_conv.norm.bias.data = convert_array(batch_norm.beta)
            pytorch_conv.norm.running_var.data = convert_array(batch_norm.avg_var)
            pytorch_conv.norm.running_mean.data = convert_array(batch_norm.avg_mean)
    def transfer_res(chainer_res, pytorch_res):
        transfer_conv(chainer_res.c1, pytorch_res.block[0], chainer_res.b1)
        transfer_conv(chainer_res.c2, pytorch_res.block[1], chainer_res.b2)
    chainer_conv_blocks = []
    chainer_batch_norms = []
    chainer_res_blocks = []
    for layer in chainer_net.children():
        if layer.name[0] in "cd":
            chainer_conv_blocks += [layer]
        elif layer.name[0] == "b":
            chainer_batch_norms += [layer]
        elif layer.name[0] == "r":
            chainer_res_blocks += [layer]
    chainer_batch_norms += [None]
    pytorch_conv_blocks = []
    pytorch_res_blocks = []
    for layer in pytorch_net.model:
        if isinstance(layer, model.ConvBlock):
             pytorch_conv_blocks += [layer]
        elif isinstance(layer, model.ResidualBlock):
             pytorch_res_blocks += [layer]
    [transfer_conv(c, p, b) for c, p, b in zip(chainer_conv_blocks, pytorch_conv_blocks, chainer_batch_norms)]
    [transfer_res(c, p) for c, p in zip(chainer_res_blocks, pytorch_res_blocks)]
              
    

chainer_net = chainer_model.FastStyleNet()
serializers.load_npz("composition.model", chainer_net)
net = model.TransformerNet().eval()
weights = np.load("composition.model")
# there's some sort of bug causing the chainer model not to load the bias for the conv layers in the residual layers
# comment the line below to see this
# print("this is None: ", chainer_net.r1.c1.b)
fix_res_bias = True 
if fix_res_bias:
    for i in range(1,6):
        chainer_net[f"r{i}"].c1.b = weights[f"r{i}/c1/b"]
        chainer_net[f"r{i}"].c2.b = weights[f"r{i}/c2/b"]
transfer_weights(chainer_net, net)
model_path = "model.pt"
torch.save(net.state_dict(), model_path)
net = model.TransformerNet().eval()
net.load_state_dict(torch.load(model_path))
# if args.gpu >= 0:
#     cuda.get_device(args.gpu).use()
#     model.to_gpu()
# xp = np if args.gpu < 0 else cuda.cupy

xp = np



start = time.time()
original = Image.open("tubingen.jpg").convert('RGB')
image = np.asarray(original, dtype=np.float32).transpose(2, 0, 1)
image = image.reshape((1,) + image.shape)
if args.padding > 0:
	image = np.pad(image, [[0, 0], [0, 0], [args.padding, args.padding], [args.padding, args.padding]], 'symmetric')
     
image = xp.asarray(image)
x = Variable(image)
with chainer.using_config('train', False):
    y = chainer_net(x, True)
result = cuda.to_cpu(y.data)

with torch.no_grad():
    image = torch.from_numpy(image)
    result = net(image).numpy()

if args.padding > 0:
	result = result[:, :, args.padding:-args.padding, args.padding:-args.padding]
result = np.uint8(result[0].transpose((1, 2, 0)))
med = Image.fromarray(result)
if args.median_filter > 0:
	med = med.filter(ImageFilter.MedianFilter(args.median_filter))
if args.keep_colors:
    med = original_colors(original, med)
print(time.time() - start, 'sec')

med.save(args.out)






