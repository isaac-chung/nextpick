import torch
from torch.autograd import Variable
import torchvision.models as models

import onnx
from onnx_tf.backend import prepare

import warnings
warnings.filterwarnings("ignore")


model_file = 'transfer_model.pth'
arch='resnet18'

# load pre-trained weights
model = models.__dict__[arch](num_classes=132)
model.load_state_dict(torch.load(model_file))

dummy_input = Variable(torch.randn(1, 3, 224, 224)) # one black and white 28 x 28 picture will be the input to the model
torch.onnx.export(model, dummy_input, "transfer_learning.onnx")

# Load the ONNX file
model_onnx = onnx.load('transfer_learning.onnx')

# Import the ONNX model to Tensorflow
tf_rep = prepare(model_onnx)

# Input nodes to the model
print('inputs:', tf_rep.inputs)

# Output nodes from the model
print('outputs:', tf_rep.outputs)

# All nodes in the model
print('tensor_dict:')
print(tf_rep.tensor_dict)

# outputs the tensorflow model
tf_rep.export_graph('transfer.pb')