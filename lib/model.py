import torch
import torchvision
from torch import nn
from lib.base_settings import BaseSettings
from lib.config_parser import ConfigParser

# A wrapper for the PyTorch model.
class Model:
	def __init__(self, conf, idx_to_class, state=None):
		# Save the config. This is useful for persisting the model.
		self.conf = conf

		# Save the idx_to_class. Also useful when persisting.
		self.idx_to_class = idx_to_class

		# Build the underlying model.
		self.__build(state)

	def predict(self, image, k=3, device='cpu'):
		# Put the model in evaluation model.
		self.network.eval()

		# Move network and image to GPU if required.
		image, self.network = image.to(device), self.network.to(device)

		# Run forward pass. Reshape into a batch of size 1.
		output = self.network.forward(image.reshape(1, *image.shape))

		# Put model back into training mode.
		self.network.train()

		# Get the classes with the highest probability. There's only one image in this batch so we'll 
		# have to select the 0th element.
		p = torch.exp(output)
		top_ps, top_is = p.topk(k, dim=1)
		top_cs = [self.idx_to_class[top_i.item()] for top_i in top_is[0]]
		top_ps = [top_p.item() for top_p in top_ps[0]]

		return top_cs, top_ps

	@property
	def state(self):
		return self.classifier.state_dict()

	def forward(self, input):
		return self.network.forward(input)

	def eval(self):
		return self.network.eval()

	def train(self):
		return self.network.train()

	def to(self, device):
		return self.network.to(device)

	def __build(self, state):
		self.__build_base()
		self.__build_classifier(state)

	def __build_base(self):
		# Get pre-trained model.
		self.network = getattr(torchvision.models, self.conf['base'])(pretrained=True)

		# Freeze pre-trained model params.
		for param in self.network.parameters():
			param.requires_grad = False

	def __build_classifier(self, state):
		# Get input size from pre-trained model.
		n_inputs = BaseSettings.get(self.conf['base'], 'in_features')

		layers = []

		# Add input layer.
		layers_conf = self.conf['layers']
		layers.extend([
			nn.Linear(n_inputs, layers_conf[1]['size']),
			getattr(nn, layers_conf[0]['activation'])(),
			nn.Dropout(layers_conf[0]['dropout'])
		])

		# Add hidden layers.
		zipped_layers = zip(layers_conf[1:-2], layers_conf[2:-1])
		for input_layer, output_layer in zipped_layers:
			layers.extend([
				nn.Linear(input_layer['size'], output_layer['size']),
				getattr(nn, input_layer['activation'])(),
				nn.Dropout(input_layer['dropout'])
			])

		# Add output layer.
		layers.extend([
			nn.Linear(layers_conf[-2]['size'], layers_conf[-1]['size']),
			getattr(nn, layers_conf[-1]['activation']['name'])(**layers_conf[-1]['activation']['kwargs'])
		])

		# Replace classifier.
		cls_attr = BaseSettings.get(self.conf['base'], 'classifier')
		self.classifier = nn.Sequential(*layers)
		setattr(self.network, cls_attr, self.classifier)

		# Load state if present.
		if state != None:
			self.classifier.load_state_dict(state)
