import os
import torch

from lib.model import Model

class Checkpoint:
	@classmethod
	def save(self, model, path):
		# Create checkpoint data.
		data = {
			'conf': model.conf,
			'idx_to_class': model.idx_to_class,
			'state': model.network.classifier.state_dict()
		}

		# Ensure that directory exists before saving.
		dir = os.path.dirname(path)
		if not os.path.exists(dir):
			os.makedirs(dir)

    # Save to disk.
		torch.save(data, path)

	@classmethod
	def load(self, path):
		data = torch.load(path)

		return Model(conf=data['conf'], idx_to_class=data['idx_to_class'], state=data['state'])
