import torch
import torchvision
from torchvision import transforms

# This dataloader provides an iterable class that releases image and labels in batches.
# This is really just a wrapper for the PyTorch dataloader.
class DataLoader:
	def __init__(self, data_folder, trans_conf, batch_size=16):
		self.data_folder = data_folder

		# Create the transforms.
		self.transforms = self.__build_transforms(trans_conf)

		# Create PyTorch dataset and dataloader.
		self.dataset = torchvision.datasets.ImageFolder(self.data_folder, transform=self.transforms)
		self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

		# Create mapping from index to class.
		self.index_to_class = {v: k for k, v in self.dataset.class_to_idx.items()}

	def num_samples(self):
		return len(self.dataset)
		
	def __iter__(self):
		return iter(self.dataloader)

	def __next__(self):
		# Iterate over the dataloader.
		return next(self.dataloader)

	def __build_transforms(self, transforms_conf):
		ts = []

		for transform_conf in transforms_conf:
			ts.append(self.__build_transform(transform_conf))

		return transforms.Compose(ts)

	def __build_transform(self, transform_conf):
		name = transform_conf['name']

		if 'args' in transform_conf:
			args = transform_conf['args']
			return getattr(transforms, name)(*args)
		else:
			return getattr(transforms, name)()
