import torch
import os
from torch import nn, optim
from lib.checkpoint import Checkpoint
from lib.logger import Logger

class ModelTrainer:
	def __init__(self, model, train_data, valid_data, conf, validate_every, save_dir):
		self.device = conf['device']
		self.model = model
		self.train_data = train_data
		self.valid_data = valid_data
		self.epochs = conf['epochs']
		self.criterion = getattr(nn, conf['criterion'])()
		self.optimizer = getattr(optim, conf['optimizer']['name'])(model.classifier.parameters(), lr=conf['optimizer']['lr'])
		self.validate_every = validate_every
		self.save_dir = save_dir

	def train(self):
		# Move the model to the device.
		self.model.to(self.device)

		# Train an epoch.
		for i in range(1, self.epochs + 1):
			self.__train_epoch(i)

	def __train_epoch(self, epoch):
		# Keep track of the batch number.
		batch = 0

		# Training loss is the cumulative loss across an epoch.
		training_loss = 0

		# Process all training data.
		for images, labels in self.train_data:
			# Increment the batch number.
			batch += 1

			if batch % self.validate_every == 0:
				Logger.log(f"Epoch: {epoch}. Batch: {batch}. Training...", color='m')

			# Train on the batch.
			loss = self.__train_batch(images, labels)
			training_loss += loss

			if batch % self.validate_every == 0:
				# Average the training loss across the number of batches of training we've done.
				loss = training_loss / (batch)
				Logger.log(f"Training loss: {loss:.3f}. Validating...", indent=1)

				self.__validate()

		# Save a checkpoint.
		self.__create_checkpoint(epoch)

	def __train_batch(self, images, labels):
		# Move images and labels to the device.
		images, labels = images.to(self.device), labels.to(self.device)

		# Zero the optimizer gradient.
		self.optimizer.zero_grad()

		# Perform forward pass.
		outputs = self.model.forward(images)

		# Calculate the loss.
		loss = self.criterion(outputs, labels)

		# Perform the backward pass.
		loss.backward()
		self.optimizer.step()

		return loss.item()

	def __validate(self):
		# Put model into evaluation mode, this stops dropout.
		self.model.eval()

		# Keep track of the batch number.
		batch = 0

		# Keep track of the validation loss so we can average over the number of batches and compare
		# to our training loss.
		validation_loss = 0

		# Track how many we got right.
		num_correct = 0

		for images, labels in self.valid_data:
			# Increment the batch number.
			batch += 1

			# Validate against the batch.
			loss, correct = self.__validate_batch(images, labels)
			validation_loss += loss
			num_correct += correct

		# Calculate the validation loss.
		loss = validation_loss / batch
		Logger.log(f"Validation loss: {loss:.3f}.", indent=1)

		# Calculate the validation accuracy.
		accuracy = 100 * num_correct / self.valid_data.num_samples()
		Logger.log(f"Validation accuracy: {accuracy:.3f}%", indent=1)

		# Put model back into training mode.
		self.model.train()

	def __validate_batch(self, images, labels):
		# Move images and labels to the device.
		images, labels = images.to(self.device), labels.to(self.device)

		# Perform forward pass.
		outputs = self.model.forward(images)

		# Calculate the loss.
		loss = self.criterion(outputs, labels)

		# Calculate the accuracy.
		ps = torch.exp(outputs)
		_, top_cs = ps.topk(1, dim=1)
		correct = top_cs == labels.reshape(*top_cs.shape)

		return loss.item(), correct.sum().item()

	def __create_checkpoint(self, epoch):
		filepath = os.path.join(self.save_dir, "checkpoint_{}.pth".format(epoch))

		Checkpoint.create(self.model, filepath)

		Logger.log('Model saved.', color='r')
