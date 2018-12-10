import torch
import os
from torch import nn, optim
from lib.checkpoint import Checkpoint

class ModelTrainer:
	def __init__(self, model, train_data, valid_data, conf, validate_every, save_dir):
		self.device = conf['device']
		self.model = model
		self.train_data = train_data
		self.valid_data = valid_data
		self.epochs = conf['epochs']
		self.criterion = getattr(nn, conf['criterion'])()
		self.optimizer = getattr(optim, conf['optimizer']['name'])(model.network.classifier.parameters(), lr=conf['optimizer']['lr'])
		self.validate_every = validate_every
		self.save_dir = save_dir

	def train(self):
		# Move the network to the device.
		self.model.network.to(self.device)

		# Train an epoch.
		for epoch in range(1, self.epochs + 1):
			# Keep track of the batch number.
			batch = 1

			# Training loss is the cumulative loss across an epoch.
			training_loss = 0

			# Process all training data.
			for images, labels in self.train_data:
				if batch % self.validate_every == 1:
					print(f"Epoch: {epoch}. Batch: {batch}. Training...")

				# Train on the batch.
				loss = self.__train_batch(images, labels)
				training_loss += loss

				if batch % self.validate_every == 1:
					self.__save_checkpoint(epoch)
					# Average the training loss across the number of batches of training we've done.
					loss = training_loss / batch
					print(f"Training loss: {loss:.3f}. Validating...")

					self.__validate()

				# Increment the batch number.
				batch += 1

			# Save a checkpoint.
			self.__save_checkpoint(epoch)

	def __train_batch(self, images, labels):
		# Move images and labels to the device.
		images, labels = images.to(self.device), labels.to(self.device)

		# Zero the optimizer gradient.
		self.optimizer.zero_grad()

		# Perform forward pass.
		outputs = self.model.network.forward(images)

		# Calculate the loss.
		loss = self.criterion(outputs, labels)

		# Perform the backward pass.
		loss.backward()
		self.optimizer.step()

		return loss.item()

	def __validate(self):
		# Put model into evaluation mode, this stops dropout.
		self.model.network.eval()

		valid_batch = 1

		# Keep track of the validation loss so we can average over the number of batches and compare
		# to our training loss.
		validation_loss = 0

		# Track how many we got right.
		num_correct = 0

		for images, labels in self.valid_data:
			# Validate against the batch.
			loss, correct = self.__validate_batch(images, labels)
			validation_loss += loss
			num_correct += correct

			# Increment the batch number.
			valid_batch += 1

		# Calculate the validation loss.
		loss = validation_loss / valid_batch
		print(f"Validation loss: {loss:.3f}.")

		# Calculate the validation accuracy.
		accuracy = 100 * num_correct / self.valid_data.num_samples()
		print(f"Validation accuracy: {accuracy:.3f}")

		# Put model back into training mode.
		self.model.network.train()

	def __validate_batch(self, images, labels):
		# Move images and labels to the device.
		images, labels = images.to(self.device), labels.to(self.device)

		# Perform forward pass.
		outputs = self.model.network.forward(images)

		# Calculate the loss.
		loss = self.criterion(outputs, labels)

		# Calculate the accuracy.
		ps = torch.exp(outputs)
		_, top_cs = ps.topk(1, dim=1)
		correct = top_cs == labels.reshape(*top_cs.shape)

		return loss.item(), correct.sum()

	def __save_checkpoint(self, epoch):
		filepath = os.path.join(self.save_dir, "checkpoint_{}.pth".format(epoch))

		Checkpoint.save(self.model, filepath)

