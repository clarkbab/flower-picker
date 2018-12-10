import os
import yaml

# Converts YAML config files into Python data objects.
class ConfigParser:
	CONFIG_DIR = 'config'

	@classmethod
	def parse(self, name):
		# Construct filepath.
		file = os.path.join(ConfigParser.CONFIG_DIR, name + '.yml')

		with open(file, 'r') as stream:
			# Allow any exceptions to be raised up to caller.
			return yaml.load(stream)
