from lib.config_parser import ConfigParser

class BaseSettings:
	@classmethod
	def get(self, base, key):
		return self.__values()[base][key]

	@classmethod
	def __values(self):
		return ConfigParser.parse('base_settings')
