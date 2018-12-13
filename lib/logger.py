from tabulate import tabulate

class Logger:
	COLOR_END_CODE = '\033[0m'

	@classmethod
	def log(self, message, color='w', indent=0):
		code = self.__color_code(color)
		indent = self.__indent(indent)
		print(code + indent + message + Logger.COLOR_END_CODE)

	@classmethod
	def error(self, message):
		self.log(message, color='r')

	@classmethod
	def table(self, data, headers, color):
		self.log(tabulate(data, headers=headers), color=color)

	@classmethod
	def __color_code(self, color):
		return {
			'bk': '\u001b[30m',
			'r': '\u001b[31m',
			'g': '\u001b[32m',
			'y': '\u001b[33m',
			'b': '\u001b[34m',
			'm': '\u001b[35m',
			'c': '\u001b[36m',
			'w': '\u001b[37m'
		}[color]

	@classmethod
	def __indent(self, num_tabs):
		indent = ''

		for i in range(0, num_tabs):
			indent += '\t'

		return indent
