from optparse import OptionParser

def parse_train():
	parser = OptionParser()

	parser.add_option('-a', '--arch', dest='arch', help='architecture of the pre-trained model. Options are alexnet, resnet18 and vgg16. (default=vgg16)')
	parser.add_option('-e', '--epochs', dest='epochs', type='int', help='number of epochs for training. (default=3)')
	parser.add_option('-g', '--gpu', action='store_true', dest='gpu', help='train on a GPU. (default=false)')
	parser.add_option('-u', '--hidden_units', dest='hidden_units', type='int', help='number of neurons in the hidden layer. (default=512)')
	parser.add_option('-l', '--learning_rate', dest='learning_rate', type='float', help='learning rate for training model. (default=0.01)')
	parser.add_option('-s', '--save_dir', dest='save_dir', help="location for saved checkpoints. (default=checkpoints)")

	return parser.parse_args()

def parse_predict():
	parser = OptionParser()

	parser.add_option('-g', '--gpu', action='store_true', dest='gpu', help='train on a GPU. (default=false)')
	parser.add_option('-k', '--top_k', dest='k', help='number of classes to show. (default=3)')
	parser.add_option('-n', '--category_names', dest='category_names', help='YAML mapping of categories to names. (default=config/classes)')

	return parser.parse_args()
