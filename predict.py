import os
from PIL import Image
from torchvision import transforms
from lib.options import parse_predict
from lib.checkpoint import Checkpoint
from lib.config_parser import ConfigParser

# Parse prediction options.
options, args = parse_predict()

# Override defaults.
k = options.k or 3
device = 'cuda' if options.gpu else 'cpu'

# Load the saved model.
path = os.path.join('checkpoints', 'checkpoint_1.pth')
model = Checkpoint.load(path)

# Load image and transform.
path = 'data/test/22/image_05360.jpg'
image = Image.open(path)

# Define pre-processing transformations.
preprocess = transforms.Compose([
	transforms.Resize(225),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(
		[0.485, 0.456, 0.406],
		[0.229, 0.224, 0.225]
	)
])

# Run transforms.
image = preprocess(image)

# Predict the flower classes.
classes, probs = model.predict(image, k=k, device=device)

# Load class-to-name mapping.
classes_to_names = ConfigParser.parse('classes')

# Convert to friendly names.
names = [classes_to_names[cls] for cls in classes]

for n, p in zip(names, probs):
	print(f"{n}\t({p})")
