from utils import clear_folders
from mnist_trainer import MNIST
from visualizations import Visualizer, ModelComparator
import argparse


if __name__ == "__main__":

	# read the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--clear', default=False, action='store_true')
	parser.add_argument('--train', default=False, action='store_true')
	parser.add_argument('--visualize', default=False, action='store_true')
	parser.add_argument('--filename', default='trial', type=str)
	args = parser.parse_args()

	if args.clear:
		clear_folders()
	if args.train:
		mnist = MNIST(name=args.filename)
		mnist.train()
	if args.visualize:
		vis = ModelComparator(names=["trial", "trial1", "trial2", "trial3", "trial4"])
		#vis = Visualizer(name=args.filename, visualize=True)
