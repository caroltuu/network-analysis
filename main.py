from utils import clear_folders
from mnist_trainer import MNIST
from visualizations import Visualizer
import argparse




if __name__ == "__main__":

	# read the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--clear', default=False, action='store_true')
	parser.add_argument('--train', default=False, action='store_true')
	parser.add_argument('--visualize', default=False, action='store_true')
	parser.add_argument('--name', default="trial", type=str)
	args = parser.parse_args()

	if args.clear:
		clear_folders()
	if args.train:
		mnist = MNIST(name=args.name)
		mnist.train()
	if args.visualize:
		vis = Visualizer(name=args.name)
