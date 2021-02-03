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
	parser.add_argument('-normgraphtype', default='histogram', type=str)
	parser.add_argument('-filename', default='trial', type=str)
	parser.add_argument('-n_models', default=100, type=int)
	parser.add_argument('-gpu', default=-1, type=int)
	args = parser.parse_args()
	if args.clear:
		clear_folders()
	if args.train:
		mnist = MNIST(name=args.filename, n_models=args.n_models, gpu=args.gpu)
		mnist.train()
	if args.visualize:
		vis = ModelComparator(name = args.filename, n_models=args.n_models, normgraphtype=args.normgraphtype)
