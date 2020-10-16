from utils import clear_folders
from mnist_trainer import MNIST


if __name__ == "__main__":
	clear_folders()
	mnist = MNIST(name="trial1")
	mnist.train()