import os
import torch
import sys



class ChestX_ray14:

    server = "lab"  # server = lab | psc | agave
    debug_mode = False
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.lr = args.lr
        self.task = args.task
        self.dataset = args.dataset
        self.test = args.test
        self.weight = args.weight
        self.gpu = args.gpu
        self.depth = args.depth
        self.heads = args.heads
        self.in_channel = args.in_channel

        self.method = self.task+"_depth"+str(self.depth)+ \
                      "_head" + str(self.heads)+"_" + \
                      self.dataset + "_in_channel" + str(self.in_channel)

        if self.dataset == "nih14":
            self.train_image_path_file = [
                (TRAIN_DATA_PATH, "data/xray14/official/train_official.txt"),
                (VALID_DATA_PATH, "data/xray14/official/val_official.txt"),
            ]
            self.test_image_path_file = [
                (TEST_DATA_PATH, "data/xray14/official/test_official.txt"),
            ]

        self.model_path = os.path.join("pretrained_weight", self.method )


        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        logs_path = os.path.join(self.model_path, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        if os.path.exists(os.path.join(logs_path, "log.txt")):
            self.log_writter = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            self.log_writter = open(os.path.join(logs_path, "log.txt"), 'w')

        self.graph_path = os.path.join(logs_path, "graph_path")
        if not os.path.exists(self.graph_path):
            os.makedirs(self.graph_path)

        if args.gpu is not None:
            self.device = "cuda"
        else:
            self.device = "cpu"

        if self.debug_mode:
            self.log_writter = sys.stdout

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:", file=self.log_writter)
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)), file=self.log_writter)
        print("\n", file=self.log_writter)