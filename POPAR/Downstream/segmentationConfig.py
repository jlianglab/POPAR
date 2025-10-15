import os
import sys
class Segmentation :
    server = "sol"  # server = lab | bridges2 | agave
    debug_mode = False
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.lr = args.lr
        self.dataset = args.dataset
        self.weight = args.weight
        self.gpu = args.gpu
        self.runs = args.runs
        self.patience = args.patience
        self.img_size  =args.img_size
        self.num_classes = args.num_classes
        self.weight = args.weight
        self.anno_percent = args.anno_percent
        self.ape = args.ape

        if self.server == "lab":
            if self.dataset =="montgomery":
                self.train_image_path_file = [
                    ("/mnt/dfs/jpang12/datasets/MontgomeryCountyX-ray/MontgomerySet/", "data/montgomery/train.txt"),
                ]
                self.val_image_path_file = [
                    ("/mnt/dfs/jpang12/datasets/MontgomeryCountyX-ray/MontgomerySet/", "data/montgomery/val.txt"),
                ]
                self.test_image_path_file = [
                    ("/mnt/dfs/jpang12/datasets/MontgomeryCountyX-ray/MontgomerySet/", "data/montgomery/test.txt"),
                ]
                self.model_path = "downstream_models/{}/run_{}".format(self.dataset, self.runs)
            elif "jsrt" in self.dataset:
                self.train_image_path_file = [
                    ("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/train.txt"),
                ]
                self.val_image_path_file = [
                    ("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/val.txt"),
                ]
                self.test_image_path_file = [
                    ("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/test.txt"),
                ]
                self.model_path = "downstream_models/{}/run_{}".format(self.dataset, self.runs)
            elif "vindrribcxr" in self.dataset:
                self.train_image_path_file = [
                    ("/mnt/dfs/jpang12/datasets/VinDr-RibXCR", "data/vindrribcxr/train.json"),
                ]
                self.val_image_path_file = [
                    ("/mnt/dfs/jpang12/datasets/VinDr-RibXCR", "data/vindrribcxr/val.json"),
                ]
                self.test_image_path_file = [
                    ("/mnt/dfs/jpang12/datasets/VinDr-RibXCR", "data/vindrribcxr/test.json"),
                ]
                self.model_path = "downstream_models/{}/run_{}".format(self.dataset, self.runs)

        elif self.server == "agave" or self.server=="sol":
            if self.dataset == "montgomery":
                self.train_image_path_file = [
                    ("/data/jliang12/jpang12/dataset/MontgomerySet", "data/montgomery/train.txt"),
                ]
                self.val_image_path_file = [
                    ("/data/jliang12/jpang12/dataset/MontgomerySet", "data/montgomery/val.txt"),
                ]
                self.test_image_path_file = [
                    ("/data/jliang12/jpang12/dataset/MontgomerySet", "data/montgomery/test.txt"),
                ]
                self.model_path = "/scratch/jpang12/LADIT/downstream_models/{}/run_{}".format(self.dataset, self.runs)
            elif "jsrt" in self.dataset:
                self.train_image_path_file = [
                    ("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/train.txt"),
                ]
                self.val_image_path_file = [
                    ("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/val.txt"),
                ]
                self.test_image_path_file = [
                    ("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/test.txt"),
                ]
                self.model_path = "/scratch/jpang12/LADIT/downstream_models/{}/run_{}".format(self.dataset, self.runs)
            elif "vindrribcxr" in self.dataset:
                self.train_image_path_file = [
                    ("/data/jliang12/jpang12/dataset/VinDr-RibXCR", "data/vindrribcxr/train.json"),
                ]
                self.val_image_path_file = [
                    ("/data/jliang12/jpang12/dataset/VinDr-RibXCR", "data/vindrribcxr/val.json"),
                ]
                self.test_image_path_file = [
                    ("/data/jliang12/jpang12/dataset/VinDr-RibXCR", "data/vindrribcxr/test.json"),
                ]
                self.model_path = "/scratch/jpang12/LADIT/downstream_models/{}/run_{}".format(self.dataset, self.runs)



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


