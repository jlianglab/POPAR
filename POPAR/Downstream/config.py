import os
import sys




class ChestX_ray14:

    server = "sol"  # server = lab | bridges2 | agave | sol | agave_scratch | sol_scratch
    debug_mode = False
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.lr = args.lr
        self.run = args.run
        self.weight = args.weight
        self.gpu = args.gpu

        self.method = args.run
        self.patience = args.patience
        self.optm = args.optm
        self.test_only = args.test_only
        self.ape = args.ape

        self.anno_percent = args.anno_percent
        self.input_size = args.img_sizecam
        if self.server == "lab":

            self.train_image_path_file = [
                ("/mnt/dfs/jpang12/datasets/nih_xray14/images/images/", "data/xray14/train_official.txt"),
            ]
            self.valid_image_path_file = [
                ("/mnt/dfs/jpang12/datasets/nih_xray14/images/images/", "data/xray14/val_official.txt"),
            ]
            self.test_image_path_file = [
                ("/mnt/dfs/jpang12/datasets/nih_xray14/images/images/", "data/xray14/test_official.txt"),
            ]
            self.model_path = os.path.join("downstream_models/ChestXray14", self.method )
        elif self.server == "agave" or self.server=="sol":
            self.data_root = ""

            self.train_image_path_file = [
                ("/data/jliang12/jpang12/dataset/nih_xray14/images/images/", "data/xray14/train_official.txt"),
            ]
            self.valid_image_path_file = [
                ("/data/jliang12/jpang12/dataset/nih_xray14/images/images/", "data/xray14/val_official.txt"),
            ]
            self.test_image_path_file = [
                ("/data/jliang12/jpang12/dataset/nih_xray14/images/images/", "data/xray14/test_official.txt"),
            ]

            self.model_path = os.path.join("/scratch/jpang12/LADIT/downstream_models/ChestXray14",self.method)


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

        if self.debug_mode or self.test_only:
            self.log_writter = sys.stdout

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:", file=self.log_writter)
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)), file=self.log_writter)
        print("\n", file=self.log_writter)


class CheXpert:
    server = "sol"  # server = lab | bridges2 | agave | sol | agave_scratch | sol_scratch
    debug_mode = False
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.lr = args.lr
        self.run = args.run
        self.weight = args.weight
        self.gpu = args.gpu

        self.method = args.run
        self.patience = args.patience
        self.optm = args.optm
        self.test_only = args.test_only
        self.ape = args.ape

        self.anno_percent = args.anno_percent
        self.input_size = args.img_size
        if self.server == "lab":
            self.train_image_path_file = [
                ("/mnt/dfs/jpang12/datasets/", "/mnt/dfs/jpang12/datasets/CheXpert-v1.0/train.csv"),
            ]
            self.valid_image_path_file = [
                ("/mnt/dfs/jpang12/datasets/", "/mnt/dfs/jpang12/datasets/CheXpert-v1.0/valid.csv"),
            ]
            self.test_image_path_file = [
                ("/mnt/dfs/jpang12/datasets/", "/mnt/dfs/jpang12/datasets/CheXpert-v1.0/valid.csv"),
            ]

            self.model_path = os.path.join("downstream_models/CheXpert", self.method )
        elif self.server == "agave" or self.server=="sol":
            self.data_root = ""

            self.train_image_path_file = [
                ("/data/jliang12/jpang12/dataset/CheXpert-v1.0", "/data/jliang12/jpang12/dataset/CheXpert-v1.0/CheXpert-v1.0/train.csv"),
            ]
            self.valid_image_path_file = [
                ("/data/jliang12/jpang12/dataset/CheXpert-v1.0", "/data/jliang12/jpang12/dataset/CheXpert-v1.0/CheXpert-v1.0/valid.csv"),
            ]
            self.test_image_path_file = [
                ("/data/jliang12/jpang12/dataset/CheXpert-v1.0", "/data/jliang12/jpang12/dataset/CheXpert-v1.0/CheXpert-v1.0/valid.csv"),
            ]
            self.model_path = os.path.join("/scratch/jpang12/LADIT/downstream_models/CheXpert",self.method)

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



class CheXpert_Official_Test:
    server = "sol"  # server = lab | bridges2 | agave | sol | agave_scratch | sol_scratch
    debug_mode = False
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.lr = args.lr
        self.run = args.run
        self.weight = args.weight
        self.gpu = args.gpu

        self.method = args.run
        self.patience = args.patience
        self.optm = args.optm
        self.test_only = args.test_only
        self.ape = args.ape

        self.anno_percent = args.anno_percent
        self.input_size = args.img_size
        if self.server == "lab":
            self.train_image_path_file = [
                ("/mnt/dfs/jpang12/datasets/", "/mnt/dfs/jpang12/datasets/CheXpert-v1.0/train.csv"),
            ]
            self.valid_image_path_file = [
                ("/mnt/dfs/jpang12/datasets/", "/mnt/dfs/jpang12/datasets/CheXpert-v1.0/valid.csv"),
            ]
            self.test_image_path_file = [
                ("/mnt/dfs/jpang12/datasets/", "/mnt/dfs/jpang12/datasets/CheXpert-v1.0/test.csv"),
            ]

            self.model_path = os.path.join("downstream_models/CheXpert_official_test", self.method )
        elif self.server == "agave" or self.server=="sol":
            self.data_root = ""

            self.train_image_path_file = [
                ("/data/jliang12/jpang12/dataset/CheXpert-v1.0", "/data/jliang12/jpang12/dataset/CheXpert-v1.0/CheXpert-v1.0/train.csv"),
            ]
            self.valid_image_path_file = [
                ("/data/jliang12/jpang12/dataset/CheXpert-v1.0", "/data/jliang12/jpang12/dataset/CheXpert-v1.0/CheXpert-v1.0/valid.csv"),
            ]
            self.test_image_path_file = [
                ("/data/jliang12/jpang12/dataset/CheXpert-v1.0", "/data/jliang12/jpang12/dataset/CheXpert-v1.0/CheXpert-v1.0/test.csv"),
            ]
            self.model_path = os.path.join("/scratch/jpang12/LADIT/downstream_models/CheXpert_official_test",self.method)


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


class RSNA_Pneumonia:
    server = "sol"  # server = lab | bridges2 | agave | sol | agave_scratch | sol_scratch
    debug_mode = False
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.lr = args.lr
        self.run = args.run
        self.weight = args.weight
        self.gpu = args.gpu

        self.method = args.run
        self.patience = args.patience
        self.optm = args.optm
        self.test_only = args.test_only
        self.ape = args.ape

        self.anno_percent = args.anno_percent
        self.input_size = args.img_size
        if self.server == "lab":
            self.train_image_path_file = [
                ("/mnt/dfs/jpang12/datasets/rsna-pneumonia-detection-challenge/stage_2_train_images_png", "data/rsna_pneumonia/train.txt"),
            ]
            self.valid_image_path_file = [
                ("/mnt/dfs/jpang12/datasets/rsna-pneumonia-detection-challenge/stage_2_train_images_png", "data/rsna_pneumonia/val.txt"),
            ]
            self.test_image_path_file = [
                ("/mnt/dfs/jpang12/datasets/rsna-pneumonia-detection-challenge/stage_2_train_images_png", "data/rsna_pneumonia/test.txt"),
            ]

            self.model_path = os.path.join("downstream_models/RSNA_Pneumonia", self.method )
        elif self.server == "agave" or self.server=="sol":
            self.data_root = ""

            self.train_image_path_file = [
                ("/data/jliang12/jpang12/dataset/rsna-pneumonia-detection-challenge/stage_2_train_images_png", "data/rsna_pneumonia/train.txt"),
            ]
            self.valid_image_path_file = [
                ("/data/jliang12/jpang12/dataset/rsna-pneumonia-detection-challenge/stage_2_train_images_png", "data/rsna_pneumonia/val.txt"),
            ]
            self.test_image_path_file = [
                ("/data/jliang12/jpang12/dataset/rsna-pneumonia-detection-challenge/stage_2_train_images_png", "data/rsna_pneumonia/test.txt"),
            ]
            self.model_path = os.path.join("/scratch/jpang12/LADIT/downstream_models/RSNA_Pneumonia",self.method)


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


class ShenzhenRCXR:
    server = "sol"  # server = lab | bridges2 | agave | sol | agave_scratch | sol_scratch
    debug_mode = False
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.lr = args.lr
        self.run = args.run
        self.weight = args.weight
        self.gpu = args.gpu

        self.method = args.run
        self.patience = args.patience
        self.optm = args.optm
        self.test_only = args.test_only
        self.ape = args.ape

        self.anno_percent = args.anno_percent
        self.input_size = args.img_size

        if self.server == "lab":
            self.data_root = "/mnt/dfs/jpang12/datasets/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png"
            self.model_path = os.path.join("saved_models", self.method)
        elif self.server == "agave" or self.server == "sol":

            self.train_image_path_file = [
                ("/data/jliang12/jpang12/dataset/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png", "data/shenzhen/ShenzenCXR_train_data.txt"),
            ]
            self.valid_image_path_file = [
                ("/data/jliang12/jpang12/dataset/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png", "data/shenzhen/ShenzenCXR_valid_data.txt"),
            ]
            self.test_image_path_file = [
                ("/data/jliang12/jpang12/dataset/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png", "data/shenzhen/ShenzenCXR_test_data.txt"),
            ]
            self.model_path = os.path.join("/scratch/jpang12/LADIT/downstream_models/shenzhenCXR",self.method)

        self.num_classes = 1


        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        logs_path = os.path.join(self.model_path, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        if os.path.exists(os.path.join(logs_path, "log.txt")):
            self.log_writter = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            self.log_writter = open(os.path.join(logs_path, "log.txt"), 'w')

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
