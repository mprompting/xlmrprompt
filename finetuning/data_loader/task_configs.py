from .NLI import XNLIDataset


task2dataset = {
    "mldoc": None,
    "marc": None,
    "argustan": None,
    "pawsx": None,
    "xnli": XNLIDataset,
}


task2labelsetsize = {"mrpc": 2, "mldoc": 4, "marc": 5, "xnli": 3}
