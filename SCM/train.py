import logging
import os

from model import HyperParameters, ModelTrainer

logger = logging.getLogger("train model")
logger.setLevel(logging.INFO)
logger.propagate = False
logging.getLogger("transformers").setLevel(logging.ERROR)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

MODEL_DIR = "model"
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
fh = logging.FileHandler(os.path.join(MODEL_DIR, "train.log"), encoding="utf-8")
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

if __name__ == "__main__":
    TRAINING_DATASET = 'data/train/input_clean.txt'  # for quick dev
    # TRAINING_DATASET = "data/raw/CAIL2019-SCM-big/SCM_5k.json"

    test_input_path = "data/valid/input_clean.txt"
    test_ground_truth_path = "data/valid/ground_truth.txt"
    dict_path = "data/dict"

    config = {
        "max_length": 399,
        "epochs": 15,
        "batch_size": 16,
        "learning_rate": 0.0001,
        "max_grad_norm": 1.0,
        "warmup_steps": 0.1,
        "embed_dim":300,
        "filter_num":200,
        "filter_sizes":[2,3,4,5],
        "textcnn_dropout":0.5,
        "part_num":5,
        "cross_margin":5,
    }
    hyper_parameter = HyperParameters()
    hyper_parameter.__dict__ = config
    algorithm = "TripleMatch"
    with open(dict_path, "r", encoding='utf-8') as f:
        dic = {}
        for i, data in enumerate(f.readlines()):
            word = data.strip('\n')
            dic[word] = i
    vocab_size = len(dic)
    # test_dataloader = data_generate.data_generator(config.test_data_path, config.batch_size, dict)
    trainer = ModelTrainer(
        TRAINING_DATASET,
        hyper_parameter,
        algorithm,
        test_input_path,
        test_ground_truth_path,
        vocab_size,
        dic,
    )
    trainer.train(MODEL_DIR, 1,train=True)
