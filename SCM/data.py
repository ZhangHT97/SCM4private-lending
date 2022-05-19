import json
import logging
import os

import random
from zipfile import ZipFile

from transformers import cached_path

logger = logging.getLogger(__name__)

# 比赛各阶段数据集，CAIL2019-SCM-big 是第二阶段数据集
DATASET_ARCHIVE_MAP = {
    "CAIL2019-SCM-big": "https://cail.oss-cn-qingdao.aliyuncs.com/cail2019/CAIL2019-SCM.zip"
}


def download_data(dataset_name):
    """
    下载数据集

    :param dataset_name: 数据集名称。
    :return:
    """
    url = DATASET_ARCHIVE_MAP[dataset_name]
    try:
        resolved_archive_file = cached_path(url)
    except EnvironmentError:
        logger.error("Dataset Download failed!")
        return None

    data_dir = os.path.join("data/raw", dataset_name)
    with ZipFile(resolved_archive_file, "r") as zipObj:
        data_file_name = list(filter(lambda f: f.endswith(".json"), zipObj.namelist()))
        zipObj.extract(data_file_name[0], data_dir)
        zipObj.extract(data_file_name[1], data_dir)
        zipObj.extract(data_file_name[2], data_dir)
        return os.path.join(data_dir, data_file_name[0]),os.path.join(data_dir, data_file_name[1]),\
               os.path.join(data_dir, data_file_name[2])


def generate_fix_test_data(train_file,valid_file,test_file):
    """
    生成训练数据和测试数据。

    :param raw_input_file: 原始的数据集文件
    :return:
    """
    valid_input_file = "data/valid/input.txt"
    test_input_file = "data/test/input.txt"
    train_input_file = "data/train/input.txt"
    label_output_valid = "data/valid/ground_truth.txt"
    label_output_test = "data/test/ground_truth.txt"
    valid_lines = []
    train_lines = []
    test_lines = []
    with open(train_file, encoding="utf-8") as raw_train:
        with open(valid_file,encoding="utf-8") as raw_valid:
            with open(test_file, encoding="utf-8") as raw_test:
                for line_tr in raw_train:
                    train_lines.append(line_tr.strip())
                for line_v in raw_valid:
                    valid_lines.append(line_v.strip())
                for line_t in raw_test:
                    test_lines.append(line_t.strip())

    os.makedirs("data/train", exist_ok=True)
    with open(train_input_file, mode="w", encoding="utf-8") as train_input:
        for line in train_lines:
            x = json.loads(line, encoding="utf-8")
            data = json.dumps({"A": x["A"], "B": x["B"], "C": x["C"],"label":x["label"]}, ensure_ascii=False).strip()
            train_input.write(data)
            train_input.write("\n")

    os.makedirs("data/valid", exist_ok=True)
    with open(valid_input_file, mode="w", encoding="utf-8") as valid_input, open(
        label_output_valid, encoding="utf-8", mode="w") as label_output_valid:
        for line in valid_lines:
            item = json.loads(line, encoding="utf-8")
            a = item["A"]
            b = item["B"]
            c = item["C"]
            label = item["label"]
            label_output_valid.write(label)
            label_output_valid.write("\n")
            data = json.dumps({"A": a, "B": b, "C": c}, ensure_ascii=False).strip()
            valid_input.write(data)
            valid_input.write("\n")

    os.makedirs("data/test", exist_ok=True)
    with open(test_input_file, mode="w", encoding="utf-8") as test_input, open(
            label_output_test, encoding="utf-8", mode="w") as label_output_test:
        for line in test_lines:
            item = json.loads(line, encoding="utf-8")
            a = item["A"]
            b = item["B"]
            c = item["C"]
            label = item["label"]
            label_output_test.write(label)
            label_output_test.write("\n")
            data = json.dumps({"A": a, "B": b, "C": c}, ensure_ascii=False).strip()
            test_input.write(data)
            test_input.write("\n")


if __name__ == "__main__":
    test_file,train_file,valid_file = download_data("CAIL2019-SCM-big")
    generate_fix_test_data(train_file,valid_file,test_file)
