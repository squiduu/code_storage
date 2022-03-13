import json
import os
import data_mwoz21


class DatasetPreprocessor:
    def __init__(self, dataset_config):
        with open(file=dataset_config, mode="r", encoding="utf-8") as f:
            rawdata_config = json.load(f)
        self.gates = rawdata_config["slot_gates"]
        self.slot_list = rawdata_config["slots"]
        self.label_maps = rawdata_config["label_maps"]


class MultiWoz21Preprocessor(DatasetPreprocessor):
    def get_train_examples(self, data_dir: str, args):
        return data_mwoz21.create_examples(
            input_file=os.path.join(data_dir, "train_dials.json"),
            act_file=os.path.join(data_dir, "dialogue_acts.json"),
            run_type="train",
            slot_list=self.slot_list,
            label_maps=self.label_maps,
            **args
        )

    def get_valid_examples(self, data_dir: str, args):
        return data_mwoz21.create_examples(
            input_file=os.path.join(data_dir, "valid_dials.json"),
            act_file=os.path.join(data_dir, "dialogue_acts.json"),
            run_type="valid",
            slot_list=self.slot_list,
            label_maps=self.label_maps,
            **args
        )

    def get_test_examples(self, data_dir: str):
        return data_mwoz21.create_examples(
            input_file=os.path.join(data_dir, "test_dials.json"),
            act_file=os.path.join(data_dir, "dialogue_acts.json"),
            run_type="test",
            slot_list=self.slot_list,
            label_maps=self.label_maps,
        )


class SimPreprocessor(DatasetPreprocessor):
    def get_train_examples(self, data_dir: str):
        pass

    def get_valid_examples(self, data_dir: str):
        pass

    def get_test_examples(self, data_dir: str):
        pass


class AuxTaskPreprocessor(DatasetPreprocessor):
    def get_train_examples(self, data_dir: str):
        pass

    def get_valid_examples(self, data_dir: str):
        pass

    def get_test_examples(self, data_dir: str):
        pass


PREPROCESSORS = {
    "mwoz21": MultiWoz21Preprocessor,
    "sim-m": SimPreprocessor,
    "sim-r": SimPreprocessor,
    "aux_task": AuxTaskPreprocessor,
}
