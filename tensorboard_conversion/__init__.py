#
# Script to extract scalars and data from tensorboard. Based on laszukdawid's
# gist: https://gist.github.com/laszukdawid/62656cf7b34cac35b325ba21d46ecfcd
#

import logging
import sys
import os
import io
from pathlib import Path
from typing import List

from PIL import Image
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator


logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) != 2:
        print("ERROR! Should be called with <experiment_dir> as first and only argument")
        exit(1)

    args = sys.argv

    logging.basicConfig(level=logging.INFO)
    dir_path = Path(args[1])

    tags_to_keep = [
        "loss/train",
        "avg_accuracy/train",
        "precision/train",
        "recall/train",
        "total_accuracy/train",
        "epoch_loss/train",
        "avg_accuracy/validation",
        "precision/validation",
        "recall/validation",
        "total_accuracy/validation",
        "epoch_loss/validation",
    ]

    for experiment_name in os.listdir(dir_path):
        logger.info(f"Handling experiment '{experiment_name}'")
        summary_dir = dir_path / experiment_name / "summaries"
        out_dir = dir_path / experiment_name / "out"

        dfs, images = convert_tb_data(str(summary_dir), tags_to_keep=tags_to_keep)

        for name, df in dfs.items():
            output_path = out_dir / f"{name.replace('/', '-')}.csv"
            logger.info(f"Exporting dataframe '{name}' to '{output_path}'")
            df.to_csv(output_path)

        for name, values in images.items():
            for value in values:
                step, image = value["step"], value["value"]
                output_path = out_dir / f"{name.replace('/', '-')}_step{step}.png"
                logger.info(f"Exporting image '{name}' to '{output_path}'")
                image.save(output_path, "PNG")


def convert_tb_data(root_dir, tags_to_keep: List[str] = []):
    def convert_tfevents(filepath):
        return filter(
            lambda x: x is not None,
            [
                parse_tfevent(e)
                for e in summary_iterator(filepath)
                if len(e.summary.value) > 0
            ],
        )

    def parse_tfevent(tfevent):
        value = tfevent.summary.value[0]

        if len(tags_to_keep) > 0 and value.tag not in tags_to_keep:
            return None

        d = dict(
            wall_time=tfevent.wall_time,
            name=value.tag,
            step=tfevent.step,
        )
        if "batch_balance" in value.tag:
            # TODO these needs to be parsed to be returned
            # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tsl/protobuf/histogram.proto
            # d["value"] = tfevent.summary.value[0].histo,
            return None
        elif (
            "confusion_matrix" in value.tag
            or "wrong_examples" in value.tag
            or "correct_examples" in value.tag
        ):
            bytes_ = io.BytesIO(value.image.encoded_image_string)
            d["value"] = Image.open(bytes_)
        else:
            d["value"] = float(value.simple_value)
        return d

    columns_order = ["wall_time", "name", "step", "value"]

    results = {}
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            parsed_events = convert_tfevents(file_full_path)
            for event in parsed_events:
                if event:
                    if event["name"] not in results:
                        results[event["name"]] = []
                    results[event["name"]].append(event)

    dfs = {}
    images = {}
    for key, values in results.items():
        if len(values) > 0 and (
            isinstance(values[0]["value"], int) or isinstance(values[0]["value"], float)
        ):
            dfs[key] = pd.DataFrame(values)[columns_order]
        elif len(values) > 0 and isinstance(values[0]["value"], Image.Image):
            images[key] = values

    return dfs, images


if __name__ == "__main__":
    main()
