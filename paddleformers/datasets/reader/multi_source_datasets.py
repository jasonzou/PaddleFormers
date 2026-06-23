# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random

from paddle.io import IterableDataset

from .file_reader import (
    FileListReader,
    FileReader,
    HuggingFaceReader,
    get_hf_dataset_config,
)


class InfiniteDataset(IterableDataset):
    """Infinite iterable dataset with shuffle support.

    This dataset supports continuous iteration and optional random shuffling.
    """

    def __init__(self, dataset, rng=None, random_shuffle=True):
        """Initialize InfiniteDataset.

        Args:
            dataset (Iterable): The original dataset to wrap.
            rng (Random, optional): Random number generator for shuffling.
            random_shuffle (bool): Whether to enable random shuffling.
        """
        self.data = list(iter(dataset))
        self.indices = list(range(len(self.data)))
        if rng is None:
            rng = random.Random()
        self.rng = rng
        self.random_shuffle = random_shuffle

    def __iter__(self):
        """Infinite iterator with optional shuffling.

        Yields:
            object: The next data sample from the dataset.
        """
        while True:
            if self.random_shuffle:
                self.rng.shuffle(self.indices)
            for i in self.indices:
                yield self.data[i]


class MultiSourceDataset(IterableDataset):
    """Dataset that combines multiple data sources with probability sampling."""

    def __init__(self, **dataset_config):
        """Initialize the multi-source dataset.

        Args:
            dataset_config (dict): dataset configurations.
        """

        # arguments process
        def _parse_multi_field(raw):
            """Parse a multi-source config field that may be a list, a
            comma-separated string, or a Python list-repr string
            (e.g. ``"['a', 'b']"`` produced by ``str(list)``).
            """
            import ast

            if isinstance(raw, (list, tuple)):
                return [str(p).strip() for p in raw if str(p).strip()]
            raw_str = str(raw).strip()
            if raw_str.startswith("["):
                try:
                    parsed = ast.literal_eval(raw_str)
                    if isinstance(parsed, (list, tuple)):
                        return [str(p).strip() for p in parsed if str(p).strip()]
                except (ValueError, SyntaxError):
                    pass
            return [p for p in raw_str.replace(" ", "").split(",") if p]

        task_dataset_path = _parse_multi_field(dataset_config["task_group"])
        task_dataset_prob = [float(p) for p in _parse_multi_field(dataset_config["task_group_prob"])]
        task_dataset_type = _parse_multi_field(dataset_config["sub_dataset_type"])

        if not (len(task_dataset_path) == len(task_dataset_prob) == len(task_dataset_type)):
            raise ValueError(
                f"The len of dataset path, prob, type are inconsistent, get task_dataset_path : {task_dataset_path}, task_dataset_prob : {task_dataset_prob}, task_dataset_type : {task_dataset_type}"
            )

        if len(task_dataset_path) == 0:
            raise ValueError("The len of dataset path is zero, please check the configuration.")

        task_dataset_samplenum = []
        for i in range(len(task_dataset_path)):
            path = task_dataset_path[i]
            if "#" in path:
                parts = path.split("#")
                if len(parts) == 2 and parts[1].isdigit():
                    task_dataset_samplenum.append(int(parts[1]))
                    task_dataset_path[i] = parts[0]
                else:
                    raise ValueError(
                        f"Invalid format for task group path: {path}. Expected '<path>#<num_samples>', got {path}"
                    )
            else:
                task_dataset_samplenum.append(None)

        tasks = []
        for i in range(len(task_dataset_path)):
            tasks.append(
                {
                    "prob": task_dataset_prob[i],
                    "filepath": task_dataset_path[i],
                    "type": task_dataset_type[i],
                    "sampling_number": task_dataset_samplenum[i],
                }
            )
        # filter zero probability task
        filtered_tasks = []
        for i, task in enumerate(tasks):
            if task["prob"] > 0:
                filtered_tasks.append(task)
        self._task_group = filtered_tasks
        supported_type = ["erniekit", "messages"]
        for idx, task in enumerate(self._task_group):
            if get_hf_dataset_config(task["filepath"]) is not None:
                task["dataset"] = HuggingFaceReader(
                    file_path=task["filepath"],
                    file_type=task["type"],
                    file_samplenum=task["sampling_number"],
                    split_multi_turn=dataset_config.get("split_multi_turn", False),
                    template_backend=dataset_config.get("template_backend", "jinja"),
                )
            elif os.path.isdir(task["filepath"]):
                task["dataset"] = FileListReader(
                    file_path=task["filepath"],
                    file_type=task["type"],
                    file_samplenum=task["sampling_number"],
                    split_multi_turn=dataset_config.get("split_multi_turn", False),
                    template_backend=dataset_config.get("template_backend", "jinja"),
                )
            elif task["type"] in supported_type:
                task["dataset"] = FileReader(
                    file_path=task["filepath"],
                    file_type=task["type"],
                    file_samplenum=task["sampling_number"],
                    split_multi_turn=dataset_config.get("split_multi_turn", False),
                    template_backend=dataset_config.get("template_backend", "jinja"),
                )
            else:
                raise NotImplementedError(f"Cannot support {task['type']} now, only support types: {supported_type}")
        sum_prob = sum([task["prob"] for task in self._task_group])
        for task in self._task_group:
            task["prob_origin"] = task["prob"]
            task["prob"] = task["prob"] / sum_prob

        self.random_seed = dataset_config["random_seed"]

    def __iter__(self):
        """Iterate through examples from multiple sources with probability sampling.

        Yields:
            dict: Processed examples from randomly selected data sources.
        """
        rng = random.Random(self.random_seed)
        probs = [task["prob"] for task in self._task_group]
        # Initialize task iterator
        for task in self._task_group:
            task["iterator"] = iter(task["dataset"])
        while True:
            task = rng.choices(self._task_group, weights=probs)[0]
            try:
                yield next(task["iterator"])
            except StopIteration:
                task["iterator"] = iter(task["dataset"])
                yield next(task["iterator"])
