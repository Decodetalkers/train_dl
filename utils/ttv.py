# This is a library to make text to vectors
# TODO: let me read the book

from numpy import dtype
from collections.abc import Iterable, Hashable
from typing import List, Dict, Tuple
import pandas as pd
from pathlib import Path
import re
import torch

_MISC_LIB = Path(__file__).parent.parent.resolve() / "misc/movie_data.csv"

VOCABULARYL: Dict[str, int] = {"<unk>": 0}

SENTIMENT_FEATURES = 2

_Data = pd.read_csv(_MISC_LIB)

_WordMatch = re.compile(r"[a-z\d]+")


_MAX_WORD_LEN = 0


def _vocabulary_init():
    index = 1
    global _MAX_WORD_LEN
    for _, row in _Data.iterrows():
        content: str = row["review"]
        len = 0
        for word in re.split(r"[;,!?\\\/<>().\s]+", content):
            if word.lower() not in VOCABULARYL and _WordMatch.fullmatch(word.lower()):
                VOCABULARYL[word.lower()] = index
                index += 1
            len += 1
        _MAX_WORD_LEN = max(_MAX_WORD_LEN, len)
    VOCABULARYL["<pad>"] = index


_vocabulary_init()

_RESERVE_PAD = VOCABULARYL["<pad>"]
VOCABULARY_LEN = len(VOCABULARYL)


class TextToVector:
    batch_size: int
    vectors: torch.Tensor
    labels: List[int] = []
    current: int
    max_len: int
    features = _MAX_WORD_LEN
    _iter = Iterable[Tuple[Hashable, pd.Series]]

    def __init__(self, dataset: pd.DataFrame, batch_size: int):
        self.current = 0
        self.batch_size = batch_size
        self.max_len = dataset.shape[0]
        self._iter = dataset.iterrows()

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.current == self.max_len:
            raise StopIteration
        end = min(self.current + self.batch_size, self.max_len)
        step = end - self.current
        start_current = 0
        next_list = torch.zeros(step, self.features, dtype=torch.int)
        next_label = torch.zeros(step, dtype=torch.int64)
        while start_current < step:
            _index, row = next(self._iter)  # ty:ignore[invalid-argument-type]
            review: str = row["review"]
            reserve_index = 0
            operator_list = next_list[start_current]
            for word in re.split(r"[;,!?\\\/<>().\s]+", review):
                if word.lower() not in VOCABULARYL or not _WordMatch.fullmatch(
                    word.lower()
                ):
                    operator_list[reserve_index] = _RESERVE_PAD
                else:
                    operator_list[reserve_index] = VOCABULARYL[word.lower()]
                reserve_index += 1

            for lindex in range(reserve_index, self.features):
                operator_list[lindex] = _RESERVE_PAD
            sentiment: int = row["sentiment"]
            next_label[start_current] = sentiment
            start_current += 1

        self.current = end
        return (next_list, next_label)
