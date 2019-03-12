import os
import numpy as np
from collections import Counter
import itertools
from sklearn.utils import shuffle

from utils import load_vocab
import data.vlsp.constants as constants

np.random.seed(13)

vocab_words = load_vocab(constants.ALL_WORDS)
vocab_chars = load_vocab(constants.ALL_CHARS)
vocab_poses = load_vocab(constants.ALL_POSES)


def parse_data():
    labels = []
    words = []
    chars = []
    poses = []
    ls = []
    ws = []
    cs = []
    ps = []
    for file_name in os.listdir(constants.RAW_DATA):
        with open(constants.RAW_DATA + file_name) as f:
            lines = f.readlines()
            for l in lines[4:]:
                sl = l.strip().split('\t')
                if len(sl) == 0 or sl[0] in {'', '<s>'}:
                    continue

                if sl[0] != '</s>':
                    w, c = _process_word(sl[0])
                    ws.append(w)
                    cs.append(c)

                    ls.append(constants.ALL_LABELS.index(sl[3]))

                    ps.append(vocab_poses[sl[1]] if sl[1] in vocab_poses else vocab_poses['$UNK$'])
                else:
                    labels.append(ls)
                    ls = []
                    words.append(ws)
                    ws = []
                    chars.append(cs)
                    cs = []
                    poses.append(ps)
                    ps = []

    return labels, words, chars, poses


def _process_word(word):
    """

    :param str word:
    :return:
    """
    char_ids = []
    # 0. get chars of words
    for char in word:
        # ignore chars out of vocabulary
        if char in vocab_chars:
            char_ids += [vocab_chars[char]]

    # 2. get id of word
    word = word.lower()
    word_id = vocab_words[word] if word in vocab_words else vocab_words['$UNK$']

    # 3. return tuple word id, char ids
    return word_id, char_ids


class Dataset:
    def __init__(self, labels, words, chars, poses):
        self.labels = labels
        self.words = words
        self.chars = chars
        self.poses = poses

    def one_vs_nine(self):
        c = Counter(itertools.chain.from_iterable(self.labels))
        print('shape of data: {}'.format({k: c[k] for k in c}))
        num_of_example = len(self.labels)
        indicates = np.random.choice(num_of_example, num_of_example // 10, replace=False)

        one_data = Dataset(
            labels=[v for i, v in enumerate(self.labels) if i in indicates],
            words=[v for i, v in enumerate(self.words) if i in indicates],
            chars=[v for i, v in enumerate(self.chars) if i in indicates],
            poses=[v for i, v in enumerate(self.poses) if i in indicates],
        )
        c = Counter(itertools.chain.from_iterable(one_data.labels))
        print('shape of 10% data: {}'.format({k: c[k] for k in c}))

        nine_data = Dataset(
            labels=[v for i, v in enumerate(self.labels) if i not in indicates],
            words=[v for i, v in enumerate(self.words) if i not in indicates],
            chars=[v for i, v in enumerate(self.chars) if i not in indicates],
            poses=[v for i, v in enumerate(self.poses) if i not in indicates],
        )
        c = Counter(itertools.chain.from_iterable(nine_data.labels))
        print('shape of 90% data: {}'.format({k: c[k] for k in c}))

        return one_data, nine_data

    def shuffle(self):
        (
            self.labels,
            self.words,
            self.chars,
            self.poses,
        ) = shuffle(
            self.labels,
            self.words,
            self.chars,
            self.poses,
        )
