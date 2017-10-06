import functools
import itertools
import contextlib
import collections
import copy
import hashlib
import re
import os
import os.path
import shutil
import multiprocessing
from multiprocessing.pool import Pool
from tempfile import NamedTemporaryFile
import numpy as np
import requests
from tqdm import tqdm
from fake_useragent import FakeUserAgent
import torch
from torch.utils.data import Dataset


@contextlib.contextmanager
def _progress(m1, m2=None, end1='\n', end2='\n', flush1=True, flush2=False):
    print(m1, end=end1, flush=flush1)
    yield
    if m2 is not None:
        print(m2, end=end2, flush=flush2)


class BabiQA(Dataset):
    _UNKNOWN = '<UNK>'
    _PADDING = '<PAD>'
    _DIRNAME = 'babi'
    _URL = (
        'http://www.thespermwhale.com/jaseweston/'
        'babi/tasks_1-20_v1-2.tar.gz'
    )

    def __init__(self,
                 dataset_name='en-valid-10k',
                 tasks=None,
                 sentence_size=20,
                 vocabulary=None,
                 vocabulary_size=200,
                 train=True, download=True, fresh=False, path='./datasets'):
        self._dataset_name = dataset_name
        self._tasks = tasks or [i+1 for i in range(20)]
        self._train = train
        self._path = os.path.join(path, self._DIRNAME)
        self._path_to_dataset = os.path.join(self._path, self._dataset_name)
        self._path_to_preprocessed = os.path.join(
            self._path + '-preprocessed', self._dataset_name
        )

        if download:
            self._download()

        self._sentence_size = sentence_size
        self._vocabulary = vocabulary or self._generate_vocabulary(
            self._tasks, vocabulary_size-2, train=self._train
        )
        self._word2idx = {w: i+2 for i, w in enumerate(self._vocabulary)}
        self._word2idx[self._UNKNOWN] = self.unknown_idx
        self._word2idx[self._PADDING] = self.padding_idx
        self._idx2word = {i+2: w for i, w in enumerate(self._vocabulary)}
        self._idx2word[self.unknown_idx] = self._UNKNOWN
        self._idx2word[self.padding_idx] = self._PADDING

        self._paths = self._load_to_disk(
            self._tasks,
            sentence_size=self._sentence_size,
            train=self._train, fresh=fresh,
        )

    def __getitem__(self, index):
        path = self._paths[index]
        loaded = np.load(path)
        return (
            torch.from_numpy(loaded['sentences']),
            torch.from_numpy(loaded['query']),
            torch.from_numpy(loaded['answer']),
        )

    def __len__(self):
        return len(self._paths)

    @property
    def vocabulary_hash(self):
        joined = ' '.join(self._vocabulary)
        return hashlib.sha256(joined.encode()).hexdigest()[:10]

    @property
    def vocabulary_size(self):
        return len(self._vocabulary) + 2

    @property
    def unknown_idx(self):
        return 0

    @property
    def padding_idx(self):
        return 1

    def word2idx(self, word):
        return self._word2idx.get(word, self.unknown_idx)

    def idx2word(self, idx):
        return self._idx2word[idx]

    @staticmethod
    def _remove_ending_punctuation(w):
        return w.split('?')[0].split('.')[0]

    def _cleanup_disk(self):
        for path in self._paths:
            os.unlink(path)

    def _load_to_disk(self, tasks, sentence_size,
                      train=True, fresh=False,
                      pool_size=multiprocessing.cpu_count()*3):
        with _progress('=> Preprocessing the data... '):
            paths = itertools.chain(*Pool(pool_size).map(functools.partial(
                self._load_task_data_to_disk,
                sentence_size=sentence_size,
                train=train, fresh=fresh,
            ), tasks))
            """
            paths = itertools.chain(*[self._load_task_data_to_disk(
                task, sentence_size=sentence_size, train=train, fresh=fresh
            ) for task in tasks])
            """

        return list(paths)

    def _load_task_data_to_disk(self, task, sentence_size,
                                train=True, fresh=False):
        dirpath = os.path.join(
            self._path_to_preprocessed,
            '{task}-{train}-{vocabulary_hash}'.format(
                task=task, train=('train' if train else 'test'),
                vocabulary_hash=self.vocabulary_hash
            )
        )

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        elif not fresh and os.listdir(dirpath):
            print('  * Using preprocessed data in {}'.format(dirpath))
            return [os.path.join(dirpath, n) for n in os.listdir(dirpath)]

        paths = []
        data = self._task_data(task, train=train)
        parsed = self._parse(data)
        for i, x in enumerate(parsed):
            # Load the data to the disk and retain their paths instaed of
            # loading them to the memory directly. This is to prevent OOM
            # error. The temporary files will be discarded on exit by the
            # `~_cleanup_disk()` callback.
            tmp = NamedTemporaryFile(delete=False, dir=dirpath)
            s, q, a = self._encode_x(x, sentence_size=sentence_size)
            np.savez(tmp, sentences=s, query=q, answer=a)
            tmp.close()
            paths.append(tmp.name)
        return paths

    def _encode_x(self, x, sentence_size):
        sentences, query, answer = x
        encoded_sentences = np.array([
            self._encode_words(*s.split(), sentence_size=sentence_size) for
            s in sentences
        ])
        encoded_query = self._encode_words(
            query, sentence_size=sentence_size
        )[0, None]
        encoded_answer = self._encode_words(
            answer, sentence_size=sentence_size
        )[0, None]
        return encoded_sentences, encoded_query, encoded_answer

    def _encode_words(self, *words, sentence_size):
        paddings = (self._PADDING,) * (sentence_size-len(words))
        indices = np.array([self.word2idx(w) for w in words + paddings])
        return indices.astype(np.int64)

    def _generate_vocabulary(self, tasks, vocabulary_size, train=True):
        with _progress('=> Generating a vocabulary... '):
            counter = collections.Counter()
            for task in tasks:
                words = self._task_data(task, train=train).split()
                words = [
                    self._remove_ending_punctuation(w) for w
                    in words if not w.isdigit()
                ]
                counter.update(words)
        vocabulary = [w for w, c in counter.most_common(vocabulary_size)]
        vocabulary.sort()
        return tuple(vocabulary)

    def _parse(self, data):
        sentences = []
        still_in_the_same_story = False

        for l in data.splitlines():
            i, l = re.split(' ', l, maxsplit=1)
            i, l = int(i), l.strip()
            still_in_the_same_story = int(i) != 1
            try:
                query, answer, _ = l.split('\t')
                query = query.rstrip()
                query = self._remove_ending_punctuation(query)
                yield copy.copy(sentences), query, answer
            except ValueError:
                if not still_in_the_same_story:
                    sentences.clear()
                sentences.append(l)

    def _task_data(self, task, train=True):
        startswith = 'qa{task}_'.format(task=task)
        endswith = '_{train_or_test}.txt'.format(train_or_test=(
            'train' if train else 'test'
        ))

        try:
            with open([
                    os.path.join(self._path_to_dataset, n) for n in
                    os.listdir(self._path_to_dataset) if
                    n.startswith(startswith) and
                    n.endswith(endswith)
            ][0]) as f:
                return f.read()

        except IndexError:
            raise FileNotFoundError

    def _download(self):
        if not os.path.exists(self._path):
            os.makedirs(self._path)
        # Check if the dataset already exists or not.
        elif os.listdir(self._path):
            print()
            print('=> Using the dataset in "{dir}" for "{target}"'.format(
                dir=self._path, target='{dataset_name}-{train}'
                .format(
                    dataset_name=self._dataset_name,
                    train=('train' if self._train else 'test')
                )
            ))
            return

        fa = FakeUserAgent()
        stream = requests.get(
            self._URL, stream=True, timeout=3,
            headers={'user-agent': fa.chrome}
        )

        if not stream.ok:
            raise RuntimeError(
                '=> {url} returned {code} for following reason: {reason}'
                .format(
                    url=self._URL,
                    code=stream.status_code,
                    reason=stream.reason
                ))

        total_size = int(stream.headers.get('content-length', 0))
        chunk_size = 1024 * 32
        stream_with_progress = tqdm(
            stream.iter_content(chunk_size=chunk_size),
            total=total_size//chunk_size,
            unit='KiB',
            unit_scale=True,
            desc='=> Downloading a bAbI QA dataset',
        )

        download_path = self._path + '.tar.gz'
        with open(download_path, 'wb') as f:
            for chunk in stream_with_progress:
                f.write(chunk)
            shutil.copyfileobj(stream.raw, f)
            stream.close()

        with _progress('=> Extracting... '):
            os.system('tar -xzf {tar} --strip-components=1 -C {dest}'.format(
                tar=download_path, dest=self._path
            ))
            os.remove(download_path)
