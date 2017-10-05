import atexit
import collections
import copy
import re
import os
import os.path
import shutil
import tempfile
import numpy as np
import requests
from tqdm import tqdm
from fake_useragent import FakeUserAgent
from torch.utils.data import Dataset


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
                 task_cache_size=2,
                 train=True, download=True, path='./datasets'):
        self._dataset_name = dataset_name
        self._tasks = tasks or [i+1 for i in range(20)]
        self._train = train
        self._path = os.path.join(path, self._DIRNAME)
        self._path_to_dataset = os.path.join(self._path, self._dataset_name)
        self._path_to_tempfiles = self._path + '-temp'

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
            train=self._train,
        )

        # cleanup loaded data from the disk.
        atexit.register(self._cleanup_disk)

    def __getitem__(self, index):
        path = self._paths[index]
        loaded = np.load(path)
        return loaded['sentences'], loaded['query'], loaded['answer']

    def __len__(self):
        return len(self._paths)

    @property
    def vocabulary_size(self):
        return len(self._vocabulary)

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

    def _load_to_disk(self, tasks, sentence_size, train=True):
        if not os.path.exists(self._path_to_tempfiles):
            os.makedirs(self._path_to_tempfiles)

        paths = []
        for task in tqdm(tasks, desc='=> Loading the data to the disk'):
            data = self._task_data(task, train=train)
            parsed = self._parse(data)
            for i, x in enumerate(parsed):
                # Load the data to the disk and retain their paths instaed of
                # loading them to the memory directly. This is to prevent OOM
                # error. The temporary files will be discarded on exit by the
                # `~_cleanup_disk()` callback.
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, dir=self._path_to_tempfiles
                )
                sentences, query, answer = self._encode_x(
                    x, sentence_size=sentence_size
                )
                np.savez(tmp, sentences=sentences, query=query, answer=answer)
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
        )[0]
        encoded_answer = self._encode_words(
            answer, sentence_size=sentence_size
        )[0]
        return encoded_sentences, encoded_query, encoded_answer

    def _encode_words(self, *words, sentence_size):
        paddings = (self._PADDING,) * (sentence_size-len(words))
        indices = np.array([self.word2idx(w) for w in words + paddings])
        onehot = np.zeros((sentence_size, self.vocabulary_size))
        onehot[np.arange(sentence_size), indices] = 1
        return onehot.astype(np.uint8)

    def _generate_vocabulary(self, tasks, vocabulary_size, train=True):
        print('=> Generating a vocabulary... ', end='')
        counter = collections.Counter()
        for task in tasks:
            words = self._task_data(task, train=train).split()
            words = [self._remove_ending_punctuation(w) for w in words]
            counter.update(words)
        print('Done')
        return [w for w, c in counter.most_common(vocabulary_size)]

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
        fa = FakeUserAgent()
        stream = requests.get(
            self._URL, stream=True,
            headers={'user-agent': fa.chrome}
        )

        if not os.path.exists(self._path):
            os.makedirs(self._path)
        elif os.listdir(self._path):
            print("=> Using the existing dataset in '{dir}'".format(
                dir=self._path
            ))
            return

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
            desc='=> Downloading bAbI QA dataset',
        )

        download_path = self._path + '.tar.gz'
        with open(download_path, 'wb') as f:
            for chunk in stream_with_progress:
                f.write(chunk)
            shutil.copyfileobj(stream.raw, f)
            stream.close()

        print('=> Extracting... ', end='', flush=True)
        os.system('tar -xzf {tar} --strip-components=1 -C {dest}'.format(
            tar=download_path, dest=self._path
        ))
        os.remove(download_path)
        print('Done')
