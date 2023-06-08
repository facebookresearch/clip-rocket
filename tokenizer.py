# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from github.com/openai/CLIP
import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re
import torch
from textaugment import EDA
import random
from nltk.tokenize import word_tokenize
import nltk

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(
            self, 
            bpe_path: str = default_bpe(),
            text_augment=False,
            no_text_augment_prob=0.0,
            remove_stopwords_prob=0.5,
            synonym_replacement_prob=0.2,
            random_swap_prob=0.2,
            random_deletion_prob=0.1,
            clean_before_augment=False,
            num_augs=2,
        ):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
        self.clean_before_augment = clean_before_augment
        self.remove_stopwords_prob = remove_stopwords_prob
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.remove_stopwords = lambda x:  " ".join([w for w in word_tokenize(x) if w.lower() not in self.stopwords])
        if text_augment:
            eda = EDA()
            identity = lambda x: x
            self.text_augment = lambda x: random.choices(
                [
                    identity,
                    eda.synonym_replacement,
                    eda.random_swap,
                    eda.random_deletion
                ],
                weights=[
                    no_text_augment_prob,
                    synonym_replacement_prob,
                    random_swap_prob,
                    random_deletion_prob
                ],
                k=1
            )[0](x)
        else:
            self.text_augment = None
        self.num_augs = num_augs

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def weak_augment(self, text):
        if len(text) == 0:
            return text
        if random.random() < self.remove_stopwords_prob:
            stripped_texts = self.remove_stopwords(text)
            text = stripped_texts if len(stripped_texts) > 0 else text
        return text

    def strong_augment(self, text):
        if len(text) == 0:
            return text
        if random.random() < self.remove_stopwords_prob:
            stripped_texts = self.remove_stopwords(text)
            text = stripped_texts if len(stripped_texts) > 0 else text
        if self.text_augment is not None:
            augmented_texts = self.text_augment(text)
            augmented_texts = augmented_texts[0] if isinstance(augmented_texts, list) else augmented_texts
            text = augmented_texts if len(augmented_texts) > 0 else text
        return text

    def __call__(self, texts, context_length=77):

        if isinstance(texts, tuple): # training time
            texts = list(texts)
            if self.clean_before_augment:
                for i, txt in enumerate(texts):
                    texts[i] = whitespace_clean(basic_clean(txt)).lower()
            texts = [
                self.weak_augment(random.choice(texts)),
                *[self.strong_augment(random.choice(texts)) for _ in range(self.num_augs)],
            ]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            tokens = tokens[:context_length]
            if tokens[-1] != eot_token:
                tokens[-1] = eot_token
            result[i, :len(tokens)] = torch.tensor(tokens)

        if len(result) == 1:
            return result[0]
        return result