from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    with open(filename, 'r') as f:
        string = f.read()
    string = string.replace('&', '&amp;')
    root = ET.fromstring(string)

    sentence_pairs = []
    alignments = []

    for elem in root:
        sentence_pairs.append(SentencePair(elem.find('english').text.split(),
                                           elem.find('czech').text.split()))
        sure_align = []
        poss_align = []
        sure_str = elem.find('sure').text
        poss_str = elem.find('possible').text
        if sure_str:
            for pair in sure_str.split():
                s, t = pair.split('-')
                sure_align.append((int(s), int(t)))
        if poss_str:
            for pair in poss_str.split():
                s, t = pair.split('-')
                poss_align.append((int(s), int(t)))
        alignments.append(LabeledAlignment(sure_align, poss_align))
    return sentence_pairs, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    src_counter = Counter()
    tgt_counter = Counter()
    for pair in sentence_pairs:
        src_counter.update(pair.source)
        tgt_counter.update(pair.target)

    if freq_cutoff is not None:
        src_vocab = [token for token, _ in src_counter.most_common(freq_cutoff)]
        tgt_vocab = [token for token, _ in tgt_counter.most_common(freq_cutoff)]
    else:
        src_vocab = list(src_counter.keys())
        tgt_vocab = list(tgt_counter.keys())

    source_dict = {token: idx for idx, token in enumerate(src_vocab)}
    target_dict = {token: idx for idx, token in enumerate(tgt_vocab)}
    return source_dict, target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_pairs = []
    for pair in sentence_pairs:
        src_indices = [source_dict[token] for token in pair.source if token in source_dict]
        tgt_indices = [target_dict[token] for token in pair.target if token in target_dict]
        if len(src_indices) > 0 and len(tgt_indices) > 0:
            tokenized_pairs.append(TokenizedSentencePair(np.array(src_indices), np.array(tgt_indices)))
    return tokenized_pairs
