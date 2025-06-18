from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple

import numpy as np

from preprocessing import TokenizedSentencePair


class BaseAligner(ABC):
    """
    Describes a public interface for word alignment models.
    """

    @abstractmethod
    def fit(self, parallel_corpus: List[TokenizedSentencePair]):
        """
        Estimate alignment model parameters from a collection of parallel sentences.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
        """
        pass

    @abstractmethod
    def align(self, sentences: List[TokenizedSentencePair]) -> List[List[Tuple[int, int]]]:
        """
        Given a list of tokenized sentences, predict alignments of source and target words.

        Args:
            sentences: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            alignments: list of alignments for each sentence pair, i.e. lists of tuples (source_pos, target_pos).
            Alignment positions in sentences start from 1.
        """
        pass


class DiceAligner(BaseAligner):
    def __init__(self, num_source_words: int, num_target_words: int, threshold=0.5):
        self.cooc = np.zeros((num_source_words, num_target_words), dtype=np.uint32)
        self.dice_scores = None
        self.threshold = threshold

    def fit(self, parallel_corpus):
        for sentence in parallel_corpus:
            # use np.unique, because for a pair of words we add 1 only once for each sentence
            for source_token in np.unique(sentence.source_tokens):
                for target_token in np.unique(sentence.target_tokens):
                    self.cooc[source_token, target_token] += 1
        self.dice_scores = (2 * self.cooc.astype(np.float32) /
                            (self.cooc.sum(0, keepdims=True) + self.cooc.sum(1, keepdims=True)))

    def align(self, sentences):
        result = []
        for sentence in sentences:
            alignment = []
            for (i, source_token), (j, target_token) in product(
                    enumerate(sentence.source_tokens, 1),
                    enumerate(sentence.target_tokens, 1)):
                if self.dice_scores[source_token, target_token] > self.threshold:
                    alignment.append((i, j))
            result.append(alignment)
        return result


class WordAligner(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters

    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        """
        Given a parallel corpus and current model parameters, get a posterior distribution over alignments for each
        sentence pair.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            posteriors: list of np.arrays with shape (src_len, target_len). posteriors[i][j][k] gives a posterior
            probability of target token k to be aligned to source token j in a sentence i.
        """
        posteriors = []
        for sentence in parallel_corpus:
            probs = self.translation_probs[np.ix_(sentence.source_tokens, sentence.target_tokens)]
            posteriors.append(probs / probs.sum(axis=0))
        return posteriors

    def _compute_elbo(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]) -> float:
        """
        Compute evidence (incomplete likelihood) lower bound for a model given data and the posterior distribution
        over latent variables.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo: the value of evidence lower bound
        """
        elbo = 0
        for sentence, posterior in zip(parallel_corpus, posteriors):
            probs = self.translation_probs[np.ix_(sentence.source_tokens, sentence.target_tokens)]
            elbo += np.sum(posterior * np.log(probs / len(sentence.source_tokens) / (posterior + 1e-9) + 1e-9))
        return elbo

    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo:  the value of evidence lower bound after applying parameter updates
        """
        self.translation_probs *= 0
        for sentence, posterior in zip(parallel_corpus, posteriors):
            np.add.at(self.translation_probs, np.ix_(sentence.source_tokens, sentence.target_tokens), posterior)
        self.translation_probs /= np.sum(self.translation_probs, axis=1, keepdims=True)
        return self._compute_elbo(parallel_corpus, posteriors)

    def fit(self, parallel_corpus):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.
        Sorry for not sticking to my own interface ;)

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for i in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)
        return history

    def align(self, sentences):
        result = []
        posteriors = self._e_step(sentences)
        for posterior in posteriors:
            alignment = []
            for j in range(posterior.shape[1]):
                alignment.append((np.argmax(posterior[:, j]) + 1, j + 1))
            result.append(alignment)
        return result


class WordPositionAligner(WordAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.alignment_probs = {}

    def _get_probs_for_lengths(self, src_length: int, tgt_length: int):
        """
        Given lengths of a source sentence and its translation, return the parameters of a "prior" distribution over
        alignment positions for these lengths. If these parameters are not initialized yet, first initialize
        them with a uniform distribution.

        Args:
            src_length: length of a source sentence
            tgt_length: length of a target sentence

        Returns:
            probs_for_lengths: np.array with shape (src_length, tgt_length)
        """
        if (src_length, tgt_length) not in self.alignment_probs:
            self.alignment_probs[(src_length, tgt_length)] = np.full((src_length, tgt_length), 1 / src_length,
                                                                     dtype=np.float32)
        return self.alignment_probs[(src_length, tgt_length)]

    def _e_step(self, parallel_corpus):
        posteriors = []
        for sentence in parallel_corpus:
            phi = self._get_probs_for_lengths(len(sentence.source_tokens), len(sentence.target_tokens))
            probs = self.translation_probs[np.ix_(sentence.source_tokens, sentence.target_tokens)]
            posterior = phi * probs
            posteriors.append(posterior / (posterior.sum(axis=0, keepdims=True) + 1e-9))
        return posteriors

    def _compute_elbo(self, parallel_corpus, posteriors):
        elbo = 0
        for sentence, posterior in zip(parallel_corpus, posteriors):
            phi = self._get_probs_for_lengths(len(sentence.source_tokens), len(sentence.target_tokens))
            probs = self.translation_probs[np.ix_(sentence.source_tokens, sentence.target_tokens)]
            elbo += np.sum(posterior * np.log(phi * probs / (posterior + 1e-9) + 1e-9))
        return elbo

    def _m_step(self, parallel_corpus, posteriors):
        self.translation_probs *= 0
        self.alignment_probs = {}
        for sentence, posterior in zip(parallel_corpus, posteriors):
            np.add.at(self.translation_probs, np.ix_(sentence.source_tokens, sentence.target_tokens), posterior)
            if (len(sentence.source_tokens), len(sentence.target_tokens)) not in self.alignment_probs:
                self.alignment_probs[len(sentence.source_tokens), len(sentence.target_tokens)] = np.zeros(
                    (len(sentence.source_tokens), len(sentence.target_tokens)), dtype=np.float32)
            self.alignment_probs[len(sentence.source_tokens), len(sentence.target_tokens)] += posterior
        for key, value in self.alignment_probs.items():
            self.alignment_probs[key] /= value.sum(axis=0, keepdims=True)
        self.translation_probs /= (np.sum(self.translation_probs, axis=1, keepdims=True) + 1e-9)
        return self._compute_elbo(parallel_corpus, posteriors)
