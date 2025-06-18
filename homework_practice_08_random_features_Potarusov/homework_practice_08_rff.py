import numpy as np

from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from scipy.stats import norm
from scipy.fftpack import fft, ifft


def estimate_sigma2(X, sample_pairs: int = 10000):
    idx1 = np.random.randint(0, X.shape[0], size=sample_pairs)
    idx2 = np.random.randint(0, X.shape[0], size=sample_pairs)
    diffs = X[idx1] - X[idx2]
    sq_dists = np.sum(diffs ** 2, axis=1)
    sigma2 = np.median(sq_dists)
    return sigma2


class FeatureCreatorPlaceholder(BaseEstimator, TransformerMixin):
    def __init__(self, n_features, new_dim, func: Callable = np.cos):
        self.n_features = n_features
        self.new_dim = new_dim
        self.w = None
        self.b = None
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class RandomFeatureCreator(FeatureCreatorPlaceholder):
    def fit(self, X, y=None):
        sigma2 = estimate_sigma2(X)
        d = X.shape[1]
        self.w = np.random.normal(loc=0, scale=np.sqrt(1.0 / sigma2), size=(d, self.n_features))
        self.b = np.random.uniform(low=-np.pi, high=np.pi, size=self.n_features)
        return self

    def transform(self, X, y=None):
        Z = self.func(np.dot(X, self.w) + self.b)
        return Z


class OrthogonalRandomFeatureCreator(RandomFeatureCreator):
    def fit(self, X, y=None):
        sigma2 = estimate_sigma2(X)
        d = X.shape[1]
        if self.n_features <= d:
            G = np.random.normal(loc=0, scale=1.0, size=(d, self.n_features))
            Q, _ = np.linalg.qr(G)
            self.w = Q * np.sqrt(1.0/sigma2)
        else:
            scale = np.sqrt(d) / np.sqrt(sigma2)
            m = int(np.ceil(self.n_features / d))
            blocks = []
            for i in range(m):
                G = np.random.normal(loc=0.0, scale=1.0, size=(d, d))
                Q, _ = np.linalg.qr(G)
                blocks.append(Q)
            self.w = np.hstack(blocks)[:, :self.n_features] * scale
        self.b = np.random.uniform(low=-np.pi, high=np.pi, size=self.n_features)
        return self


class TensorSketchFeatureCreator(FeatureCreatorPlaceholder):
    def __init__(self, n_features=1000, new_dim=None, func=None, p=2):
        super().__init__(n_features=n_features, new_dim=new_dim, func=func)
        self.p = p        
        self.D = n_features
        self.hashes_h = [] 
        self.hashes_s = [] 

    def _init_hash_functions(self, d):
        prime = 998244353
        self.hashes_h = []
        self.hashes_s = []
        for _ in range(self.p):
            a = np.random.randint(1, prime)
            b = np.random.randint(0, prime)
            self.hashes_h.append((a, b))

            a_s = np.random.randint(1, prime)
            b_s = np.random.randint(0, prime)
            self.hashes_s.append((a_s, b_s))

    def _hash_h(self, idx, k):
        a, b = self.hashes_h[k]
        return ((a * idx + b) % 998244353) % self.D

    def _hash_s(self, idx, k):
        a, b = self.hashes_s[k]
        return 1 if ((a * idx + b) % 998244353) % 2 == 0 else -1

    def fit(self, X, y=None):
        self._init_hash_functions(X.shape[1])
        return self

    def transform(self, X, y=None):        
        tensor_sketches = np.zeros((len(X), self.D), dtype=np.complex128)
        for i in range(len(X)):
            x = X[i]
            sketches = []
            for k in range(self.p):
                C = np.zeros(self.D, dtype=np.complex128)
                for j in range(X.shape[1]):
                    h = self._hash_h(j, k)
                    s = self._hash_s(j, k)
                    C[h] += s * x[j]
                sketches.append(fft(C))
            product = np.prod(sketches, axis=0)
            tensor_sketches[i] = ifft(product)
        return np.real(tensor_sketches) / np.sqrt(self.D)


class RFFPipeline(BaseEstimator):
    """
    Пайплайн, делающий последовательно три шага:
        1. Применение PCA
        2. Применение RFF
        3. Применение классификатора
    """
    def __init__(
            self,
            n_features: int = 1000,
            new_dim: int = 50,
            use_PCA: bool = True,
            feature_creator_class=FeatureCreatorPlaceholder,
            classifier_class=LogisticRegression,
            classifier_params=None,
            func=np.cos,
            regression: bool = False
    ):
        """
        :param n_features: Количество признаков, генерируемых RFF
        :param new_dim: Количество признаков, до которых сжимает PCA
        :param use_PCA: Использовать ли PCA
        :param feature_creator_class: Класс, создающий признаки, по умолчанию заглушка
        :param classifier_class: Класс классификатора
        :param classifier_params: Параметры, которыми инициализируется классификатор
        :param func: Функция, которую получает feature_creator при инициализации.
                     Если не хотите, можете не использовать этот параметр.
        """
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        if classifier_params is None:
            classifier_params = {}
        if regression:
            self.classifier = Ridge(**classifier_params)
        else:
            self.classifier = classifier_class(**classifier_params)
        self.feature_creator = feature_creator_class(
            n_features=self.n_features, new_dim=self.new_dim, func=func
        )
        self.pipeline = None

    def fit(self, X, y):
        pipeline_steps: list[tuple] = []
        if self.use_PCA:
            self.new_dim = X.shape[1]
            pipeline_steps.append(('pca', PCA(n_components=self.new_dim, random_state=42)))
        pipeline_steps.append(('rff', self.feature_creator))
        pipeline_steps.append(('clf', self.classifier))
        self.pipeline = Pipeline(pipeline_steps).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)
