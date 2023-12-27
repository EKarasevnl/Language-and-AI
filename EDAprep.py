import nltk
from sklearn.base import TransformerMixin
from nltk.corpus import wordnet
import random
nltk.download('wordnet')

class SynonymReplacementTransformer(TransformerMixin):
    def __init__(self, p=0.2):
        self.p = p

    def synonym_replacement(self, sentence):
        words = sentence.split()

        self.n = int(self.p * len(sentence))
        for _ in range(self.n):
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            synonyms = [syn.name() for syn in wordnet.synsets(word)]
            if synonyms:
                replacement = random.choice(synonyms)
                words[idx] = replacement
        return ' '.join(words)

    def transform(self, X, y=None):
        return [self.synonym_replacement(sentence) for sentence in X]
    
class RandomInsertionTransformer(TransformerMixin):
    def __init__(self, p=0.2):
        self.p = p

    def random_insertion(self, sentence):
        words = sentence.split()
        self.n = int(self.p * len(sentence))
        for _ in range(self.n):
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            
            # Get synonyms of the word that are not stop words
            synonyms = [syn.name() for syn in wordnet.synsets(word) if syn.name() not in nltk.corpus.stopwords.words('english')]
            
            if synonyms:
                synonym = random.choice(synonyms)
                words.insert(random.randint(0, len(words)), synonym)
        return ' '.join(words)

    def transform(self, X, y=None):
        return [self.random_insertion(sentence) for sentence in X]

class RandomSwapTransformer(TransformerMixin):
    def __init__(self, p=0.2):
        self.p = p

    def random_swap(self, sentence):
        words = sentence.split()
        self.n = int(self.p * len(sentence))
        for _ in range(self.n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

    def transform(self, X, y=None):
        return [self.random_swap(sentence) for sentence in X]

class RandomDeletionTransformer(TransformerMixin):
    def __init__(self, p=0.2):
        self.p = p

    def random_deletion(self, sentence):
        words = sentence.split()
        words = [word for word in words if random.uniform(0, 1) > self.p]
        return ' '.join(words)

    def transform(self, X, y=None):
        return [self.random_deletion(sentence) for sentence in X]
