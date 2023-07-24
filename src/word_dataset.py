from nltk.tokenize import word_tokenize
from typing import List

class WordDataset:
    def __init__(self, sentences, word2ind):
        self.data = sentences
        self.unk_id = word2ind['<unk>']
        self.bos_id = word2ind['<bos>']
        self.eos_id = word2ind['<eos>']
        self.pad_id = word2ind['<pad>']
        self.word2ind = word2ind

    def __getitem__(self, idx: int) -> List[int]:
        # Допишите код здесь
        sentence = self.data[idx]

        tokenized_sentence = word_tokenize(sentence)
        tokenized_sentence = [self.bos_id] + [self.word2ind.get(word, self.unk_id) for word in tokenized_sentence] + [self.eos_id]
        return tokenized_sentence

    def __len__(self) -> int:
        return len(self.data)
    
