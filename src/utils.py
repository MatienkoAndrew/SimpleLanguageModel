from collections import Counter
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm.auto import tqdm
from typing import List
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from src.config import CONFIG

from .word_dataset import WordDataset


def load_data():
    '''Загрузка датасета'''
    dataset = load_dataset('imdb')
    return dataset


def preprocess_data(dataset, word_theshold: int=30) -> List[str]:
    '''Препроцессинг данных'''
    sentences = []

    for sentence in tqdm(dataset['train']['text']):
        sentences.extend([
            x.lower() for x in sent_tokenize(sentence) if len(word_tokenize(x)) < word_theshold
        ])
    return sentences


def count_words(sentences: List[str]) -> dict:
    '''Подсчет слов во всех предложениях'''
    words = Counter()

    # Расчет встречаемости слов
    for sentence in tqdm(sentences):
        for word in word_tokenize(sentence):
            words[word] += 1
    return words


def create_vocab(words: dict, special_tokens: List[str]=['<unk>', '<bos>', '<eos>', '<pad>'], vocab_size: int=40_000) -> set:
    '''Создание словаря'''
    vocab = set(special_tokens)
    most_common_words = words.most_common(vocab_size)
    for word, freq in tqdm(most_common_words):
        vocab.add(word)
    # print("Топ-5 самых встречающихся слов:", most_common_words[:5])
    return vocab


def prepare_data(sentences: List[str], word2ind: dict):
    '''Подготовка данных'''
    train_sentences, eval_sentences = train_test_split(sentences, test_size=0.2)
    # eval_sentences, test_sentences = train_test_split(sentences, test_size=0.5)

    train_dataset = WordDataset(train_sentences, word2ind)
    eval_dataset = WordDataset(eval_sentences, word2ind)
    # test_dataset = WordDataset(test_sentences)

    train_dataloader = DataLoader(
        train_dataset, 
        collate_fn=lambda batch: collate_fn_with_padding(batch, word2ind['<pad>'], CONFIG['device']), 
        batch_size=CONFIG['batch_size']
        )

    eval_dataloader = DataLoader(
        eval_dataset, 
        collate_fn=lambda batch: collate_fn_with_padding(batch, word2ind['<pad>'], CONFIG['device']), 
        batch_size=CONFIG['batch_size']
        )
    return (train_dataloader, eval_dataloader)


def collate_fn_with_padding(input_batch: List[List[int]], pad_id, device) -> torch.Tensor:
    """
    Функция выполняет заполнение и выравнивание последовательностей, 
    чтобы они имели одинаковую длину.
    """
    seq_lens = [len(x) for x in input_batch]
    max_seq_len = max(seq_lens)

    new_batch = []
    for sequence in input_batch:
        for _ in range(max_seq_len - len(sequence)):
            sequence.append(pad_id)
        new_batch.append(sequence)

    sequences = torch.LongTensor(new_batch).to(device)

    new_batch = {
        'input_ids': sequences[:,:-1],
        'target_ids': sequences[:,1:]
    }

    return new_batch
