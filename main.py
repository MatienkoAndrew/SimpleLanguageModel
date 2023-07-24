import torch
import torch.nn as nn
import seaborn
seaborn.set(palette='summer')

import nltk
nltk.download('punkt')

from src.utils import load_data, preprocess_data, count_words, create_vocab, prepare_data
from src.config import CONFIG
from src.model_training import train_model, generate_sequence
from src.nn_models import LanguageModelGRU, LanguageModelLSTM

device = CONFIG['device']


def main() -> int:
    dataset = load_data()

    sentences = preprocess_data(dataset, CONFIG['word_threshold'])
    print("Всего предложений:", len(sentences))

    words = count_words(sentences)
    print('Всего слов:', len(words))

    vocab = create_vocab(words, CONFIG['special_tokens'], CONFIG['vocab_size'])
    print("Всего слов в словаре:", len(vocab))

    ##-- подготовка данных
    word2ind = {char: i for i, char in enumerate(vocab)}
    ind2word = {i: char for i, char in enumerate(vocab)}

    (train_dataloader, eval_dataloader) = prepare_data(sentences, word2ind)

    # GRU: Первый эксперимент
    criterion = nn.CrossEntropyLoss(ignore_index=word2ind['<pad>'])
    model_gru = LanguageModelGRU(hidden_dim=256, vocab_size=len(vocab)).to(device)
    optimizer_gru = torch.optim.Adam(model_gru.parameters())

    losses, perplexities = train_model(model_gru, optimizer_gru, criterion, train_dataloader, eval_dataloader, n_epochs=5)

    # LSTM: второй эксперимент
    model_lstm = LanguageModelLSTM(hidden_dim=256, vocab_size=len(vocab)).to(device)
    optimizer_lstm = torch.optim.Adam(model_gru.parameters())

    losses, perplexities = train_model(model_lstm, optimizer_lstm, criterion, train_dataloader, eval_dataloader, n_epochs=5)
    generate_sequence(model_lstm, word2ind, ind2word, starting_seq='по системе гол плюс пас ')
    
    return 0


if __name__ == '__main__':
    main()
