from tqdm.auto import tqdm
from typing import Tuple, List
import torch
from nltk.tokenize import word_tokenize


def train_model(model, optimizer, criterion, train_dataloader, eval_dataloader, n_epochs: int) -> Tuple[List[float], List[float]]:
    '''
    Обучение модели
    '''
    losses = []
    perplexities = []
    for epoch in range(n_epochs):
        epoch_losses = []
        model.train()
        for batch in tqdm(train_dataloader, desc=f'Training epoch {epoch}'):
            optimizer.zero_grad()
            logits = model(batch['input_ids']).flatten(start_dim=0, end_dim=1)
            loss = criterion(logits, batch['target_ids'].flatten())
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_epoch_loss)

        perplexity = evaluate(model, criterion, eval_dataloader)
        perplexities.append(perplexity)
        print(f"Avg loss: {avg_epoch_loss:.5f} | Perplexity: {perplexity:.5f}")

    return (losses, perplexities)


def evaluate(model, criterion, dataloader) -> float:
    """
    Оценка модели метрикой perplexity
    """
    model.eval()
    perplexity = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            logits = model(batch['input_ids']).flatten(start_dim=0, end_dim=1)
            loss = criterion(logits, batch['target_ids'].flatten())
            perplexity.append(torch.exp(loss).item())

    perplexity = sum(perplexity) / len(perplexity)
    return perplexity


def generate_sequence(model, word2ind: dict, ind2word: dict, starting_seq: str, max_seq_len: int = 128) -> str:
    """
    Функция генерирует текст
    """
    device = 'cpu'
    model = model.to(device)
    input_ids = [word2ind['<bos>']] + [
        word2ind.get(word, word2ind['<unk>']) for word in word_tokenize(starting_seq)]
    input_ids = torch.LongTensor(input_ids).to(device)

    model.eval()
    with torch.no_grad():
        for i in range(max_seq_len):
            next_char_distribution = model(input_ids)[-1]
            next_char = next_char_distribution.squeeze().argmax()
            input_ids = torch.cat([input_ids, next_char.unsqueeze(0)])

            if next_char.item() == word2ind['<eos>']:
                break

    words = ' '.join([ind2word[idx.item()] for idx in input_ids])

    return words
