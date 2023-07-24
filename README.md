# SimpleLanguageModel
Этот проект содержит примеры рекуррентных нейронных сетей (RNN), способных генерировать текст на основе данных из датасета IMDb. Проект включает в себя две модели: GRU и LSTM.

## Структура проекта
```bash
│
├── src
│   ├── config.py          # файл конфигурации проекта
│   ├── model_training.py  # функции для обучения и оценки моделей
│   ├── nn_models.py       # описание моделей GRU и LSTM
│   ├── utils.py           # полезные функции для подготовки данных
│   └── word_dataset.py    # класс Dataset для работы с датасетом IMDb
│
├── models
│   ├── model_lstm.pth     # обученная модель LSTM
│   ├── model_gru.pth      # обученная модель GRU
│
├── main.py                # главный файл проекта
└── requirements.txt       # файл со списком зависимостей
```

## Как использовать
1. Убедитесь, что у вас установлены все необходимые библиотеки из файла requirements.txt. Вы можете установить их, выполнив следующую команду:
```
pip install -r requirements.txt
```
2. Запустите файл main.py, чтобы начать обучение и генерацию текста:
```
python main.py
```
