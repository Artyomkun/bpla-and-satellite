# BPLA Person Detection

Система детекции людей на снимках с БПЛА (беспилотных летательных аппаратов) и спутников для поиска пропавших без вести.

## Описание

Приложение представляет собой веб-интерфейс на базе Genie.jl для детекции людей на изображениях и видео с спутников с использованием Julia ML моделей.

## Функциональность

- **Детекция на изображениях** - загрузка и обработка отдельных изображений
- **Детекция на видео** - обработка видеопотоков с спутников
- **Российские модели** - поддержка интеграции с локальными моделями
- **Docker-контейнеризация** - готовность к развёртыванию в production

## Установка

```bash
# Клонирование репозитория
git clone <repo-url>
cd bpla_person_detection

# Установка Julia пакета
] dev .

# Или из репозитория
] add https://github.com/Artyomkun/bpla_person_detection.git
```

## Запуск

```bash
# Запуск Julia приложения
julia --project app.jl

# Или через Julia REPL
] enter
julia> include("app.jl")
```

Приложение будет доступно по адресу <http://127.0.0.1:8000>

## Docker

```bash
# Сборка образа
docker build -t bpla-person-detection bpla_person_detection/

# Запуск контейнера
docker run -p 8000:8000 bpla-person-detection
```

## 🖥️ Веб-интерфейс (Genie.jl)

Для быстрого тестирования доступен веб-интерфейс:

```bash
julia --project app.jl
Открыть http://127.0.0.1:8000
```

## CI/CD

Проект использует GitHub Actions для:

- Проверки форматирования (JuliaFormatter)
- Тестирования (Pkg.test())
- Сборки Docker-образа
- Security-сканирования

## Структура проекта

```text
bpla_person_detection/
├── deploy/                         # Kubernetes / Helm
└── src/                          # Основной код (Julia)
    ├── __main__.jl                # CLI entry point
    ├── app.jl                     # Genie app
    ├── BPLADetection.jl           # Главный модуль
    ├── Config.jl                  # Конфигурация
    ├── Dockerfile                 # Docker образ
    ├── docker-compose.yml         # Docker Compose
    ├── Inference.jl               # Inference engine (CPU)
    ├── Models.jl                  # RussianYOLO модель
    ├── Project.toml               # Julia dependencies
    ├── serve.jl                   # REST API сервер
    ├── Training.jl                # Training pipeline
    ├── LICENSE                    # Лицензия
    ├── README.md                  # Документация модуля
    │
    ├── train_project/              # Тренировка моделей
    │   └── Training.jl
    │
    ├── core/                       # Production-модули
    │
    ├── serve/
    │   └── serve.jl                  # Сервер
    │
    └── models/                        # Модели
```

## Модели

Проект поддерживает несколько источников моделей:

1. **Локальные веса** - обученные модели в формате .jls (Julia serialized)
2. **ONNX модели** - через ONNXRuntime.jl
3. **PyTorch модели** - через PyTorch.jl

## Конфигурация

Параметры приложения задаются в `Config.jl`:

- `CLASSES` - классы для детекции
- `DEVICE` - устройство для инференса (auto/cpu/gpu)
- `NUM_CLASSES` - количество классов
- `THRESHOLD_DEFAULT` - порог уверенности по умолчанию

## Julia Packages

Основные зависимости:

- **Genie.jl** - веб-фреймворк
- **Flux.jl** - ML библиотека
- **Images.jl** - обработка изображений
- **FileIO.jl** - работа с файлами

## Безопасность

Проект разработан с учётом требований безопасности:

- Non-root контейнеры
- Dependency scanning в CI/CD
- Secrets management через Kubernetes

## Лицензия

MIT License

## Автор

Person Detection Project (c) 2026
