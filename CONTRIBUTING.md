# Руководство по вкладу в проект

Спасибо за интерес к проекту BPLA Person Detection! Это руководство поможет вам внести вклад в проект.

## Содержание

- [Code of Conduct](#code-of-conduct)
- [Как внести вклад](#как-внести-вклад)
- [Настройка окружения разработки](#настройка-окружения-разработки)
- [Структура кода](#структура-кода)
- [Стиль кода](#стиль-кода)
- [Тестирование](#тестирование)
- [Pull Request процесс](#pull-request-процесс)
- [Документация](#документация)

## Code of Conduct

Этот проект следует [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Участвуя, вы соглашаетесь соблюдать этот кодекс.

## Как внести вклад

1. **Форкните репозиторий** на GitHub
2. **Клонируйте ваш форк** локально:

   ```bash
   git clone https://github.com/YOUR-USERNAME/bpla_person_detection.git
   cd bpla_person_detection
   ```

3. **Создайте ветку** для ваших изменений:

   ```bash
   git checkout -b feature/amazing-new-feature
   # или
   git checkout -b fix/annoying-bug
   ```

4. **Внесите изменения** в код
5. **Запустите тесты** (см. [Тестирование](#тестирование))
6. **Создайте Pull Request**

## Настройка окружения разработки

### Требования

- **Julia** 1.9 или выше
- **Git**
- **Docker** (для тестирования deployment)
- **Visual Studio Code** (рекомендуется) с расширением Julia

### Установка Julia пакета

```bash
# Запустите Julia REPL
julia

# Перейдите в режим пакета
] 

# Активируйте проект
(dev) ] activate .

# Установите зависимости
(dev) ] instantiate

# Выйдите из режима пакета
(dev) ] <backspace>
```

### Запуск Julia с проектом

```bash
# Из корня проекта
julia --project=.

# Или для разработки
julia

] dev .
using BPLADetection
```

## Структура кода

### Основные модули

| Файл                    | Модуль          | Описание                                   |
|-------------------------|-----------------|--------------------------------------------|
| `src/BPLADetection.jl`  | `BPLADetection` | Главный модуль, экспортирует публичный API |
| `src/Models.jl`         | `Models`        | RussianYOLO модель и архитектура           |
| `src/Inference.jl`      | `Inference`     | Inference engine (CPU)                     |
| `src/Training.jl`       | `Training`      | Training                                   |
| `src/serve.jl`          | `Serve`         | REST API сервер                            |
| `src/serve_grpc.jl`     | -               | gRPC сервер                                |

### Core модули (production features)

```text
core/
├── circuit_breaker.R   # Circuit breaker паттерн
├── exceptions.R        # Кастомные исключения
├── logging.R          # Структурированное логирование
├── metrics.R          # Prometheus метрики
├── rate_limiter.R     # Rate limiting
├── redis_cache.R      # Redis кэширование
└── telemetry.R        # Distributed tracing
```

### Добавление нового функционала

1. **Новый модуль**: Создайте файл в `src/your_module.jl`
2. **Зарегистрируйте** в `BPLADetection.jl`
3. **Экспортируйте** публичные функции
4. **Добавьте тесты** в `test/runtests.jl`

Пример:

```julia
# src/YourModule.jl
module YourModule

export your_function

"""
Документация вашей функции
"""
function your_function(arg::Type)
    # Реализация
    return result
end

end  # module
```

## Стиль кода

### Julia Style Guide

Мы следуем стандартам Julia сообщества:

- **Именование**: camelCase для переменных, PascalCase для типов
- **Отступы**: 4 пробела
- **Документация**: Все публичные функции должны иметь docstrings
- **Типизация**: Явные типы для аргументов публичных функций

### Форматирование кода

Используйте JuliaFormatter:

```bash
] add JuliaFormatter
using JuliaFormatter
format("src/")
format("test/")
```

### Docstrings

Все публичные функции должны иметь документацию:

```julia
"""
    your_function(arg::Type) -> ReturnType

Краткое описание функции.

# Arguments
- `arg::Type`: Описание аргумента

# Returns
- `ReturnType`: Описание возвращаемого значения

# Examples
```julia
julia> your_function(value)
result
```

"""
function your_function(arg::Type)
    # Реализация
end

## Тестирование

### Запуск всех тестов

```bash
] test
```

### Запуск конкретного testset

```julia
] test
julia> include("test/runtests.jl")
julia> @testset "Models Module" begin ... end
```

### Написание тестов

```julia
@testset "Your Module" begin
    
    @testset "1. Function Name" begin
        # Тест с утверждениями
        @test your_function(input) == expected
        @test your_function(invalid) throws(ErrorException)
    end
    
end
```

### Покрытие кода

Для проверки покрытия кода тестами:

```bash
] test --coverage
```

## Pull Request процесс

### before creating PR

1. **Обновите вашу ветку** с последними изменениями из main:

   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   git checkout your-branch
   git merge main
   ```

2. **Убедитесь, что все тесты проходят**:

   ```bash
   ] test
   ```

3. **Проверьте форматирование**:

   ```bash
   using JuliaFormatter
   format("src/")
   format("test/")
   ```

### Создание PR

1. **Заголовок**: Краткое описание изменений
2. **Описание**:
   - Что было изменено
   - Почему это изменение необходимо
   - Как тестировалось
3. **Скриншоты**: Если есть визуальные изменения

### Review процесс

1. PR должны быть reviewed минимум одним мейнтейнером
2. Все CI checks должны пройти
3. Конфликты должны быть разрешены

## Документация

### Обновление README

Если вы добавили новую функцию:

1. Обновите раздел с описанием
2. Добавьте примеры использования
3. Обновите таблицу API endpoints

### Добавление в ARCHITECTURE.md

Для архитектурных изменений:

1. Обновите диаграммы
2. Добавьте описание новых компонентов
3. Обновите раздел с производительностью

### Документирование API

Каждый публичный API должен быть задокументирован:

```julia
detect_people(image_path::String; conf_threshold=0.25, iou_threshold=0.5) -> Vector{Detection}

Детекция людей на изображении с БПЛА.

# Arguments
- `image_path::String`: Путь к изображению (поддерживаются форматы: jpg, png, bmp)
- `conf_threshold::Float64`: Порог уверенности (по умолчанию 0.25)
- `iou_threshold::Float64`: Порог IoU для NMS (по умолчанию 0.5)

# Returns
- `Vector{Detection}`: Вектор обнаруженных объектов с полями:
  - `bbox::BoundingBox`: координаты рамки [x1, y1, x2, y2]
  - `score::Float64`: уверенность
  - `class_id::Int`: ID класса (1 - person)
  - `class_name::String`: имя класса ("person")

# Examples
```julia
# Простая детекция
result = detect_people("drone_shot.jpg")

# С изменёнными порогами
result = detect_people("night_vision.jpg", conf_threshold=0.15, iou_threshold=0.3)

# Обработка результата
for det in result
    println("Найден человек с уверенностью $(det.score)")
end
```

## Notes

Дополнительные замечания
"""

```text

## Руководство по git

### Ветки

- `main`: Production-ready код
- `develop`: Development branch
- `feature/*`: Новые функции
- `fix/*`: Исправления багов
- `refactor/*`: Рефакторинг

### Сообщения коммитов

```

type(scope): subject

body (optional)

footer (optional)

```text

Типы:
- `feat`: Новая функция
- `fix`: Исправление бага
- `docs`: Изменения в документации
- `style`: Форматирование кода
- `refactor`: Рефакторинг
- `test`: Добавление тестов
- `chore`: Обслуживание

Примеры:
```

feat(inference): add batch inference support
fix(nms): correct IoU calculation for edge cases
docs(readme): update API documentation

```text

## Получение помощи

- **Issues**: [GitHub Issues](https://github.com/Artyomkun/bpla_person_detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Artyomkun/bpla_person_detection/discussions)
- **Email**: [selemetartem@gmail.com](selemetartem@gmail.com)
- **Telegram**: [@Artyomkun](https://t.me/Artyomkun)

## Спасибо!

Спасибо за ваш вклад в проект BPLA Person Detection!
