# MOEX Trading Bots

Два независимых алгоритмических робота для Московской биржи:

- **🧲 @magnets_bot** — классификация свечных паттернов через метрику Минковского (акции MOEX)
- **📈 @futures_bot** — пробой каналов Bollinger / Keltner с Contango-фильтром (фьючерсы MOEX)

Оба используют Finam Arena API (торговля) + MOEX ISS (исторические данные).  
Каждый живёт в своей папке — никаких конфликтов модулей, можно запускать одновременно.

---

## Стратегии

### 🧲 @magnets_bot — Пространство Минковского (акции)

Классификация на основе kNN в пространстве Минковского.

| Компонент | Описание |
|-----------|----------|
| **5 признаков** | RSI, WT (WaveTrend), CCI, ADX (+ второй RSI) |
| **Классификатор** | kNN с метрикой Минковского, FIFO-буфер, динамический порог (75-й перцентиль) |
| **Фильтры** | Kernel Regression (Rational Quadratic), волатильность (ATR₁ vs SMA₂₀), ADX |
| **Выходы** | Strict (isHeldFourBars + isLastSignalBuy) или Dynamic (смена наклона ядра) |
| **Нормализация** | Кумулятивная (expanding min/max) — точно по Pine Script MLExtensions |
| **Сглаживание** | Wilder's RMA для RSI и ADX (не SMA) |

**Сигналы:** `BUY` / `SELL` / `CLOSE_LONG` / `CLOSE_SHORT`

**Инструменты:** 34 акции MOEX (список в `config.json` → `stocks`)

---

### 📈 @futures_bot — Bollinger / Keltner + Contango (фьючерсы)

Два варианта стратегии, выбираемые per-pair в `pairs.json`:

| Стратегия | Вход | Выход |
|-----------|------|-------|
| **Bollinger** (230, 2.1) | Цена выше BB₊ или ниже BB₋ + contango stage | Цена с обратной стороны канала |
| **Keltner** (EMA 150 + ATR 24, 3.9) | Цена выше KC₊ или ниже KC₋ + contango stage | Цена с обратной стороны канала |

**ContangoFilter** — ранжирование всех пар по спреду фьючерс/спот (через bid/ask из стакана):
- TOP-N → только LONG (stage 1)
- BOTTOM-N → только SHORT (stage 2)
- Коэффициент контанго: авто `10^decimals`, особые правила для VTBR (20→100 с 15.07.2024) и GMKN (100→10 с 04.04.2024)

**Дополнительно:**
- Iceberg-ордера
- Guard экспирации: вход 3–100 дней, выход <3 дня
- Торговое окно: 10:05–18:30 MSK, только будни

**Инструменты:** 9 фьючерсных пар (SBER, GAZP, ROSN, LKOH, VTBR, GMKN, ALRS, AFLT, MGNT)

---

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt  # если есть
# Или вручную:
pip install aiohttp pandas numpy streamlit plotly python-dotenv requests
```

### 2. Настройка

Скопируй и заполни конфиги:

```bash
# Для @magnets_bot
cp @magnets_bot/config.example.json @magnets_bot/config.json
# Отредактируй: api_secret, account_id, stocks

# Для @futures_bot — .env уже есть
# (ARENA_API_TOKEN и ARENA_ACCOUNT_ID в @futures_bot/.env)
```

**Важно:** В `.env` ключи должны называться `ARENA_API_TOKEN` и `ARENA_ACCOUNT_ID` (как в коде `main.py`).  
Сейчас там `FINAM_API_TOKEN` / `FINAM_ACCOUNT_ID` — при первом запуске поправь название переменных.

### 3. Запуск UI (рекомендуется)

```bash
# Оба сразу:
python run_uis.py

# Или по отдельности:
streamlit run @magnets_bot/app.py --server.port 8501
streamlit run @futures_bot/app.py --server.port 8502
```

- http://localhost:8501 — 🧲 Минковский (акции)
- http://localhost:8502 — 📈 Bollinger+Keltner (фьючерсы)

### 4. Запуск ботов (без UI)

```bash
python @magnets_bot/main.py
python @futures_bot/main.py
```

---

## Архитектура проекта

```
Minkowski-space/
│
├── @magnets_bot/                  # 🧲 Пространство Минковского (акции)
│   ├── app.py                     #   Streamlit UI (порт 8501)
│   ├── main.py                    #   Торговый движок (polling)
│   ├── strategy.py                #   MinkowskiClassifier + StrategyManager
│   ├── indicators.py              #   RSI, WT, CCI, ADX, Kernel Regression
│   ├── arena_client.py            #   Arena API + MOEX ISS (sync, requests, CSV-кэш)
│   ├── config.example.json        #   Пример конфига
│   └── .env.example               #   Пример переменных
│
├── @futures_bot/                  # 📈 Bollinger/Keltner + Contango (фьючерсы)
│   ├── app.py                     #   Streamlit UI (порт 8502)
│   ├── main.py                    #   Торговый движок (async polling)
│   ├── arena.py                   #   Arena API + MOEX ISS (async, aiohttp, Parquet-кэш)
│   ├── indicators.py              #   Bollinger, Keltner, ContangoCalculator, ContangoFilter
│   ├── pairs.json                 #   Список фьючерсных пар со стратегиями
│   ├── .env                       #   API-ключи
│   ├── cache/                     #   Parquet-кэш 1m свечей
│   ├── data/                      #   Дополнительные данные
│   └── scripts/                   #   Вспомогательные скрипты
│
├── run_uis.py                     # Запуск обоих Streamlit UI
├── README.md                      # Этот файл
├── .gitignore
└── ТЕОРЕТИЧЕСКАЯ БАЗА.md          # Теоретическое описание
```

---

## Сравнение стратегий

| | @magnets_bot | @futures_bot |
|---|---|---|
| **Инструменты** | 34 акции MOEX | 9 фьючерсных пар |
| **Таймфрейм** | 15m | 15m |
| **Данные** | MOEX ISS (CSV-кэш) | MOEX ISS (Parquet-кэш) |
| **API** | requests (sync) | aiohttp (async) |
| **Сигналы** | BUY / SELL / CLOSE | BUY / SELL |
| **Фильтр** | Kernel Regression + Volatility | Contango ranking (bid/ask) |
| **Выход** | Strict (4 bars) / Dynamic | Обратная сторона канала / экспирация |
| **Риск-менеджмент** | Hard Stop, Daily Limit, Drawdown | Hard Stop, Daily Limit, Drawdown |

---

## Параметры (defaults)

### @futures_bot (`main.py`, константы)

| Параметр | Значение | Описание |
|----------|----------|----------|
| `STRATEGY` | `"bollinger"` | Стратегия по умолчанию (переопределяется в `pairs.json`) |
| `REGIME` | `"On"` | On / Off / OnlyLong / OnlyShort |
| `BOLLINGER_LENGTH` | 230 | Период Bollinger |
| `BOLLINGER_DEVIATION` | 2.1 | Множитель std для Bollinger |
| `KELTNER_EMA_LENGTH` | 150 | Период EMA для Keltner |
| `KELTNER_ATR_LENGTH` | 24 | Период ATR для Keltner |
| `KELTNER_DEVIATION` | 3.9 | Множитель ATR для Keltner |
| `ICEBERG_COUNT` | 1 | Количество айсберг-ордеров |
| `VOLUME_VALUE` | 15.0 | % депозита на сделку |
| `MIN_EXPIRATION_DAYS` | 3 | Мин. дней до экспирации для входа |
| `MAX_EXPIRATION_DAYS` | 100 | Макс. дней до экспирации для входа |
| `CONTANGO_FILTER_COUNT` | 5 | Пар в每组 stage |
| `LOOP_SLEEP_SEC` | 60 | Пауза между циклами (сек) |

### @magnets_bot (`config.json`)

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `neighbors_count` | 8 | Число соседей kNN |
| `feature_count` | 4 | Количество признаков (⚠ в примере 4, но features[] содержит 5) |
| `use_kernel_filter` | true | Фильтр наклона ядра |
| `use_volatility_filter` | true | Фильтр волатильности |
| `use_dynamic_exits` | false | Динамические выходы (вместо strict) |
| `volume_value` | 14.0 | % депозита на сделку |
| `drawdown_stop_pct` | 0.3 | Остановка при просадке 30% |

---

## Формат тикеров

```
Акции:     TICKER@MISX       → SBER@MISX
Фьючерсы:  TICKER-MM.YY@FORTS → SBER-9.26@FORTS
```

---

## Риск-менеджмент

| Защита | @magnets_bot | @futures_bot |
|--------|-------------|-------------|
| Hard Stop Loss | -2% от капитала | -2% от капитала |
| Drawdown Reduce | -5% → объём /2 | -5% → объём /2 |
| Drawdown Stop | -30% → полная остановка | -10% → полная остановка |
| Daily Limit | 190 заявок/день | 190 заявок/день |

---

## API

| Источник | URL | Аутентификация |
|----------|-----|---------------|
| **Finam Arena** (торговля) | `https://arena.finam.ru/v1` | Bearer token |
| **Finam API** (стакан) | `https://arena.finam.ru/v1/instruments/{symbol}/orderbook` | Bearer token |
| **MOEX ISS** (данные) | `https://iss.moex.com/iss` | Не требуется |

---

## Советы

- **Логи** — stdout, видно в терминале
- **Сделки** — `trades.csv` в папке каждой стратегии
- **Состояние** — `state.json` для UI (обновляется каждый цикл)
- **Hot reload** — `@magnets_bot` перечитывает `config.json` каждый цикл; `@futures_bot` — `pairs.json`
- **Кэш** — Parquet (`@futures_bot`) / CSV (`@magnets_bot`) в `cache/`. Первый запуск загружает данные, следующие — только догрузка
- **Остановка** — `touch stop.flag` в папке стратегии, бот остановится после текущего цикла
