<h1 align="center">🤖 MOEX Trading Bots</h1>

> 🔥 **Две независимые алгоритмические стратегии для торговли на Московской бирже.** Классификация через пространство Минковского на акциях + Bollinger / Keltner с Contango-фильтром на фьючерсах. Finam Arena API (торговля) + MOEX ISS (данные). 
>
> ⚡ **Первый заход — 7 дней данных. Дальше — только догрузка свежих свечей.** Parquet-кеш 1m свечей делает каждый цикл в 10+ раз быстрее.
>
> 🏆 **Два робота в двух процессах — никаких конфликтов модулей.** Каждая стратегия живёт в своей изоляции.

---

## 📋 Что это

🎯 **Два робота для MOEX в одной папке.** Один торгует акции через ML-классификатор, второй — фьючерсы через пробой каналов с контанго-ранжированием. 

🖥️ **Нативные Streamlit UI.** Каждая стратегия — отдельный процесс на своём порту. Открываешь два окна в браузере и мониторишь both.

⚡ **Умный кеш 1m свечей.** Первый запуск грузит 7 дней данных (~5 сек на символ). Каждый следующий цикл — только свежие минуты (~0.1 сек). Никаких лишних запросов к MOEX ISS.

🔒 **Конфиг на лету.** Режимы `On / Off / OnlyLong / OnlyShort`, параметры полос и каналов, лимиты рисков — меняются без перезапуска.

---

## 🧠 Стратегии

### 🧲 1. Пространство Минковского — акции

Классификация свечных паттернов через метрику Минковского.

| Компонент | Что делает |
|---|---|
| **5 фичей** | RSI, WT (WaveTrend), CCI, ADX, цена от ядра |
| **Классификатор** | kNN в пространстве Минковского — взвешенная метрика на искажённом пространстве признаков |
| **Фильтр тренда** | Kernel Regression — сглаживает шум, определяет направление |
| **Торговля** | Сигналы BUY / SELL / CLOSE_LONG / CLOSE_SHORT |

### 📈 2. Bollinger / Keltner + Contango — фьючерсы

Две независимые стратегии, работающие параллельно через общий ContangoFilter.

| Стратегия | Вход | Выход |
|---|---|---|
| **BollingerContangoStrategy** | Пробой верхней/нижней полосы Bollinger (200, 1.5) + contango stage | Цена заходит обратно за полосу |
| **KeltnerContangoStrategy** | Пробой канала Keltner (EMA 140 + ATR 30, 2.6) + contango stage | Цена заходит обратно в канал |

**ContangoFilter** — ранжирует все 10 пар по спреду фьючерс/спот. Топ-5 → только LONG, низ-5 → только SHORT, середина — не торгуем.

**Формат тикеров:** новый MOEX формат `SBER-9.26@FORTS` (сентябрь 2026). Автоматическое определение экспирации — не входит в позицию за 3 дня до окончания контракта.

---

## 📊 Сравнение

| | Минковский (акции) | Bollinger+Keltner (фьючерсы) |
|---|---|---|
| Инструменты | 10+ акций MOEX | 10 фьючерсных пар |
| Таймфрейм | 15m | 15m |
| Сигналы | BUY / SELL / CLOSE | BUY / SELL |
| Фильтры | Kernel Regression | Contango ranking |
| Риск-менеджмент | Hard Stop, Daily Limit, Drawdown | Hard Stop, Daily Limit, Drawdown |
| Кеш | CSV | **Parquet** (в 10x быстрее) |

---

## 🚀 Быстрый старт

### 1. Установка

### 2. Настройка

В файле `strategy_*/config.json` укажи свои:

```json
{
  "api_secret": "твой_секрет_arena",
  "account_id": "номер_счёта"
}
```

### 3. Запуск UI (рекомендую)

```bash
# Два окна в браузере одной командой
python run_uis.py

# Или по отдельности
streamlit run @magnets_bot/app.py --server.port 8501   # акции
streamlit run @futures_bot/app.py --server.port 8502   # фьючерсы
```

Открывай:
- **http://localhost:8501** 🧠 Минковский (акции)
- **http://localhost:8502** 📈 Bollinger+Keltner (фьючерсы)

### 4. Запуск ботов (без UI, терминал)

```bash
# Каждый отдельно
python @magnets_bot/main.py
python @futures_bot/main.py
```

---

## 🏗 Архитектура

```
├── run_uis.py               # Запуск обоих Streamlit UI
│
├── @magnets_bot/            # 🧲 Пространство Минковского
│   ├── app.py               #    Streamlit UI (порт 8501)
│   ├── main.py              #    Торговый движок
│   ├── strategy.py          #    Минковский-классификатор
│   ├── arena.py             #    Arena API + MOEX ISS
│   ├── indicators.py        #    RSI, WT, CCI, ADX, Kernel
│   ├── config.example.json  #    Настройки
│   └── .env.example         #    Переменные окружения
│
└── @futures_bot/            # 📈 Bollinger + Keltner + Contango
    ├── app.py               #    Streamlit UI (порт 8502)
    ├── main.py              #    Торговый движок
    ├── arena.py             #    Arena API + MOEX ISS
    ├── indicators.py        #    Bollinger, Keltner, Contango
    ├── pairs.json           #    Список пар
    ├── .env                 #    Переменные окружения
    └── cache/               #    Parquet-кеш свечей
```


### Риск-менеджмент

| Защита | Параметр | Дефолт |
|---|---|---|
| 🛑 Hard Stop Loss | `hard_stop_loss_pct` | -2% от капитала |
| 📉 Drawdown Reduce | `drawdown_reduce_pct` | -5% → объём /2 |
| ⛔ Drawdown Stop | `drawdown_stop_pct` | -10% → полная остановка |
| 🔄 Daily Limit | `max_daily_orders` | 190 заявок в день |

---

## 🔌 API

| Источник | URL | Аутентификация |
|---|---|---|
| **Arena API** (торговля) | `https://arena.finam.ru/v1` | Bearer token (`api_secret`) |
| **MOEX ISS** (данные) | `https://iss.moex.com/iss` | Не требуется |

### Формат тикеров

```
Акции:     TICKER@MISX       → SBER@MISX
Фьючерсы:  TICKER-MM.YY@FORTS → SBER-9.26@FORTS
Старый:    SECID@FORTS        → SRU6@FORTS (тоже работает)
```

### Маппинг SECID → новый формат

MOEX ISS принимает старые SECID (SRU6), но конфиг пишется в новом формате (SBER-9.26). Маппинг живёт в `arena_client.py`:

| Тикер | SECID | SHORTNAME |
|---|---|---|
| SBER | SRU6 | SBRF-9.26 |
| GAZP | GZU6 | GAZR-9.26 |
| LKOH | LKU6 | LKOH-9.26 |
и т.д.

---

## 🧪 Советы

- **Логи** в каждом `bot.py` пишут в stdout — видно в терминале
- **Сделки** пишутся в `trades.csv` (CSV, можно открыть в Excel)
- **Состояние** в `state.json` — обновляется каждый цикл для UI
- **Hot reload** конфига — меняй параметры в `config.json` без перезапуска
- **Остановка** — создай файл `stop.flag` в папке стратегии, бот остановится после текущего цикла

---

