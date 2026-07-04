"""Запуск Streamlit UI для обеих стратегий (2 порта, 2 браузера)"""
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent

print("=" * 50)
print("ЗАПУСК UI ДЛЯ ОБЕИХ СТРАТЕГИЙ")
print("=" * 50)

p1 = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "@magnets_bot/app.py",
     "--server.port", "8501", "--server.headless", "true"],
    cwd=str(BASE),
)

p2 = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "@futures_bot/app.py",
     "--server.port", "8502", "--server.headless", "true"],
    cwd=str(BASE),
)

time.sleep(3)
print("\nОткрой в браузере:")
print("  📊 http://localhost:8501  — Минковский (акции)")
print("  📈 http://localhost:8502  — Bollinger+Keltner (фьючерсы)")
print("\nНажми Ctrl+C для остановки.\n")

try:
    p1.wait()
    p2.wait()
except KeyboardInterrupt:
    print("\n⛔ Остановка...")
    p1.terminate()
    p2.terminate()
    for p in (p1, p2):
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    print("✅ Остановлены")
