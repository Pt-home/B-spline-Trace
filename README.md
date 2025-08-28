
# 🧭 B‑spline Tracer — PNG → Контур → B‑spline → SVG/JSON

Застосунок на **Streamlit**, який:
- завантажує PNG/JPG (ч/б або кольорові),
- виділяє контур(и) через *marching squares*,
- підганяє **періодичний B‑spline** (ступінь 2–5, згладжування `s`),
- візуалізує «сирий» контур → спрощену ламану → гладку криву,
- експортує результати у **SVG/JSON/CSV**,
- має **строгу** Bézier‑екстракцію з B‑spline (вставка вузлів до кратності *k+1*) у **кубічні Безьє**.

## 🚀 Швидкий старт (локально)

1. **Встанови залежності**
   ```bash
   pip install -r requirements.txt
   ```

2. **Запусти застосунок**
   ```bash
   streamlit run bspline_trace_app.py
   ```

## 🧩 Можливості

- **Бінаризація**: Otsu / Adaptive / Manual (з інверсією), даунскел перед обробкою.
- **Морфологія**: closing/opening, заливання дірок, видалення дрібних об’єктів.
- **Контури**: знаходження й сортування за площею; опція «лише найбільший».
- **Спрощення ламаної**: параметр `tolerance`.
- **B‑spline**: ступінь 2–5 (`degree`), згладжування `s`, кількість вибірки.
- **Експорт**:
  - **B‑spline JSON (`tck`)** — вузли, коефіцієнти, ступінь.
  - **SVG (poly)** — шлях з відрізків крізь дискретизацію сплайну.
  - **SVG (Bezier)** — кубічні Безьє через Cardinal (Catmull‑Rom) з `tension`.
  - **Strict SVG/JSON** — **строга** Bézier‑екстракція (вузлова вставка → кубіки).
  - **CSV** — дискретизовані точки (debug/аналіз).
- **Координати**: система зображення (вісь `y` донизу), 1px = 1у.о.
- **Великі файли**: ліміт завантаження можна підвищити у `.streamlit/config.toml` 

## 📁 Структура репозиторію

```
.
├─ bspline_trace_app.py      # головний застосунок
├─ requirements.txt          # залежності
├─ .streamlit/
│  └─ config.toml            # конфіг (ліміти, тема тощо)
└─ README.md
```

## 🚀 Live demo (Streamlit)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://b-spline-tracegit-hrj9sjs5am7jvgxzcpvmjm.streamlit.app/)
