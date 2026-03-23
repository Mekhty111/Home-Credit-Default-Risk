```
home-credit-default-risk/
│
├── README.md                          # Описание проекта, выводы, как запустить
├── requirements.txt                   # Все зависимости
├── .gitignore                         # data/, models/, __pycache__/
│
├── data/
│   ├── raw/                           # Сюда кладёшь всё скачанное с Kaggle (в .gitignore)
│   │   ├── application_train.csv
│   │   ├── application_test.csv
│   │   ├── bureau.csv
│   │   ├── bureau_balance.csv
│   │   ├── previous_application.csv
│   │   ├── installments_payments.csv
│   │   ├── credit_card_balance.csv
│   │   └── POS_CASH_balance.csv
│   └── processed/                     # Сюда сохраняешь промежуточные файлы (в .gitignore)
│       ├── application_clean.parquet
│       ├── bureau_features.parquet
│       ├── prev_app_features.parquet
│       └── master_table.parquet
│
├── notebooks/
│   ├── 01_data_overview.ipynb         # Обзор всех таблиц, схема связей, dtypes, размеры
│   ├── 02_eda_main.ipynb              # EDA application_train: пропуски, распределения, таргет
│   ├── 03_feature_engineering.ipynb   # Join всех таблиц, агрегации, новые признаки
│   ├── 04_woe_iv_selection.ipynb      # WoE/IV биннинг, отбор признаков, VIF, Chi-Square
│   ├── 05_baseline_models.ipynb       # DummyClassifier → LogReg → сравнение
│   ├── 06_lgbm_model.ipynb            # LightGBM, подбор гиперпараметров, SHAP
│   ├── 07_scorecard.ipynb             # Перевод LogReg в банковскую скоринговую карту
│   ├── 08_validation.ipynb            # OOT валидация, KS/AUC/Gini/PSI, calibration
│   └── 09_final_summary.ipynb         # Итоговое сравнение всех моделей, выводы
│
├── src/                               # Переиспользуемый код (импортируется в ноутбуки)
│   ├── __init__.py
│   ├── preprocessing.py               # Функции очистки, обработки пропусков
│   ├── feature_engineering.py         # Агрегации по bureau, prev_app и т.д.
│   ├── woe_encoding.py                # Обёртка над optbinning для WoE трансформации
│   ├── metrics.py                     # KS, Gini, PSI, AUC — все в одном месте
│   ├── scorecard.py                   # Логика перевода коэффициентов LR в баллы
│   └── visualization.py              # Общие функции для графиков
│
├── models/                            # Сохранённые модели (в .gitignore)
│   ├── logreg_woe.pkl
│   ├── lgbm_model.pkl
│   └── scorecard_table.csv            # Сама скоринговая карта — это показываешь всем
│
└── app/                               # Продакшн-демо (критерий "реализация в продакшне")
    ├── app.py                         # Streamlit или FastAPI
    └── predict.py                     # Загрузка модели, предсказание
```