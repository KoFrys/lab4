"""Лабораторная №4: ансамбль для задачи кредитного скоринга (german.csv).

Выбранная модель - мягкое голосование (Soft Voting) трёх разнородных моделей:
1. логистическая регрессия (с one-hot кодированием категорий и сильной L2-регуляризацией);
2. Extra Trees;
3. Gradient Boosting.

Все преобразования и обучение выполняются только на train-части. test_target
используется лишь для вычисления итоговых метрик.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

# 20 признаков german.csv в порядке столбцов файла (без Creditability)
FEATURE_NAMES = [
    "Account_Balance",
    "Duration_of_Credit_monthly",
    "Payment_Status_of_Previous_Credit",
    "Purpose",
    "Credit_Amount",
    "Value_Savings_Stocks",
    "Length_of_current_employment",
    "Instalment_per_cent",
    "Sex_Marital_Status",
    "Guarantors",
    "Duration_in_Current_address",
    "Most_valuable_available_asset",
    "Age_years",
    "Concurrent_Credits",
    "Type_of_apartment",
    "No_of_Credits_at_this_Bank",
    "Occupation",
    "No_of_dependents",
    "Telephone",
    "Foreign_Worker",
]

# Категориальные признаки (коды без естественного порядка или с нелинейным эффектом).
# Для логистической регрессии они кодируются one-hot, деревьям достаточно исходных кодов.
NOMINAL_COLUMNS = [
    "Account_Balance",
    "Payment_Status_of_Previous_Credit",
    "Purpose",
    "Value_Savings_Stocks",
    "Length_of_current_employment",
    "Sex_Marital_Status",
    "Guarantors",
    "Most_valuable_available_asset",
    "Concurrent_Credits",
    "Type_of_apartment",
    "Occupation",
]

MODEL_NAME = "Soft Voting: LogisticRegression + ExtraTrees + GradientBoosting"


# --------------------------------------------------------------------------
# Подготовка данных
# --------------------------------------------------------------------------
def to_frame(features) -> pd.DataFrame:
    """Приводит признаки к DataFrame с именами столбцов german.csv.

    Если пришёл массив из 20 столбцов, ему присваиваются имена FEATURE_NAMES
    (порядок как в файле). Иначе столбцам даются общие имена f0, f1, ...
    """
    if isinstance(features, pd.DataFrame):
        frame = features.copy()
    else:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if matrix.shape[1] == len(FEATURE_NAMES):
            names = FEATURE_NAMES
        else:
            names = []
            for index in range(matrix.shape[1]):
                names.append("f" + str(index))
        frame = pd.DataFrame(matrix, columns=names)
    frame = frame.reset_index(drop=True)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признаки, связанные с суммой и сроком кредита.

    - log_amount: логарифм суммы кредита (сумма сильно скошена вправо);
    - amount_per_month: сумма, делённая на срок (размер ежемесячной нагрузки);
    - burden: нагрузка, умноженная на процент платежа от дохода.
    Если нужных столбцов нет (данные другого вида), возвращает frame без изменений.
    """
    needed = ["Credit_Amount", "Duration_of_Credit_monthly", "Instalment_per_cent"]
    for column in needed:
        if column not in frame.columns:
            return frame
    result = frame.copy()
    duration = result["Duration_of_Credit_monthly"].clip(lower=1)
    amount = result["Credit_Amount"].clip(lower=0)
    result["log_amount"] = np.log1p(amount)
    result["amount_per_month"] = amount / duration
    result["burden"] = result["Instalment_per_cent"] * amount / duration
    return result


def binary_labels(target) -> np.ndarray:
    """Целевая переменная в виде 0/1 (если метки другие, большая метка = 1)."""
    y = np.asarray(target, dtype=float).ravel()
    classes = np.unique(y)
    if len(classes) == 2 and not set(classes) <= {0.0, 1.0}:
        y = (y == classes[1]).astype(float)
    return y.astype(int)


def split_columns(columns) -> tuple[list[str], list[str]]:
    """Делит столбцы на числовые и категориальные (по списку NOMINAL_COLUMNS)."""
    numeric = []
    nominal = []
    for column in columns:
        if column in NOMINAL_COLUMNS:
            nominal.append(column)
        else:
            numeric.append(column)
    return numeric, nominal


# --------------------------------------------------------------------------
# Модели
# --------------------------------------------------------------------------
def make_logistic(columns, seed: int, c_value: float = 0.05, one_hot: bool = True) -> Pipeline:
    """Логистическая регрессия: стандартизация чисел и (по желанию) one-hot для категорий."""
    numeric, nominal = split_columns(columns)
    if one_hot:
        parts = [
            ("num", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), nominal),
        ]
    else:
        parts = [("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler())]), list(columns))]
    return Pipeline([
        ("prep", ColumnTransformer(parts)),
        ("model", LogisticRegression(C=c_value, max_iter=2000, random_state=seed)),
    ])


def make_tree_preprocessor(columns) -> ColumnTransformer:
    """Для деревьев масштабирование не нужно, достаточно заполнить пропуски."""
    return ColumnTransformer([("all", SimpleImputer(strategy="median"), list(columns))])


def make_extra_trees(columns, seed: int, n_estimators: int = 400, min_samples_leaf: int = 3) -> Pipeline:
    return Pipeline([
        ("prep", make_tree_preprocessor(columns)),
        ("model", ExtraTreesClassifier(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=seed)),
    ])


def make_gradient_boosting(columns, seed: int, n_estimators: int = 300,
                           max_depth: int = 3, learning_rate: float = 0.03) -> Pipeline:
    return Pipeline([
        ("prep", make_tree_preprocessor(columns)),
        ("model", GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=0.8, random_state=seed)),
    ])


def make_voting_ensemble(columns, seed: int) -> VotingClassifier:
    """Итоговая модель: мягкое голосование LR + ExtraTrees + GradientBoosting."""
    return VotingClassifier(
        estimators=[
            ("logistic", make_logistic(columns, seed)),
            ("extra_trees", make_extra_trees(columns, seed)),
            ("gradient_boosting", make_gradient_boosting(columns, seed)),
        ],
        voting="soft",
    )


def compute_metrics(y_true, probabilities, predictions) -> dict:
    """ROC-AUC (по вероятностям), accuracy, precision, recall и F1."""
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


# --------------------------------------------------------------------------
# Контрактная функция
# --------------------------------------------------------------------------
def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Модель обучается только на train. test_target используется только после
    обучения для вычисления метрик.
    """
    seed = int(data.random_seed)

    x_train = add_features(to_frame(data.train_features))
    x_test = add_features(to_frame(data.test_features))
    # столбцы test выстраиваются так же, как в train
    x_test = x_test.reindex(columns=list(x_train.columns))
    y_train = binary_labels(data.train_target)
    y_test = binary_labels(data.test_target)

    model = make_voting_ensemble(list(x_train.columns), seed)
    model.fit(x_train, y_train)

    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(x_test)[:, positive_index]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = compute_metrics(y_test, probabilities, predictions)

    return EnsembleResult(
        model_name=MODEL_NAME,
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=metrics["roc_auc"],
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
    )
