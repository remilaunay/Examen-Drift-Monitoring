"""Train and monitor a bike rental model with Evidently."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.core.datasets import DataDefinition, Dataset, Regression
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset, RegressionPreset
from evidently.tests import lt
from evidently.ui.workspace import Workspace
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_CACHE = DATA_DIR / "bike_sharing.csv"
WORKSPACE_DIR = PROJECT_ROOT / "bike_sharing_monitoring"
HTML_REPORT = PROJECT_ROOT / "diagnostic_report.html"

PROJECT_NAME = "Bike Rental Monitoring"
RANDOM_STATE = 42
TARGET_COLUMN = "cnt"
PREDICTION_COLUMN = "prediction"
DATE_COLUMN = "dteday"
VALIDATION_RMSE_LIMIT = 50.0

CATEGORICAL_FEATURES = [
    "season",
    "yr",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]
NUMERICAL_FEATURES = ["temp", "atemp", "hum", "windspeed"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
WEATHER_FEATURES = ["weathersit", *NUMERICAL_FEATURES]
MONITORING_COLUMNS = [
    *CATEGORICAL_FEATURES,
    *NUMERICAL_FEATURES,
    TARGET_COLUMN,
    PREDICTION_COLUMN,
]


def load_bike_sharing_data() -> pd.DataFrame:
    """Load the UCI dataset once, then reuse a small local CSV cache."""
    if DATA_CACHE.exists():
        dataframe = pd.read_csv(DATA_CACHE, parse_dates=[DATE_COLUMN])
        print(f"Dataset chargé depuis le cache : {DATA_CACHE}")
        return dataframe

    print("Téléchargement du dataset UCI Bike Sharing (id=275)...")
    bike_sharing = fetch_ucirepo(id=275)
    features = bike_sharing.data.features.copy()
    targets = bike_sharing.data.targets.copy()

    target_name = TARGET_COLUMN if TARGET_COLUMN in targets.columns else None
    if target_name is None:
        target_candidates = [
            column
            for column in targets
            if column.lower() in {"count", "rental_count"}
        ]
        if target_candidates:
            target_name = target_candidates[0]
        elif len(targets.columns) == 1:
            target_name = targets.columns[0]
        else:
            raise ValueError(
                f"Cible de location introuvable dans {list(targets.columns)}"
            )

    date_name = DATE_COLUMN if DATE_COLUMN in features.columns else None
    if date_name is None:
        date_candidates = [
            column
            for column in features
            if column.lower() in {"date", "datetime"}
        ]
        if date_candidates:
            features = features.rename(columns={date_candidates[0]: DATE_COLUMN})
        else:
            raise ValueError(
                f"Colonne de date introuvable dans {list(features.columns)}"
            )

    dataframe = features.copy()
    dataframe[TARGET_COLUMN] = targets[target_name].to_numpy()
    dataframe[DATE_COLUMN] = pd.to_datetime(dataframe[DATE_COLUMN], errors="coerce")
    dataframe = dataframe.dropna(subset=[DATE_COLUMN, TARGET_COLUMN])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(DATA_CACHE, index=False)
    print(f"Cache local enregistré : {DATA_CACHE}")
    return dataframe


def prepare_january_and_february(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select January as reference and February as simulated production."""
    required_columns = {DATE_COLUMN, TARGET_COLUMN, *MODEL_FEATURES}
    missing = sorted(required_columns.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Colonnes attendues absentes du dataset UCI : {missing}")

    dates = pd.to_datetime(dataframe[DATE_COLUMN], errors="coerce")
    in_2011 = dates.dt.year == 2011
    january = dataframe.loc[in_2011 & (dates.dt.month == 1)].copy()
    february = dataframe.loc[in_2011 & (dates.dt.month == 2)].copy()

    if january.empty or february.empty:
        raise ValueError(
            "Le dataset doit contenir des lignes pour janvier et février 2011 "
            f"(janvier={len(january)}, février={len(february)})."
        )

    january[DATE_COLUMN] = pd.to_datetime(january[DATE_COLUMN])
    february[DATE_COLUMN] = pd.to_datetime(february[DATE_COLUMN])
    return january, february


def build_model() -> Pipeline:
    """Build a random forest with explicit handling of coded categories."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_evidently_dataset(dataframe: pd.DataFrame) -> Dataset:
    """Map bike data columns to Evidently's regression dataset schema."""
    definition = DataDefinition(
        numerical_columns=[*NUMERICAL_FEATURES, TARGET_COLUMN, PREDICTION_COLUMN],
        categorical_columns=CATEGORICAL_FEATURES,
        regression=[Regression(target=TARGET_COLUMN, prediction=PREDICTION_COLUMN)],
    )
    return Dataset.from_pandas(
        dataframe[MONITORING_COLUMNS].copy(),
        data_definition=definition,
    )


def predict(dataframe: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Return monitoring columns with model predictions attached."""
    result = dataframe.copy()
    result[PREDICTION_COLUMN] = model.predict(result[MODEL_FEATURES])
    return result[MONITORING_COLUMNS].copy()


def split_february_into_windows(
    production: pd.DataFrame,
) -> list[tuple[str, list[int]]]:
    """Split February into three contiguous windows with similar day counts."""
    days = sorted(production[DATE_COLUMN].dt.day.unique())
    day_groups = np.array_split(days, 3)
    return [
        (f"W{index}", [int(day) for day in group])
        for index, group in enumerate(day_groups, start=1)
    ]


def create_workspace() -> tuple[Workspace, Any]:
    """Open or create the persistent local workspace and exam project."""
    workspace = Workspace.create(str(WORKSPACE_DIR))
    projects = workspace.search_project(PROJECT_NAME)
    project = projects[0] if projects else workspace.create_project(PROJECT_NAME)
    return workspace, project


def save_snapshot(
    workspace: Workspace,
    project_id: str,
    report: Report,
    reference: Dataset,
    current: Dataset,
):
    """Run a report and persist its snapshot in the monitoring project."""
    with warnings.catch_warnings():
        # Evidently 0.7.2 emits these library-internal warnings for sparse bins.
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"evidently\.legacy\.metrics\.regression_performance\.utils",
        )
        warnings.filterwarnings(
            "ignore",
            category=UndefinedMetricWarning,
            module=r"sklearn\.metrics\._regression",
        )
        snapshot = report.run(reference_data=reference, current_data=current)
    workspace.add_run(project_id, snapshot)
    return snapshot


def main() -> None:
    dataframe = load_bike_sharing_data()
    reference, production = prepare_january_and_february(dataframe)
    print(f"Référence janvier 2011 : {len(reference)} lignes")
    print(f"Production simulée février 2011 : {len(production)} lignes")

    train_data, test_data = train_test_split(
        reference,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )
    model = build_model()
    model.fit(train_data[MODEL_FEATURES], train_data[TARGET_COLUMN])

    train_monitoring = predict(train_data, model)
    test_monitoring = predict(test_data, model)
    validation_rmse = float(
        np.sqrt(
            mean_squared_error(
                test_monitoring[TARGET_COLUMN],
                test_monitoring[PREDICTION_COLUMN],
            )
        )
    )
    validation_status = "PASS" if validation_rmse < VALIDATION_RMSE_LIMIT else "FAIL"
    print(
        f"Validation janvier : RMSE={validation_rmse:.2f} "
        f"(seuil < {VALIDATION_RMSE_LIMIT:.0f}, test {validation_status})"
    )

    workspace, project = create_workspace()
    validation_report = Report(
        metrics=[
            RegressionPreset(
                rmse_tests=[lt(VALIDATION_RMSE_LIMIT)],
                include_tests=True,
            )
        ],
        metadata={"phase": "validation", "period": "janvier 2011"},
        tags=["validation", "janvier-2011"],
        include_tests=True,
    )
    save_snapshot(
        workspace,
        project.id,
        validation_report,
        make_evidently_dataset(train_monitoring),
        make_evidently_dataset(test_monitoring),
    )

    reference_monitoring = predict(reference, model)
    reference_dataset = make_evidently_dataset(reference_monitoring)
    production_results: list[dict[str, Any]] = []

    for week_name, days in split_february_into_windows(production):
        start_day, end_day = days[0], days[-1]
        week_data = production.loc[production[DATE_COLUMN].dt.day.isin(days)]
        if week_data.empty:
            raise ValueError(f"La période {week_name} ne contient aucune donnée.")

        current_monitoring = predict(week_data, model)
        current_dataset = make_evidently_dataset(current_monitoring)
        weekly_rmse = float(
            np.sqrt(
                mean_squared_error(
                    current_monitoring[TARGET_COLUMN],
                    current_monitoring[PREDICTION_COLUMN],
                )
            )
        )
        weekly_mae = float(
            mean_absolute_error(
                current_monitoring[TARGET_COLUMN],
                current_monitoring[PREDICTION_COLUMN],
            )
        )
        production_results.append(
            {
                "name": week_name,
                "start_day": start_day,
                "end_day": end_day,
                "data": current_monitoring,
                "dataset": current_dataset,
                "rmse": weekly_rmse,
                "mae": weekly_mae,
            }
        )

        weekly_report = Report(
            metrics=[
                RegressionPreset(
                    rmse_tests=[lt(VALIDATION_RMSE_LIMIT)],
                    include_tests=True,
                )
            ],
            metadata={"phase": "production", "period": week_name},
            tags=["production", week_name],
            include_tests=True,
        )
        save_snapshot(
            workspace,
            project.id,
            weekly_report,
            reference_dataset,
            current_dataset,
        )
        print(
            f"{week_name} (jours {start_day}–{end_day}, {len(week_data)} lignes) : "
            f"RMSE={weekly_rmse:.2f}, MAE={weekly_mae:.2f}"
        )

    worst_week = max(production_results, key=lambda result: result["rmse"])
    worst_dataset = worst_week["dataset"]
    diagnostic_report = Report(
        metrics=[
            ValueDrift(column=TARGET_COLUMN),
            DataDriftPreset(columns=WEATHER_FEATURES),
        ],
        metadata={
            "phase": "diagnostic final",
            "period": worst_week["name"],
            "rmse": f"{worst_week['rmse']:.2f}",
        },
        tags=["diagnostic", worst_week["name"]],
        include_tests=True,
    )
    diagnostic_snapshot = save_snapshot(
        workspace,
        project.id,
        diagnostic_report,
        reference_dataset,
        worst_dataset,
    )
    diagnostic_snapshot.save_html(str(HTML_REPORT))

    print(
        f"Période la plus dégradée : {worst_week['name']} "
        f"(RMSE={worst_week['rmse']:.2f}, MAE={worst_week['mae']:.2f})."
    )
    print(f"Rapport de dérive exporté : {HTML_REPORT}")
    print(f"Workspace Evidently : {WORKSPACE_DIR}")


if __name__ == "__main__":
    main()
