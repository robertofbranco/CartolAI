import re
from pathlib import Path

from cartola_backtest import rodar_backtest


TUNES_SETS = [

]


# best:        300, 8, 5, 50, 0.7
# more stable: 300, 8, 20, 2, 0.7
# peak points: 300, 8, 10, 2, sqrt

# 2 and 10 min sample split doesnt seem to make any diff
# sqrt is less stable -> get higher points when goes well, and less points when goes bad
# sqrt with higher split -> maker it even less stable

# to check: use sqrt when games are more predictable?

N_ESTIMATORS = [300, 500]
MAX_DEPTH = [8, 16]
MIN_SAMPLES_LEAF = [5, 10, 20]
MIN_SAMPLES_SPLIT = [10, 50]
MAX_FEATURES = ["sqrt", 0.7, 1]
RODADA_INICIO = 9
RODADA_FIM = 15
OUTPUT_ROOT = Path("results") / "tuning_backtest"
TUNING_FOLDER_KEYS = [
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "min_samples_split",
    "max_features",
    "random_state",
    "n_jobs",
]

def create_tuning_dict(n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features):
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "random_state": 42,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "n_jobs": -1,
    }


def tuning_output_folder(tuning: dict) -> Path:
    ordered_keys = [
        key
        for key in TUNING_FOLDER_KEYS
        if key in tuning
    ] + sorted(key for key in tuning if key not in TUNING_FOLDER_KEYS)
    parts = [f"{key}_{tuning[key]}" for key in ordered_keys]
    folder_name = "__".join(parts)
    folder_name = re.sub(r"[^A-Za-z0-9_.=-]+", "_", folder_name)
    return OUTPUT_ROOT / folder_name


def main():
    for n_estimators in N_ESTIMATORS:
        for max_depth in MAX_DEPTH:
            for min_samples_leaf in MIN_SAMPLES_LEAF:
                for min_samples_split in MIN_SAMPLES_SPLIT:
                    for max_features in MAX_FEATURES:
                        tuning_dict = create_tuning_dict(
                            n_estimators,
                            max_depth,
                            min_samples_leaf,
                            min_samples_split,
                            max_features
                        )

                        rodar_backtest(
                            RODADA_INICIO,
                            RODADA_FIM,
                            tuning=tuning_dict,
                            output_folder=tuning_output_folder(tuning_dict),
                        )


if __name__ == "__main__":
    main()
