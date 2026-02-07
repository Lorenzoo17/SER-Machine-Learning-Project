# src/splitting.py

from typing import List, Dict, Iterator, Tuple
import random

from sklearn.model_selection import GroupKFold, StratifiedKFold, StratifiedShuffleSplit


def make_cv_splits(
    all_files: List[str],
    labels: List[int],
    actors: List[str],
    n_folds: int,
    speaker_independent: bool,
    seed: int = 42,
) -> Iterator[Tuple[int, List[int], List[int]]]:
    """
    Ritorna un iterable di (fold_idx, train_val_idx, test_idx).

    - speaker_independent=True:
        GroupKFold su actors (come prima): produce n_folds split.

    - speaker_independent=False:
        Split fisso 70/15/15 stratificato su labels:
        restituisce UN SOLO "fold" (fold_idx=0) con:
          train_val_idx = train + val (85%)
          test_idx      = test (15%)

        In questo modo il tuo notebook può rimanere identico (for ... in make_cv_splits()).
    """
    if speaker_independent:
        gkf = GroupKFold(n_splits=n_folds)
        for fold_idx, (train_val_idx, test_idx) in enumerate(
            gkf.split(all_files, labels, groups=actors)
        ):
            yield fold_idx, list(train_val_idx), list(test_idx)
        return

    # speaker-dependent: split fisso 85/15 (train_val / test), stratificato per label
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    idx = list(range(len(all_files)))
    train_val_idx, test_idx = next(sss.split(idx, labels))

    # unico fold
    yield 0, list(train_val_idx), list(test_idx)


def split_train_val_within_fold(
    train_val_files: List[str],
    train_val_labels: List[int],
    train_val_actors: List[str],
    speaker_independent: bool,
    fold_idx: int,
    seed: int = 42,
) -> Dict[str, List]:
    """
    Ritorna dict con:
      train_files, val_files, train_actors, val_actors

    - speaker_independent=True:
        identica alla tua (validation per attori, bilanciata M/F)

    - speaker_independent=False:
        split train/val stratificato per label in modo da ottenere:
          train = 70%
          val   = 15%
        sapendo che train_val è già 85% del totale.
        Quindi: val_size = 15/85 = 0.176470...
    """
    if speaker_independent:
        unique_train_val_actors = list(set(train_val_actors))
        val_size = max(2, len(unique_train_val_actors) // 6)  # ~15-17%

        male_actors_tv = [a for a in unique_train_val_actors if int(a) % 2 == 1]
        female_actors_tv = [a for a in unique_train_val_actors if int(a) % 2 == 0]

        random.seed(seed + fold_idx)

        val_actors = (
            random.sample(male_actors_tv, min(val_size // 2, len(male_actors_tv)))
            + random.sample(female_actors_tv, min(val_size // 2, len(female_actors_tv)))
        )
        train_actors = [a for a in unique_train_val_actors if a not in val_actors]

        train_files = [f for f, a in zip(train_val_files, train_val_actors) if a in train_actors]
        val_files = [f for f, a in zip(train_val_files, train_val_actors) if a in val_actors]

        return {
            "train_files": train_files,
            "val_files": val_files,
            "train_actors": sorted(train_actors),
            "val_actors": sorted(val_actors),
        }

    # speaker-dependent: split train/val stratificato per label
    # per ottenere 70/15/15 globali: val = 15/85 del train_val
    val_ratio_within_trainval = 0.15 / 0.85 
    
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio_within_trainval,
        random_state=seed + fold_idx,
    )
    idx = list(range(len(train_val_files)))
    tr_idx, va_idx = next(sss.split(idx, train_val_labels))

    train_files = [train_val_files[i] for i in tr_idx]
    val_files = [train_val_files[i] for i in va_idx]

    return {
        "train_files": train_files,
        "val_files": val_files,
        "train_actors": sorted(set([train_val_actors[i] for i in tr_idx])),
        "val_actors": sorted(set([train_val_actors[i] for i in va_idx])),
    }
