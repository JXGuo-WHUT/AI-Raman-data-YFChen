"""Entry point.

Usage
-----
    python run.py --data_path data/example.csv --task all --model GCN
    python run.py --data_path data/example.csv --task class --model GAT --epochs 300
    python run.py --data_path data/example.csv --task rf
"""
from config import args
from main import (RamanExplainer_single, RanmanExplainer_single,
                  RamanExplainer_all, train_multiclass, train_ml_model)


def main():
    data_path = args.data_path
    task = args.task

    if task == 'single':
        RamanExplainer_single(data_path)
    elif task == 'all':
        RamanExplainer_all(data_path)
    elif task == 'class':
        train_multiclass(data_path)
    elif task in ('svm', 'knn', 'rf', 'mlp', 'xgboost'):
        train_ml_model(data_path, task)
    else:
        raise SystemExit(
            f'不支持的任务类型: {task}，可选任务：single/all/class/svm/knn/rf/mlp/xgboost'
        )


if __name__ == '__main__':
    main()
