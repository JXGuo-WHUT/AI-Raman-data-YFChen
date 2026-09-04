# RamanGNN-Explainer

用图神经网络（GCN / GAT / HybridGNN / TransformerConv）对拉曼光谱做分类，并用
**GNNExplainer** 反推"哪些拉曼位移（Raman shift）对判别最重要"，从而完成光谱特征选择。
同时提供 SVM / KNN / RandomForest / MLP / XGBoost 等传统基线做对照。

---

## 目录结构

```
.
├── baseline.py          # 基线校正（调用 pybaselines 的 airPLS，BSD-3）
├── config.py            # 全部超参与路径（argparse）
├── data_preprocess.py   # 按类别文件夹合并原始光谱 -> 单张 CSV
├── layer.py             # 模型定义（GCNNet / GATNet / HybridGNN / TransformerGNN / CNNNet / FCNet）
├── main.py              # 训练、评估、解释、可视化
├── make_demo_data.py    # 生成合成示例数据（无需真实光谱即可跑通）
├── run.py               # 入口
├── utils.py             # 构图（kNN + 归一化邻接）、边索引、随机种子
└── requirements.txt
```

## 安装

```bash
git clone https://github.com/<your-name>/raman-gnn-explainer.git
cd raman-gnn-explainer
pip install -r requirements.txt
```

`torch` / `torch-geometric` 请按[官方指引](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
安装与本机 CUDA 匹配的版本。仓库使用 PyG 2.x 的 `GNNExplainer` API。

## 快速开始（合成数据）

```bash
python make_demo_data.py --n_per_class 40 --n_features 500   # 生成 data/example.csv

python run.py --data_path data/example.csv --task all   --model GCN --epochs 200
python run.py --data_path data/example.csv --task single --model GCN --node_index 3
python run.py --data_path data/example.csv --task class  --model GAT
python run.py --data_path data/example.csv --task rf
```

## 数据格式

`--data_path` 指向一张 CSV：

| 402.729 | 404.329 | … | 2498.xxx | class |
|---|---|---|---|---|
| 1234.5  | 1180.2  | … | 990.1    | 0     |
| …       |         |   |          | 1     |

- **行 = 样本**，**列 = 拉曼位移通道**，**最后一列 = 类别标签**。
- 表头可选：`--header auto`（默认）会自动判断；也可以强制 `--header true/false`。
- 无表头时用 `--begin 402.729` 指定首列位移、 `--step`(画图用) 只影响坐标标注。

### 从原始光谱文件构建数据集

每个类别放一个文件夹，每个 `.csv` 是一张谱（两列：位移 / 强度）：

```bash
python data_preprocess.py \
    --input_dirs raw/class_0 raw/class_1 raw/class_2 \
    --output data/example.csv \
    --start_shift 402.729
```

脚本会 airPLS 去基线 + 最大值归一化（缩放到 10000），并在行尾追加类别序号。

基线方法可用 `--baseline` 切换：`airpls`（默认，需 pybaselines）、`als`、
`detrend`（仅用 scipy）、`none`（跳过）。

## 任务一览（`--task`）

| 任务 | 说明 |
|---|---|
| `all` | 对**每一类**汇总所有样本的 GNNExplainer 特征掩码，取 Top-K 峰位，输出 CSV + 对比柱状图 |
| `single` | 单个样本（`--node_index`）的特征重要性谱图 |
| `class` | 训练 GNN，输出混淆矩阵、ROC/AUC，并保存权重 `.pth` |
| `svm` / `knn` / `rf` / `mlp` / `xgboost` | 传统机器学习基线（同一 75/25 分层切分 + 标准化） |

所有图片与 CSV 统一写入 `--output_dir`（默认 `results/`），**不会污染数据目录**。
服务器端建议加 `--no_show` 只保存不弹窗。

## 主要参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data_path` | `data/example.csv` | 输入 CSV |
| `--output_dir` | `results` | 结果输出目录 |
| `--model` | `GCN` | `GCN` / `GAT` / `HybridGNN` / `GTN` / `CNN` / `FC` |
| `--task` | `all` | 见上表 |
| `--epochs` / `--lr` / `--wd` | 200 / 1e-3 / 1e-6 | 训练超参 |
| `--hidden` / `--dropout` | 300 / 0.4 | 隐层维度 / Dropout |
| `--eud` | 关闭 | 用欧氏距离构图（默认余弦距离） |
| `--knn` | 2 | 构图时每个样本保留的近邻数 |
| `--top_k` | 10 | 每类挑选的重要峰位个数 |
| `--node_index` | 0 | `single` 任务解释的样本下标 |
| `--seed` | 11 | 随机种子 |

## 方法简述

1. 每张谱作为一个**节点**，节点特征即该谱的强度向量；
2. 用余弦（或欧氏）距离的 kNN 建图，对称归一化得到邻接矩阵，转成 `edge_index`；
3. GNN 在"样本图"上做半监督式分类，损失函数为 Focal Loss（`gamma=2`，缓解类别不平衡）；
4. 训练完成后用 `GNNExplainer` 求每个节点的**特征掩码**，对同类样本取平均，
   再用 `scipy.signal.find_peaks` 在掩码谱上找峰，即为候选特征峰位。

## 常见问题

**Q：批量预处理时刷屏警告**
`pybaselines` 在未收敛或 `tol` 过小时会给出 `ParameterWarning`，一般无害。
加 `--quiet` 屏蔽，或调大 `--itermax` / `--lambda_`。

**Q：不想装 pybaselines**
用 `--baseline detrend`（仅依赖 scipy）或 `--baseline none`（跳过去基线）。

**Q：airPLS 参数怎么对应原始论文版本？**
`--lambda_` → `lam`，`--porder` → `diff_order`，`--itermax` → `max_iter`。

**Q：显存/内存不够**
`--task all` 会对**每个样本**跑一次 GNNExplainer，样本多时很慢。可先用
`--task single --node_index 0` 试跑，或简化为 `--knn 2 --hidden 140` 的小模型。

**Q：构图用什么距离？**
默认余弦距离（对光谱强度缩放不敏感）。加 `--eud` 切换为欧氏距离。

**Q：`torch_geometric` 导入 `GNNExplainer` 失败**
本仓库用的是 PyG 2.x 的旧版 `GNNExplainer(模型, epochs=..., return_type=...)`
接口，请安装 `torch-geometric>=2.0,<2.4`。

## 许可证

**整个仓库为 MIT 许可**，详见 [`LICENSE`](LICENSE)。仓库内不含任何 copyleft
（GPL/LGPL/AGPL）代码 —— airPLS 基线校正是通过依赖 `pybaselines`
（BSD-3-Clause, Copyright (c) 2021 Donald Erb）实现的，而非在仓库内附带实现。

第三方依赖许可一览（均为宽松许可证）：

| 依赖 | 许可证 |
|---|---|
| numpy / scipy / pandas / scikit-learn / matplotlib | BSD-3 |
| pybaselines | BSD-3-Clause |
| torch | BSD-3-Clause |
| torch-geometric | MIT |
| xgboost（可选） | Apache-2.0 |

## 引用

若本仓库对你的研究有帮助，请引用 airPLS 原始论文：

> Z.-M. Zhang, S. Chen, and Y.-Z. Liang, "Baseline correction using adaptive
> iteratively reweighted penalized least squares", *Analyst* **135**(5), 1138–1146 (2010).

---

### ⚠️ 提交代码前请自查

- 不要提交原始光谱数据（`data/`、`*.csv` 已在 `.gitignore` 中）。
- 不要提交绝对路径、Windows 用户名、课题内部编号 —— 统一用 `--data_path` 传参。
- 输出图片、`results/`、模型权重均不入库。
