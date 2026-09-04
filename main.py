"""
Raman spectra classification + GNNExplainer-based feature selection.

Tasks (see ``--task``):
    all    : per-class top-k important Raman shifts, averaged over all samples
    single : feature importance mask of one single sample (``--node_index``)
    class  : train the selected model and report confusion matrix / ROC
    svm / knn / rf / mlp / xgboost : classic ML baselines on the same data
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

# 无显示环境（服务器/批处理）时强制使用 Agg 后端
if '--no_show' in sys.argv:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402  (必须在 use() 之后导入)
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler

from config import args
from utils import norm_adj, adjacency_to_edge_index
from layer import GATNet, GCNNet, HybridGNN, TransformerGNN, CNNNet, FCNet

try:  # PyG < 2.4
    from torch_geometric.nn import GNNExplainer
except ImportError:  # PyG >= 2.4
    try:
        from torch_geometric.explain.algorithm import GNNExplainer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'GNNExplainer not found. Install torch-geometric>=2.0 '
            '(this repo targets the PyG 2.x GNNExplainer API).'
        ) from exc


# ------------------------------------------------------------------ #
# 通用工具
# ------------------------------------------------------------------ #
def get_output_dir():
    os.makedirs(args.output_dir, exist_ok=True)
    return args.output_dir


def save_fig(fig, filename):
    """Save a figure into --output_dir and (optionally) show it."""
    path = os.path.join(get_output_dir(), filename)
    fig.savefig(path, dpi=240, bbox_inches='tight')
    print(f'Figure saved to {path}')
    if not args.no_show:
        plt.show()
    plt.close(fig)
    return path


def detect_header(data_path):
    """A CSV has a header if its first field is not a number
    (or if its last column is literally named 'class')."""
    with open(data_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    fields = first_line.split(',')
    if not fields:
        return False

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    if fields[-1].strip().lower() == 'class':
        return True
    return not is_number(fields[0].strip())


def data_loader(data_path):
    """Read the CSV, split 75/25 and build one graph per split."""
    read_csv = pd.read_csv if args.header else lambda p: pd.read_csv(p, header=None)
    data = read_csv(data_path).values

    x = data[:, :args.d]                      # 特征矩阵
    y = data[:, -1]

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.25, random_state=1, stratify=y)

    edge_index_train = adjacency_to_edge_index(norm_adj(x_train, eud=args.eud, k=args.knn))
    edge_index_val = adjacency_to_edge_index(norm_adj(x_val, eud=args.eud, k=args.knn))

    x_train = torch.tensor(x_train, dtype=torch.float)
    x_val = torch.tensor(x_val, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    return x_train, y_train, edge_index_train, x_val, y_val, edge_index_val


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, reduction='mean', device=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.device = device

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        if self.device is not None:
            targets_one_hot = targets_one_hot.to(self.device)

        # 计算 Focal Loss
        focal_loss = -self.alpha * (1 - probs) ** self.gamma * log_probs * targets_one_hot
        focal_loss = focal_loss.sum(dim=-1)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def plot_class_distribution(preds, labels, num_classes):
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    class_probs = np.zeros(num_classes)
    for i in range(num_classes):
        class_probs[i] = np.mean(preds_np[labels_np == i, i])

    fig, ax = plt.subplots()
    ax.bar(range(num_classes), class_probs)
    ax.set_xlabel('Class')
    ax.set_ylabel('Average Predicted Probability')
    ax.set_title('Average Predicted Probability per Class')
    save_fig(fig, 'class_distribution.png')


# ------------------------------------------------------------------ #
# 训练
# ------------------------------------------------------------------ #
def train(model, data_path, device):
    x_train, y_train, edge_index_train, x_val, y_val, edge_index_val = data_loader(data_path)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)
    edge_index_train = edge_index_train.to(device)
    edge_index_val = edge_index_val.to(device)

    loss_func = FocalLoss(alpha=1, gamma=2.0, device=device).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.75, patience=60)

    model.to(device)
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        train_pred = model(x_train, edge_index_train)
        train_loss = loss_func(train_pred, y_train)
        train_loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val, edge_index_val)
            val_loss = loss_func(val_pred, y_val)
            scheduler.step(val_loss)
            train_acc = accuracy_score(y_train.cpu().numpy(),
                                       train_pred.argmax(dim=-1).cpu().numpy())
            val_acc = accuracy_score(y_val.cpu().numpy(),
                                     val_pred.argmax(dim=-1).cpu().numpy())

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1 or epoch == 0:
            print(f'Epoch {epoch + 1:>3} | Train Loss: {train_loss:.3f} '
                  f'| Train Acc: {train_acc * 100:>6.2f}% | Val Loss: '
                  f'{val_loss:.3f} | Val Acc: '
                  f'{val_acc * 100:.2f}% | L Rate:'
                  f'{scheduler.get_last_lr()[0]:.6f}')


def build_model():
    if args.model == 'GAT':
        return GATNet(in_dim=args.d, out_dim=args.c, hid_dim=args.hidden)
    if args.model == 'GCN':
        return GCNNet(in_dim=args.d, out_dim=args.c, hid_dim=args.hidden)
    if args.model == 'HybridGNN':
        return HybridGNN(in_dim=args.d, out_dim=args.c, hid_dim=args.hidden)
    if args.model == 'GTN':
        return TransformerGNN(in_dim=args.d, out_dim=args.c, hid_dim=args.hidden)
    if args.model == 'CNN':
        return CNNNet(in_dim=args.d, out_dim=args.c, hid_dim=args.hidden)
    if args.model == 'FC':
        return FCNet(in_dim=args.d, out_dim=args.c, hid_dim=args.hidden)
    raise ValueError(f'Unknown model: {args.model} '
                     f'(choose from GCN/GAT/HybridGNN/GTN/CNN/FC)')


# 通用前置步骤
def init(data_path):
    # 表头判定：auto -> 自动检测；true/false -> 强制
    if args.header == 'auto':
        args.header = detect_header(data_path)
    else:
        args.header = (args.header == 'true')
    print(('存在' if args.header else '不存在') + '表头')

    if args.header:
        # 将表头中除了 'class' 的数据保存为一个列表，作为特征名称
        with open(data_path, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
        feature_names = [name for name in header_line.split(',') if name != 'class']
        # 如果特征名称的长度比 args.d 小，则将 args.d 设置为特征名称的长度
        if len(feature_names) < args.d:
            args.d = len(feature_names)
            print(f'数据长度不足，args.d 已被更新为 {args.d}')
        data = pd.read_csv(data_path).values
    else:
        n_cols = len(pd.read_csv(data_path, header=None, nrows=1).columns)
        feature_names = None
        if n_cols - 1 < args.d:
            args.d = n_cols - 1
            print(f'数据长度不足，args.d 已被更新为 {args.d}')
        data = pd.read_csv(data_path, header=None).values

    # 获取数据的类别数
    args.c = len(np.unique(data[:, -1]))
    print(f'分类已更新为：{args.c}，分别是：{np.unique(data[:, -1])}')

    if args.c * 70 > args.hidden:
        args.hidden = args.c * 70
        print(f'分类数较多，args.hidden 已被更新为 {args.hidden}')

    model = build_model()
    node_idx = args.node_index

    x = torch.tensor(data[:, :args.d], dtype=torch.float)
    y = torch.tensor(data[:, -1], dtype=torch.long)
    # 计算边索引
    edge_index = adjacency_to_edge_index(norm_adj(x.numpy(), eud=args.eud, k=args.knn))

    if args.cuda:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model.to(device)
    x, y, edge_index = x.to(device), y.to(device), edge_index.to(device)

    print(f'training on {device}')
    train(model, data_path, device)

    return feature_names, data, model, node_idx, edge_index, x, y, device


# ------------------------------------------------------------------ #
# 单个节点的特征重要性图
# ------------------------------------------------------------------ #
def RamanExplainer_single(data_path):
    feature_names, data, model, node_idx, edge_index, x, y, device = init(data_path)

    explainer = GNNExplainer(model, epochs=100, return_type='log_prob')
    node_feat_mask, _ = explainer.explain_node(node_idx, x, edge_index)
    node_feat_mask = node_feat_mask.detach().cpu()
    important_features = torch.argsort(node_feat_mask, descending=True)[:args.top_k]

    if feature_names is not None:
        important_features_adjusted = [feature_names[i] for i in important_features.tolist()]
        features_class = feature_names
    else:
        important_features_adjusted = (important_features + args.begin).tolist()
        features_class = list(range(args.begin, args.d + args.begin))

    print(f'Top {args.top_k} most important features in node {node_idx}: '
          f'{important_features_adjusted}')

    stem = os.path.splitext(os.path.basename(data_path))[0]
    tag = f'{stem}_Single_{args.model}_node{node_idx}'

    fig, ax = plt.subplots(figsize=(21, 9))   # 更大的画布以匹配较大的光谱尺度
    ax.bar(range(len(node_feat_mask)), node_feat_mask.tolist(),
           color='blue', alpha=1, width=1, label='importance')
    ax.set_xticks(range(0, len(features_class), max(1, len(features_class) // 20)))
    ax.set_xticklabels([features_class[i]
                        for i in range(0, len(features_class),
                                       max(1, len(features_class) // 20))],
                       rotation=90)
    ax.set_xlabel('Feature Index (Raman shift)')
    ax.set_ylabel('Score of Importance')
    ax.set_title(f'Feature Importance in Node {node_idx}')
    ax.legend()
    save_fig(fig, f'{tag}.png')

    out_csv = os.path.join(get_output_dir(), f'{tag}.csv')
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write('feature,score\n')
        for feat, score in zip(important_features_adjusted,
                               node_feat_mask[important_features].tolist()):
            f.write(f'{feat},{score}\n')
    print(f'Top {args.top_k} features saved to {out_csv}')


# 兼容原始拼写
RanmanExplainer_single = RamanExplainer_single


# ------------------------------------------------------------------ #
# 所有节点的重要特征清单
# ------------------------------------------------------------------ #
def RamanExplainer_all(data_path):
    feature_names, data, model, node_idx, edge_index, x, y, device = init(data_path)

    explainer = GNNExplainer(model, epochs=100, return_type='log_prob')

    labels = data[:, -1]
    classes = np.unique(labels)

    important_scores = []
    important_features_adjusted = []

    # 遍历第 i 类节点
    for i in range(len(classes)):
        total_mask = []
        featurefound = []
        scorefound = []
        classfeature = []
        classscore = []
        node_ids = np.where(labels == classes[i])[0]

        for node_idx in node_ids:
            node_feat_mask, _ = explainer.explain_node(int(node_idx), x, edge_index)
            nfmlist = node_feat_mask.detach().cpu().tolist()
            node_feature, _ = find_peaks(nfmlist)
            for j in node_feature:
                if j not in featurefound:
                    featurefound.append(int(j))
            if not total_mask:
                total_mask = nfmlist
            else:
                total_mask = [a + b for a, b in zip(total_mask, nfmlist)]

        avg_mask = np.array(total_mask) / len(node_ids)
        for j in featurefound:
            scorefound.append(float(avg_mask[j]))

        for _ in range(min(args.top_k, len(featurefound))):
            best = int(np.argmax(scorefound))
            classscore.append(scorefound[best])
            classfeature.append(featurefound[best])
            featurefound.pop(best)
            scorefound.pop(best)
        important_scores.append(classscore)

        if feature_names is not None:
            important_features_adjusted.append([feature_names[j] for j in classfeature])
        else:
            important_features_adjusted.append([int(j) + args.begin for j in classfeature])

    for i in range(len(classes)):
        print(f'Top {args.top_k} most important features in class {classes[i]} nodes: '
              f'{important_features_adjusted[i]}')

    stem = os.path.splitext(os.path.basename(data_path))[0]
    tag = f'{stem}_All_{args.model}'

    # 保存 features 和 scores
    out_csv = os.path.join(get_output_dir(), f'{tag}.csv')
    with open(out_csv, 'w', encoding='utf-8') as f:
        for i in range(len(classes)):
            f.write(f'features_class_{classes[i]},scores_class_{classes[i]}\n')
            for j in range(len(important_scores[i])):
                f.write(f'{important_features_adjusted[i][j]},{important_scores[i][j]}\n')
            f.write('\n')
    print(f'Top {args.top_k} most important features saved to {out_csv}')

    # 画图
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
              'olive', 'cyan', 'magenta', 'lime', 'navy', 'teal', 'gold', 'coral',
              'indigo', 'salmon', 'khaki', 'lavender', 'turquoise', 'maroon', 'sienna']
    fig, ax = plt.subplots(figsize=(21, 9))

    plotted = [list(map(float, feats)) for feats in important_features_adjusted]
    for i in range(len(classes)):
        ax.bar(plotted[i], important_scores[i],
               color=colors[i % len(colors)], alpha=1, label=f'class {classes[i]}', width=3)

    # 多个类别共有的特征：用各自颜色再画一次，避免被后来者覆盖
    for i in range(len(classes)):
        for j in plotted[i]:
            for h in range(i + 1, len(classes)):
                if j in plotted[h]:
                    s_i = important_scores[i][plotted[i].index(j)]
                    s_h = important_scores[h][plotted[h].index(j)]
                    if s_i < s_h:
                        ax.bar(j, s_i, color=colors[i % len(colors)], alpha=1, width=3)
                    elif s_i == s_h:
                        ax.bar(j, s_i, color=colors[i % len(colors)], alpha=0.5, width=3)

    all_scores = [s for scores in important_scores for s in scores]
    if all_scores:
        maxscore, minscore = max(all_scores), min(all_scores)
        span = maxscore - minscore if maxscore > minscore else 1.0
        ax.set_ylim(max(minscore - 2 * span, 0), min(maxscore + span, 1))
    ax.set_xlabel('Feature Index (Raman shift)')
    ax.set_ylabel('Scores of Importance')
    ax.set_title('Feature Importance Across Nodes')
    ax.legend()
    save_fig(fig, f'{tag}.png')


# ------------------------------------------------------------------ #
# GNN 分类：混淆矩阵 + ROC
# ------------------------------------------------------------------ #
def train_multiclass(data_path):
    feature_names, data, model, node_idx, edge_index, x, y, device = init(data_path)

    _, _, _, x_val, y_val, edge_index_val = data_loader(data_path)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    edge_index_val = edge_index_val.to(device)

    # ========== 训练结束后输出混淆矩阵和 ROC 曲线 ==========
    model.eval()
    with torch.no_grad():
        val_pred = model(x_val, edge_index_val)
        y_true = y_val.cpu().numpy()
        y_pred = val_pred.argmax(dim=-1).cpu().numpy()
        y_score = torch.softmax(val_pred, dim=-1).cpu().numpy()

    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))
    print(f'Overall accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%')

    stem = os.path.splitext(os.path.basename(data_path))[0]
    tag = f'{stem}_class_{args.model}'

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.tight_layout()
    save_fig(fig, f'{tag}_confusion_matrix.png')

    # ROC 曲线（多分类 One-vs-Rest）
    fig, ax = plt.subplots()
    if args.c == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])   # 取正类得分
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.set_title('ROC Curve')
    else:
        y_true_bin = label_binarize(y_true, classes=np.arange(args.c))
        for i in range(args.c):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            ax.plot(fpr, tpr, lw=2,
                    label=f'Class {i} (AUC = {auc(fpr, tpr):.3f})')
        ax.set_title('ROC Curve (One-vs-Rest)')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    fig.tight_layout()
    save_fig(fig, f'{tag}_roc.png')

    save_path = os.path.join(get_output_dir(), f'{tag}.pth')
    torch.save(model.state_dict(), save_path)
    print(f'模型已保存至 {save_path}')


# ------------------------------------------------------------------ #
# 传统机器学习基线
# ------------------------------------------------------------------ #
def _make_ml_model(kind):
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    if kind == 'svm':
        return SVC(kernel='rbf', probability=True, random_state=args.seed)
    if kind == 'knn':
        return KNeighborsClassifier(n_neighbors=5)
    if kind == 'rf':
        return RandomForestClassifier(n_estimators=500, random_state=args.seed, n_jobs=-1)
    if kind == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(args.hidden,), max_iter=2000,
                             random_state=args.seed)
    if kind == 'xgboost':
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError('xgboost is not installed: pip install xgboost') from exc
        return XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                             random_state=args.seed, eval_metric='logloss',
                             tree_method='hist', n_jobs=-1)
    raise ValueError(f'Unknown ML model: {kind}')


def train_ml_model(data_path, kind='svm'):
    """Train a classic ML baseline on the raw spectra (no graph)."""
    if args.header == 'auto':
        args.header = detect_header(data_path)
    else:
        args.header = (args.header == 'true')

    read_csv = pd.read_csv if args.header else lambda p: pd.read_csv(p, header=None)
    data = read_csv(data_path).values
    x = data[:, :args.d].astype(float)
    y = data[:, -1]

    with open(data_path, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()
    feature_names = ([n for n in header_line.split(',') if n != 'class']
                     if args.header else None)
    if feature_names is not None and len(feature_names) < x.shape[1]:
        args.d = x.shape[1] = len(feature_names)
        x = data[:, :args.d].astype(float)

    classes = np.unique(y)
    args.c = len(classes)
    print(f'分类：{args.c} 类 -> {classes}')

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.25, random_state=1, stratify=y)

    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_val_s = scaler.transform(x_val)

    clf = _make_ml_model(kind)
    clf.fit(x_train_s, y_train)
    y_pred = clf.predict(x_val_s)
    y_score = clf.predict_proba(x_val_s) if hasattr(clf, 'predict_proba') else None

    print(f'[{kind}] Confusion Matrix:')
    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred, digits=4))
    print(f'[{kind}] Accuracy: {accuracy_score(y_val, y_pred) * 100:.2f}%')

    stem = os.path.splitext(os.path.basename(data_path))[0]
    tag = f'{stem}_{kind}'

    # 混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix - {kind}')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.tight_layout()
    save_fig(fig, f'{tag}_confusion_matrix.png')

    # ROC
    if y_score is not None:
        fig, ax = plt.subplots()
        if args.c == 2:
            fpr, tpr, _ = roc_curve(y_val, y_score[:, 1])
            ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {auc(fpr, tpr):.3f})')
        else:
            y_bin = label_binarize(y_val, classes=np.arange(args.c))
            for i in range(args.c):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                ax.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {auc(fpr, tpr):.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {kind}')
        ax.legend(loc='lower right')
        fig.tight_layout()
        save_fig(fig, f'{tag}_roc.png')

    # 特征重要性（RF / XGBoost）
    if hasattr(clf, 'feature_importances_'):
        imp = np.asarray(clf.feature_importances_)
        order = np.argsort(imp)[::-1][:args.top_k]
        names = ([feature_names[i] for i in order] if feature_names is not None
                 else [int(i) + args.begin for i in order])
        print(f'[{kind}] Top {args.top_k} features: {names}')
        out_csv = os.path.join(get_output_dir(), f'{tag}_features.csv')
        with open(out_csv, 'w', encoding='utf-8') as f:
            f.write('feature,importance\n')
            for idx, name in zip(order, names):
                f.write(f'{name},{imp[idx]}\n')
        print(f'Feature importances saved to {out_csv}')

    return clf


# ------------------------------------------------------------------ #
def load_model(model, weight_path, device):
    """Load trained weights into an already-built model."""
    if not os.path.exists(weight_path):
        print(f'没有找到权重文件：{weight_path}')
        return None
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model
