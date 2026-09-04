import argparse
import torch
from utils import set_seed

parser = argparse.ArgumentParser(
    description='Graph neural networks for Raman spectra classification '
                'and GNNExplainer-based spectral feature selection.'
)

# ---------------- 数据 I/O ----------------
parser.add_argument('--data_path', type=str, default='data/example.csv',
                    help='Path to the input CSV (rows = samples, '
                         'first d columns = spectra, last column = class label).')
parser.add_argument('--output_dir', type=str, default='results',
                    help='Directory where figures / CSV / model weights are written.')
parser.add_argument('--header', type=str, default='auto',
                    choices=['auto', 'true', 'false'],
                    help="Whether the CSV has a header row. "
                         "'auto' detects it from the first line.")
parser.add_argument('--begin', type=int, default=500,
                    help='Raman shift (cm^-1) of the first feature column. '
                         'Only used to label axes when the CSV has no header.')

# ---------------- 训练超参 ----------------
parser.add_argument('--use-cuda', default=True,
                    help='Try to train on CUDA when available.')
parser.add_argument('--seed', type=int, default=11,
                    help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-6,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=300,
                    help='Dimension of hidden representations.')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='Dropout rate for GNN layers.')

# ---------------- 图构建 ----------------
parser.add_argument('--eud', action='store_true',
                    help='Build the sample graph with Euclidean distance. '
                         'Default (flag omitted) uses cosine distance.')
parser.add_argument('--knn', type=int, default=2,
                    help='k for the k-nearest-neighbour sample graph.')

# ---------------- 任务相关 ----------------
parser.add_argument('--c', type=int, default=2,
                    help='Num of classes (auto-detected from the data at runtime).')
parser.add_argument('--d', type=int, default=2000,
                    help='Num of spectral dimensions '
                         '(auto-truncated to the real column count at runtime).')
parser.add_argument('--model', type=str, default='GCN',
                    help='Model')  # GCN, GAT, HybridGNN, GTN, CNN, FC
parser.add_argument('--task', type=str, default='all',
                    help='Task')  # single, all, class, knn, svm, rf, mlp, xgboost
parser.add_argument('--node_index', type=int, default=0,
                    help='Node (sample) index explained in task: single')
parser.add_argument('--top_k', type=int, default=10,
                    help='Number of top features to be selected')
parser.add_argument('--no_show', action='store_true',
                    help='Save figures without calling plt.show() '
                         '(recommended on headless servers).')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
set_seed(args.seed, args.cuda)
