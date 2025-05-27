import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from mrot import MassRepulsiveOptimalTransport

parser = argparse.ArgumentParser(
    prog='Runs evaluation of MROT on ADBench')
parser.add_argument(
    '--dataset',
    default=1,
    type=int,
    help='Which dataset to evaluate'
)
parser.add_argument(
    '--reg_e',
    default=0.0,
    type=float,
    help='Entropic regularization'
)
parser.add_argument(
    '--k',
    default=5,
    type=int,
    help='Number of nearest neighbors'
)
args = parser.parse_args()

# NOTE: Here, you should clone the ADBench repository and set the
# variable `base_path` accordingly.
base_path = r"./data/ADBench/adbench/datasets/Classical"

datasets = {}
for d in os.listdir(base_path):
    key, value = d.split('_')
    datasets[int(key)] = d
dataset_name = datasets[args.dataset]
print("=" * 100)
print("|" + "{:^98}".format(f"Benchmarking MROT on {dataset_name}") + "|")
print("=" * 100)

data = np.load(os.path.join(base_path, dataset_name), allow_pickle=True)
X = data['X']
y = data['y']

k = args.k
reg_e = args.reg_e
indices = np.arange(X.shape[0])

for _ in range(5):
    ind_train, ind_test = train_test_split(indices, train_size=0.8, stratify=y)
    if len(ind_train) > 20000:
        ind_train, _ = train_test_split(
            ind_train, train_size=20000, stratify=y[ind_train])
    X_train, X_test = X[ind_train], X[ind_test]
    y_train, y_test = y[ind_train], y[ind_test]

    mrot = MassRepulsiveOptimalTransport(reg_e=reg_e,  k=k)
    mrot.fit(X_train)
    y_pred = mrot.predict(X_test)

    score1 = roc_auc_score(y_true=y_test, y_score=y_pred)
    score2 = average_precision_score(y_true=y_test, y_score=y_pred)

    print(f"[{reg_e},{k}] AUC of MROT: {score1}, {score2}")

    with open(f"./results/{dataset_name}_mrot.csv", 'a') as f:
        f.write(f"xgb,kde,{reg_e},{k},{score1},{score2}\n")
