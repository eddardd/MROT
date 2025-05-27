import os
import pickle
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from mrot import MassRepulsiveOptimalTransport

parser = argparse.ArgumentParser(
    prog='Runs evaluation of MROT on ADBench')
parser.add_argument(
    '--mode',
    default=1,
    type=int,
    help='Which mode to use'
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

base_path = "./data/"

selected_datasets = [
    name for name in os.listdir(base_path) if 'tep' in name]
selected_dataset = selected_datasets[args.mode - 1]

k = args.k
reg_e = args.reg_e

with open(os.path.join(base_path, selected_dataset), 'rb') as f:
    data = pickle.loads(f.read())
X = np.concatenate([data['mean_windows'], data['std_windows']], axis=1)
y = data['labels']
# Binarize labels
y_binary = 1 * (y != 0)
indices = np.arange(X.shape[0])

for n_per_class in [5, 10, 15, 20, 25, 30]:
    indices_normal = np.where(y == 0)[0]
    indices_abnormal = []
    for i in range(1, 29):
        indices_abnormal.append(np.where(y == i)[0][:n_per_class])
    indices_abnormal = np.concatenate(indices_abnormal, axis=0)
    indices = np.concatenate([indices_normal, indices_abnormal], axis=0)
    for _ in range(5):
        ind_train, ind_test = train_test_split(
            indices, train_size=0.8, stratify=y_binary[indices])

        X_train, X_test = X[ind_train], X[ind_test]
        y_train, y_test = y_binary[ind_train], y_binary[ind_test]

        u, c = np.unique(y_train, return_counts=True)
        percent_anomalies = (c / c.sum())[1] * 100

        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        X_train = (X_train - mean) / (std + 1e-9)
        X_test = (X_test - mean) / (std + 1e-9)

        mrot = MassRepulsiveOptimalTransport(reg_e=reg_e, k=k)
        mrot.fit(X_train)
        y_pred = mrot.predict(X_test)

        score1 = roc_auc_score(y_true=y_test, y_score=y_pred)
        score2 = average_precision_score(y_true=y_test, y_score=y_pred)

        print(f"[{percent_anomalies},{reg_e},{k}]"
              f" AUC of MROT: {score1},{score2}")

        with open("./results/tep_mrot.csv", 'a') as f:
            f.write(f"tep,{args.mode},{percent_anomalies},"
                    f"{reg_e},{k},{score1},{score2}\n")
