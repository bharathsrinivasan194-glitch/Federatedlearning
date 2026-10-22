"""
Federated Learning Central Server with Aggregation, DNA Encryption & Clinical Evaluation
Reference: "A Federated Learning-based CNN Architecture of Privacy-Preserving Osteoporosis Detection on Knee X-Ray Images" (ICCCES-2026, IEEE).
"""

import numpy as np
import torch
import torch.nn as nn
import time
from collections import OrderedDict
from DNA_crypto import encrypt_weights, decrypt_weights

class FLServer:
    def __init__(self, global_model, password, rule=2, x0=0.42):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = global_model.to(self.device)
        self.password = password
        self.rule = rule
        self.x0 = x0
        self.r = 3.99

        self.updates = {}
        self.sample_counts = {}
        self.round_num = 0
        self.accuracies = []
        self.losses = []
        self.recalls = []
        self.precisions = []
        self.f1_scores = []
        self.aucs = []
        self.agg_times = []
        self.criterion = nn.CrossEntropyLoss()

    def receive_update(self, client_id, encrypted_weights, client_password, num_samples):
        weights = decrypt_weights(encrypted_weights, client_password)
        self.updates[client_id] = weights
        self.sample_counts[client_id] = num_samples

    def aggregate(self):
        self.round_num += 1
        start = time.time()

        if not self.updates:
            return None

        total_n = sum(self.sample_counts.values())
        first = list(self.updates.values())[0]
        averaged = OrderedDict()

        for name in first.keys():
            weighted_sum = None
            for cid, weights in self.updates.items():
                n_k = self.sample_counts[cid]
                contribution = (n_k / total_n) * weights[name]
                if weighted_sum is None:
                    weighted_sum = contribution.copy()
                else:
                    weighted_sum += contribution
            averaged[name] = weighted_sum

        state = OrderedDict()
        for name, w in averaged.items():
            state[name] = torch.tensor(w).to(self.device)
        self.model.load_state_dict(state)

        self.agg_times.append(time.time() - start)
        self.updates.clear()
        self.sample_counts.clear()
        return averaged

    def encrypt_global(self):
        weights = {name: param.cpu().detach().numpy() for name, param in self.model.state_dict().items()}
        return encrypt_weights(weights, self.password, self.rule, self.x0, self.r)

    def evaluate(self, test_loader):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item() * data.size(0)
                
                probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
                preds = output.argmax(dim=1).cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(target.cpu().numpy())

        avg_loss = total_loss / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0.0
        acc = accuracy_score(all_labels, all_preds) * 100.0
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = 0.5

        # Confusion Matrix (TP, TN, FP, FN)
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = (100.0 * tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        metrics = {
            'accuracy': acc,
            'loss': avg_loss,
            'precision': precision * 100.0,
            'recall': recall * 100.0,            # Sensitivity
            'specificity': specificity,
            'f1_score': f1 * 100.0,
            'roc_auc': auc,
            'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
            'labels': all_labels,
            'probs': all_probs,
            'preds': all_preds
        }
        self.accuracies.append(acc)
        self.losses.append(avg_loss)
        self.recalls.append(recall * 100.0)
        self.precisions.append(precision * 100.0)
        self.f1_scores.append(f1 * 100.0)
        self.aucs.append(auc)
        return metrics

