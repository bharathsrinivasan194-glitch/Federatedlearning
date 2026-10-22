"""
Federated Learning Hospital Client with DNA Encryption & FedProx
Reference: "A Federated Learning-based CNN Architecture of Privacy-Preserving Osteoporosis Detection on Knee X-Ray Images" (ICCCES-2026, IEEE).
"""

import torch
import torch.nn as nn
import numpy as np
import time
from collections import OrderedDict
from DNA_crypto import encrypt_weights, decrypt_weights

class FLClient:
    def __init__(self, client_id, dataloader, model, password, server_password=None, lr=0.001, rule=None, x0=None, name=None):
        self.id = client_id
        self.name = name or f"Hospital_{client_id}"
        self.data = dataloader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.password = password
        self.server_password = server_password or password
        self.rule = rule or (client_id % 8) + 1
        self.x0 = x0 or round(0.2 + (client_id % 5) * 0.15, 2)
        self.r = 3.99

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-5
        )
        self.criterion = nn.CrossEntropyLoss()

        self.num_samples = len(dataloader.dataset)
        self.losses = []
        self.accuracies = []
        self.encrypt_times = []
        self.decrypt_times = []

    def train_local(self, global_weights=None, epochs=5, mu=0.01):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for _ in range(epochs):
            for X, y in self.data:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(X)
                loss = self.criterion(out, y)
                
                # ── FedProx Proximal Term (Handles Non-IID Hospital Data) ──
                if global_weights is not None:
                    proximal_term = 0.0
                    for param, global_w in zip(self.model.parameters(), global_weights):
                        proximal_term += ((param - global_w.to(self.device)) ** 2).sum()
                    loss += (mu / 2.0) * proximal_term

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        avg_loss = total_loss / (epochs * len(self.data)) if len(self.data) > 0 else 0.0
        acc = (100.0 * correct / total) if total > 0 else 0.0
        self.losses.append(avg_loss)
        self.accuracies.append(acc)
        self.scheduler.step()
        return avg_loss, acc

    def get_weights(self, apply_ldp=True, noise_scale=0.001):
        weights = {}
        for name, param in self.model.state_dict().items():
            w = param.cpu().detach().numpy()
            
            # ── Local Differential Privacy (LDP) ──
            if apply_ldp:
                noise = np.random.normal(0, noise_scale, w.shape)
                w = w + noise
                
            weights[name] = w
        return weights

    def set_weights(self, weights_dict):
        state = OrderedDict()
        for name, w in weights_dict.items():
            state[name] = torch.tensor(w).to(self.device)
        self.model.load_state_dict(state)

    def encrypt_and_send(self):
        weights = self.get_weights()
        start = time.time()
        encrypted = encrypt_weights(weights, self.password, self.rule, self.x0, self.r)
        self.encrypt_times.append(time.time() - start)
        return encrypted

    def receive_and_decrypt(self, encrypted_global):
        start = time.time()
        weights = decrypt_weights(encrypted_global, self.server_password)
        self.decrypt_times.append(time.time() - start)
        self.set_weights(weights)

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                out = self.model(X)
                loss += self.criterion(out, y).item()
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return (100.0 * correct / total) if total > 0 else 0.0, loss / len(test_loader)

