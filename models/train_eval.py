import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cross_decomposition import CCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if (phase == 'train'):
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            for inputs, labels in dataloaders[phase]:
                social, wearable, academic, peer = inputs
                # Apply CCA
                cca = CCA(n_components=2)
                X_cca, _ = cca.fit_transform(social, wearable, academic, peer)
                X_cca = torch.tensor(X_cca, dtype=torch.float32)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(X_cca)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * social.size(0)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')

def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            social, wearable, academic, peer = inputs
            # Applies CCA
            cca = CCA(n_components=2)
            X_cca, _ = cca.fit_transform(social, wearable, academic, peer)
            X_cca = torch.tensor(X_cca, dtype=torch.float32)

            outputs = model(X_cca)
            all_labels.append(labels)
            all_outputs.append(outputs)

    all_labels = torch.cat(all_labels)
    all_outputs = torch.cat(all_outputs)
    accuracy = accuracy_score(all_labels, all_outputs)
    precision = precision_score(all_labels, all_outputs)
    recall = recall_score(all_labels, all_outputs)
    f1 = f1_score(all_labels, all_outputs)
    auc_roc = roc_auc_score(all_labels, all_outputs)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}')