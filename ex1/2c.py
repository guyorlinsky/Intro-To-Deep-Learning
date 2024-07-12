# ml libraries
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # For classification
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import random
# %matplotlib inline

# load data
neg_df = pd.read_csv("ex1 data/neg_A0201.txt", header=None)
neg_df["result"] = 0.0
pos_df = pd.read_csv("ex1 data/pos_A0201.txt", header=None)
pos_df["result"] = 1.0


# combine negative and positive example

accs = []
precs = []
recalls = []
f1s = []
tot_train_losses = []
tot_val_losses = []
for i in range(30):
    print("run: {}".format(i+1))
    # combine negative and positive example
    random_seed = random.randint(1, 99999)
    df = pd.concat([neg_df, pos_df], ignore_index=True)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # add frequencies
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

    def calculate_frequencies(peptide):
        frequency = np.zeros(len(amino_acids), dtype=float)
        for aa in peptide:
            if aa in aa_to_index:
                frequency[aa_to_index[aa]] += 1
        frequency /= len(peptide)
        return frequency

    # Calculate amino acid frequencies for each 9-mer and add each as a new column
    frequencies = df[0].apply(calculate_frequencies)
    frequency_df = pd.DataFrame(frequencies.tolist(),
                                columns=[f'freq_{aa}' for aa in amino_acids])

    # Concatenate the original DataFrame with the frequency DataFrame
    df = pd.concat([df, frequency_df], axis=1)

    # transform features -> out x_df
    new_df = df[0].apply(lambda x: pd.Series(list(x))).add_prefix('col_')
    df = pd.concat([df, pd.get_dummies(new_df, dtype=float)], axis=1)
    y_df=df['result']
    x_df = df.drop([0, "result"], axis=1)
    X=x_df.values.astype('float32')
    y=y_df.values.astype('float32')

    # split to train, validation, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_seed, stratify=y)
    # X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=random_seed, stratify=y_train_val)

    X_train_resampled, y_train_resampled = X_train, y_train
    # Tensors
    X_train = torch.tensor(X_train_resampled)
    y_train = torch.tensor(y_train_resampled)
    # X_val = torch.tensor(X_val)
    # y_val = torch.tensor(y_val)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)

    """## Train Model"""
    # Model configuraitons
    num_features = X_train.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(num_features,7),
        torch.nn.Tanh(),
        torch.nn.Linear(7, 1),
        torch.nn.Flatten(0,1),
        torch.nn.Sigmoid()
    )

    loss_fn = torch.nn.BCELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # TRAINING
    train_losses = []
    val_losses = []

    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        # predict
        y_pred = model(X_train)

        # train loss
        loss = loss_fn(y_pred, y_train)
        train_losses.append(loss.item())

        # update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            loss = loss_fn(y_pred, y_test)
            val_losses.append(loss.item())

        if epoch % (num_epochs/10) == (num_epochs/10)-1:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')

    tot_train_losses.append(train_losses)
    tot_val_losses.append(val_losses)

    """## Test Evaluation"""
    with torch.no_grad():  # Disable gradient calculation
        y_pred = predictions = model(X_test)
    y_pred = (y_pred >= 0.45).float()

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred) # how many of those we said are positive, actually are
    recall = recall_score(y_test, y_pred) # how many of the actual positives we have found
    f1 = f1_score(y_test, y_pred)
    # high recall + low precision = we overpredict positive -> increase threshold
    # low recall + high precision = we underpredict positive -> decrease threshold

    accs.append(accuracy)
    precs.append(precision)
    recalls.append(recall)
    f1s.append(f1)

# Plotting the train and validation losses
plt.figure(figsize=(10, 5))
train_losses = np.mean(tot_train_losses, axis=0)
val_losses = np.mean(tot_val_losses, axis=0)
plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Losses average over 30 runs')
plt.legend()
plt.show()

print("Accuracy:", np.mean(accs))
print("Precision:", np.mean(precs))
print("Recall:", np.mean(recalls))
print("F1 Score:", np.mean(f1s))



