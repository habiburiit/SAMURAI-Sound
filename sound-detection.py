import os
import time
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset, TensorDataset
import foolbox as fb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import argparse

# ========================= Introduction and GPU Monitoring =========================

def show_intro():
    """Display an introductory message."""
    intro_lines = [
        "  Voice Attack Resilience: AI-Driven Adversarial Defense System  ",
        "",
        "          Developers: Habibur Rahaman, Swarup Bhunuia, Atri Chatterjee          ",
        "",
        "             Copyrighted to University of Florida              "
    ]
    
    for line in intro_lines:
        print(line.center(80))
        time.sleep(0.5)

# ========================= Data Loading and Preprocessing =========================

def ensure_directory_exists(path):
    """Ensure that the directory exists; if not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def get_datasets(name):
    """Load voice datasets and apply necessary transformations."""
    data_path = './data'
    ensure_directory_exists(data_path)
    
    if name == 'LibriSpeech':
        dataset = torchaudio.datasets.LIBRISPEECH(root=data_path, url='train-clean-100', download=True)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    # Use only a small subset for faster convergence
    dataset = Subset(dataset, indices=list(range(500)))  # Use only 500 samples

    # Adjust MelSpectrogram parameters to avoid all-zero filterbanks
    transform = transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=64,  # Adjust as needed to avoid zero-value filterbanks
        n_fft=512,  # Adjust as needed
        hop_length=256
    )
    
    return dataset, transform

def collate_fn(batch):
    """Collate function to handle input and target data from the batch."""
    inputs = []
    labels = []
    max_length = 0  # Initialize max length for padding

    label_encoder = LabelEncoder()

    for item in batch:
        waveform, sample_rate, label, *_ = item  # Unpack only necessary elements
        transformed_waveform = transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        transformed_waveform = transforms.MelSpectrogram(sample_rate=16000)(transformed_waveform)
        
        if isinstance(label, str):
            labels.append(label_encoder.fit_transform([label])[0])
        else:
            labels.append(label)

        if transformed_waveform.shape[2] > max_length:
            max_length = transformed_waveform.shape[2]
        
        inputs.append(transformed_waveform)

    padded_inputs = [nn.functional.pad(input, (0, max_length - input.shape[2])) for input in inputs]
    padded_inputs = torch.stack(padded_inputs)
    
    padded_inputs = padded_inputs.repeat(1, 3, 1, 1)
    
    labels = torch.tensor(labels)
    
    return padded_inputs, labels

# ========================= Model Initialization =========================

def get_model(arch, num_classes):
    """Initialize model architecture based on user input."""
    model_map = {
        'resnet18': models.resnet18,
    }

    if arch not in model_map:
        raise ValueError(f"Unsupported architecture: {arch}")

    model = model_map[arch](weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

# ========================= Model Training =========================

def train_model(dataset_name, architecture, epochs=5):
    """Train the model on a specified dataset."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataset, transform = get_datasets(dataset_name)
    trainloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, collate_fn=collate_fn)

    model = get_model(architecture, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

    print('Finished Training')
    model_save_path = f'./{dataset_name.lower()}_{architecture}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

# ========================= Adversarial Attack =========================

def perform_deepfool_attack(model, data_loader, device):
    """Perform DeepFool attack on the model."""
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = fb.attacks.LinfDeepFoolAttack(steps=50)
    epsilons = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    adv_examples = []

    for cln_data, true_label in data_loader:
        cln_data, true_label = cln_data.to(device), true_label.to(device)
        
        cln_data = cln_data - cln_data.min()
        cln_data = cln_data / cln_data.max()
        
        assert cln_data.max() <= 1.0 and cln_data.min() >= 0.0, "Input data must be in the range [0, 1]"

        _, adv, _ = attack(fmodel, cln_data, true_label, epsilons=epsilons)
        adv_examples.append(adv[0].cpu())  # Bring back to CPU for DataLoader compatibility
    
    return adv_examples

# ========================= APC Feature Extraction =========================

def extract_apc_features(model, data_loader, device, label_type):
    """Extract APC (Adversarial Perturbation Characteristics) features from the model."""
    model.eval()
    features = []

    def get_activation(name):
        """ Hook to extract features """
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks to capture outputs of each layer
    activations = {}
    for name, layer in model.named_children():
        layer.register_forward_hook(get_activation(name))

    for inputs, labels in data_loader:
        inputs = inputs.squeeze(1)  # Ensure correct dimensions
        inputs, labels = inputs.to(device), labels.to(device)
        _ = model(inputs)  # Perform a forward pass to populate activations

        sparsity = {}
        for name, activation in activations.items():
            sparsity[name] = np.mean(activation.cpu().numpy() == 0)

        features.append({
            'sparsity': sparsity,
            'labels': labels.cpu().numpy(),
            'is_adversarial': label_type  # Add flag to indicate adversarial or not
        })

    return features

# ========================= Training Detector Model =========================

def train_detector_model(file_path):
    """Train machine learning models to detect adversarial examples."""
    df = pd.read_csv(file_path)

    # Check the number of unique classes
    unique_classes = df['is_adversarial'].unique()
    print(f"Unique classes in the dataset: {unique_classes}")

    if len(unique_classes) < 2:
        raise ValueError("The data contains only one class. Please ensure the dataset is properly prepared with both original and adversarial samples.")

    y = df['is_adversarial']
    X = df.drop('is_adversarial', axis=1)

    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(0, inplace=True)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "SVM": SVC(random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train_encoded)
        joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.pkl')
        model = joblib.load(f'{name.lower().replace(" ", "_")}_model.pkl')
        predictions = model.predict(X_test_scaled)
        evaluate_model(predictions, y_test_encoded)

    y_train_cat = to_categorical(y_train_encoded)
    y_test_cat = to_categorical(y_test_encoded)

    dnn_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])
    dnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    dnn_model.fit(X_train_scaled, y_train_cat, epochs=5, batch_size=32, verbose=1)
    dnn_model.save('dnn_model.h5')

    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    lstm_model = Sequential([
        LSTM(64, input_shape=(1, X_train_scaled.shape[1])),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train_reshaped, y_train_cat, epochs=5, batch_size=32, verbose=1)
    lstm_model.save('lstm_model.h5')

    print("Detector models trained and saved.")
    return models, scaler

# ========================= Model Evaluation =========================

def evaluate_model(predictions, y_true):
    """Evaluate model performance with various metrics."""
    accuracy = accuracy_score(y_true, predictions)
    f1 = f1_score(y_true, predictions, average='weighted')
    recall = recall_score(y_true, predictions, average='weighted')
    precision = precision_score(y_true, predictions, average='weighted')
    conf_matrix = confusion_matrix(y_true, predictions)
    print(f"Accuracy: {accuracy}\nF1 Score: {f1}\nRecall: {recall}\nPrecision: {precision}\nConfusion Matrix:\n{conf_matrix}")
    return conf_matrix

def evaluate_detector():
    """Load and evaluate the detector model on the test set."""
    models = {
        "Logistic Regression": joblib.load('logistic_regression_model.pkl'),
        "SVM": joblib.load('svm_model.pkl'),
        "XGBoost": joblib.load('xgboost_model.pkl')
    }
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv('combined_apc_features.csv')

    X = df.drop('is_adversarial', axis=1)
    y = df['is_adversarial']

    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(0, inplace=True)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.30, random_state=42)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        predictions = model.predict(X_test_scaled)
        print(f"Evaluating {name} model:")
        evaluate_model(predictions, y_test)

    dnn_model = load_model('dnn_model.h5')
    lstm_model = load_model('lstm_model.h5')

    dnn_predictions = np.argmax(dnn_model.predict(X_test_scaled), axis=1)
    print("Evaluating DNN model:")
    evaluate_model(dnn_predictions, y_test)

    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    lstm_predictions = np.argmax(lstm_model.predict(X_test_reshaped), axis=1)
    print("Evaluating LSTM model:")
    evaluate_model(lstm_predictions, y_test)

# ========================= Main Execution =========================

def main():
    """Main function to execute training, attacks, APC extraction, and evaluation."""
    show_intro()

    parser = argparse.ArgumentParser(description='Train and test adversarial detection on voice datasets')
    parser.add_argument('--dataset', type=str, choices=['LibriSpeech'], required=True, help='Voice dataset to use')
    parser.add_argument('--architecture', type=str, choices=['resnet18'], required=True, help='Model architecture to use')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--attack', action='store_true', help='Perform DeepFool attack')
    parser.add_argument('--apc', action='store_true', help='Extract APC features')
    parser.add_argument('--train_detector', action='store_true', help='Train the ML detector model')
    parser.add_argument('--evaluate_detector', action='store_true', help='Evaluate the detector model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        train_model(args.dataset, args.architecture, epochs=5)

    if args.apc or args.attack:
        dataset, _ = get_datasets(args.dataset)
        testloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
        model = get_model(args.architecture, num_classes=10).to(device)
        model.load_state_dict(torch.load(f'./{args.dataset.lower()}_{args.architecture}.pth'))
        model.eval()

    if args.apc:
        # Extract APC features for original samples
        original_features = extract_apc_features(model, testloader, device, label_type=0)
        
        # Generate adversarial examples and extract APC features for them
        adv_examples = perform_deepfool_attack(model, testloader, device)
        
        # Find the maximum length for padding
        max_len = max([ex.shape[-1] for ex in adv_examples])
        padded_adv_examples = [nn.functional.pad(ex, (0, max_len - ex.shape[-1])) for ex in adv_examples]
        
        adv_labels = [y for _, y in testloader]  # Extract labels from original DataLoader
        adv_loader = DataLoader(TensorDataset(torch.stack(padded_adv_examples), torch.tensor(adv_labels)), batch_size=1, shuffle=False)
        adv_features = extract_apc_features(model, adv_loader, device, label_type=1)

        # Combine both original and adversarial features
        combined_features = original_features + adv_features
        combined_df = pd.DataFrame(combined_features)
        combined_df.to_csv('combined_apc_features.csv', index=False)
        print("Combined APC features extracted and saved to combined_apc_features.csv")

    if args.train_detector:
        train_detector_model('combined_apc_features.csv')

    if args.evaluate_detector:
        evaluate_detector()

if __name__ == '__main__':
    main()
