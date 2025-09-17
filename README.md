# Federated Learning with P4P Defense and Voting-based Detection

This project implements a federated learning simulation with robust defense mechanisms against adversarial attacks, specifically focusing on P4P (Probe-for-Probe) defense and voting-based detection. The code is structured in a Jupyter Notebook and uses PyTorch for model training and evaluation.

## Features
- **Federated Learning Simulation**: Multiple clients collaboratively train a global model without sharing raw data.
- **Attack Scenarios**: Supports benign, label-flipping, and backdoor attacks.
- **Defense Mechanisms**: Implements P4P defense and voting-based detection using clustering and anomaly detection algorithms.
- **Flexible Aggregation**: Supports both FedAvg and FedSGD aggregation strategies.
- **Non-IID Data Partitioning**: Uses Dirichlet distribution for realistic data splits among clients.
- **Result Visualization**: Plots accuracy and attack success rates for different scenarios.

## Structure
- **Config**: Set simulation parameters, attack/defense settings, and data paths.
- **Model**: Defines a simple MLP classifier using PyTorch.
- **Data Handling**: Loads and partitions the IoTDIAD dataset for federated training.
- **Defense**: Implements backdoor test set creation and malicious client detection.
- **Client/Server Classes**: Encapsulate client-side training and server-side aggregation, evaluation, and defense logic.
- **Main Execution**: Runs multiple scenarios and visualizes results.

## Usage
1. **Preprocess Data**: Use `preprocess_dataset.py` to clean, encode, and select features from the raw IoTDIAD dataset. Example:
	```bash
	python preprocess_dataset.py --input IoTDIAD_sum.csv --out IoTDIAD_processed.csv --k_features 30
	```
	The processed CSV should be placed at the path specified in the config (`csv_file`).
2. **Run Notebook**: Execute the notebook cells sequentially to simulate federated learning under different attack and defense scenarios.
3. **View Results**: Accuracy and attack success rate plots are saved as PNG files for analysis.

## Requirements
- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- matplotlib

Install dependencies with:
```bash
pip install torch scikit-learn pandas matplotlib
```

## Customization
- Adjust the `CONFIG` dictionary to change the number of clients, rounds, attack types, and defense parameters.
- Modify the model architecture in the `SimpleMLP` class as needed.

## References
- [IoTDIAD Dataset](https://drive.google.com/file/d/1LZFUD7sGw4ukyWikPZ9ukaRp11H8ZqXp/view?usp=drive_link)
- Federated Learning literature on adversarial robustness and defense strategies.

## License
This project is provided for academic and research purposes. Please cite appropriately if used in publications.