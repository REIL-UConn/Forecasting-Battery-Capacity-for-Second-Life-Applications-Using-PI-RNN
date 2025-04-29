# Forecasting Battery Capacity for Second-Life Applications Using Physics-Informed Recurrent Neural Networks

This repository accompanies our published paper  
> **“Forecasting Battery Capacity for Second-Life Applications Using Physics-Informed Recurrent Neural Networks”**  
> *(placeholder DOI)*  

It contains code and data for training, evaluating, and visualizing a physics-informed recurrent neural network (PI-RNN) alongside several baselines to forecast battery capacity fade in first-life and second-life stages.

![Graphical Abstract](figures/graphical_abstract.png)

---

## 🚀 Project Structure

```
.
├── processed_data/             # Preprocessed Cycling & RPT datasets (.pkl)
├── saved_models/               # Trained model weights (.pth, .pt)
├── simulated_PBM_data/         # Features from physics-based simulations (.pkl)
├── data_utils.py               # Data loaders & sequence builders
├── models.py                   # PI-RNN, baselines, PBM surrogates
├── pbm_experiments.py          # PyBaMM simulation & feature extraction
├── preprocessing.py            # Raw Excel → merged .pkl + capacity-fade plots
├── RMSE_evaluation.py          # Single/multi-step RMSE & MAE evaluation
├── training_strategies.py      # Scenario-based PI-RNN & baseline training
├── trajectory_forecast.py      # CLI forecasting + visualization
├── uncertainty_quantification.py # UQ trajectories & calibration curves
├── README.md                   # This file
└── requirements.txt            # Python dependencies required to install
```

---

## 📄 Script Overviews

- **data_utils.py**  
  - Centralizes dataset paths and feature/target definitions  
  - `load_pbm_surrogate()`, `load_batch()`, `load_battery_data()`, `make_sequences()`

- **models.py**  
  - `train_pbm_surrogate_for_PI_RNN()` — RandomForest surrogate for capacity-drop injection  
  - `CustomRNNCellWithSurrogate` & `MultiStepPIRNN` — physics-informed RNN  
  - `BaselineMultiStepRNN` — standard RNN baseline  
  - `GPRBaseline` — Gaussian Process baseline  
  - `PBMSurrogate` — multi-step PBM surrogate class  

- **pbm_experiments.py**  
  - Defines PyBaMM experiments for each group  
  - Runs CCCV/charge/RPT cycles, extracts features, saves to `simulated_PBM_data/`

- **preprocessing.py**  
  - Parses “Cycling n” Excel files, computes throughput/time duration features  
  - Reads RPT capacities from master Excel, merges, saves to `processed_data/`  
  - Plots capacity-fade curves per group

- **RMSE_evaluation.py**  
  - Trains PI-RNN, Baseline RNN, GPR, PBM surrogate   
  - Computes and plots single-step and multi-step RMSE/MAE comparisons

- **training_strategies.py**  
  - Implements three forecasting scenarios (S1: fixed-horizon, S2: recursive, S3: maximum horizon)  
  - Trains and saves PI-RNN and Baseline RNN for each scenario  
  - Provides visualization 

- **trajectory_forecast.py**  
  - CLI to load pretrained S3 models (PI-RNN & Baseline)  
  - (Optional) fine-tuning on first N points of a selected cell  
  - Combines RNN & GPR forecasts, plots trajectories & RMSE bars

- **uncertainty_quantification.py**  
  - Loads or trains the S3 PI-RNN model with MC dropout, saves model state  
  - Generates convex-combined prediction intervals across life phases  
  - Fits isotonic recalibration on “C2” cells (held out calibration set), plots calibration curves

---

## 🔧 Installation

1. **Clone**  
   ```bash
   git clone https://github.com/your-org/PI-RNN-for-Capacity-Forecasting.git
   cd PI-RNN-for-Capacity-Forecasting
   ```

2. **Create & activate** your Python environment  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage Examples

1. **Preprocess data & plot capacity fade**  
   ```bash
   python preprocessing.py
   ```

2. **Generate PBM-simulated features**  
   ```bash
   python pbm_experiments.py
   ```

3. **Train & evaluate RMSE/MAE**  
   ```bash
   python RMSE_evaluation.py
   ```

4. **Train forecasting scenarios**  
   ```bash
   python training_strategies.py
   ```

5. **Interactive trajectory forecast**  
   ```bash
   python trajectory_forecast.py --group G13 --cell C1
   ```

6. **Uncertainty quantification & calibration curves**  
   ```bash
   python uncertainty_quantification.py
   ```

---

## 📋 Requirements

See `requirements.txt` for full dependency list.

---

## 📖 Citation

Please cite our paper if you use this code:

```bibtex
@article{Navidi2025,
  title   = {Forecasting Battery Capacity for Second-Life Applications Using Physics-Informed Recurrent Neural Networks},
  author  = {...},
  journal = {Journal Name},
  year    = {2025},
  doi     = {placeholder DOI}
}
```

---

## ⚖️ License

This project is licensed under the MIT License. See `LICENSE` for details.
