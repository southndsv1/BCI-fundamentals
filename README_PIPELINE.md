# BCI Motor Imagery Analysis Pipeline

Complete analysis pipeline for the **2025 Multi-Day High-Quality Motor Imagery Dataset** published by Yang, B., Rong, F., Xie, Y. et al. in Nature Scientific Data.

## 🧠 Dataset Information

**Source:** [FigShare Repository](https://doi.org/10.25452/figshare.plus.22671172)

**Characteristics:**
- **62 healthy subjects** (right-handed, aged 17-30)
- **Two paradigms:**
  - 2-class: Left/right hand-grasping (51 subjects)
  - 3-class: Left/right hand-grasping + foot-hooking (11 subjects)
- **Three recording sessions** per subject (multi-day protocol)
- **64 EEG channels** (10-20 system)
- **High sampling rate** with consistent data quality

## 📋 Features

This pipeline implements a complete BCI analysis workflow:

### Stage 1: Data Loading & Preprocessing
- Automatic dataset download from FigShare (or synthetic data generation)
- Data quality inspection (SNR, outliers, class balance)
- Band-pass filtering (0.5-40 Hz, Butterworth IIR)
- Common Average Reference (CAR) spatial filtering
- Downsampling to 250 Hz
- Trial segmentation with motor imagery windows

### Stage 2: Feature Extraction & Visualization
- **Spectral Analysis:**
  - Power Spectral Density (Welch's method)
  - Band power extraction (Alpha, Beta, Gamma)
  - Topoplots showing spatial power distribution
- **Common Spatial Pattern (CSP):**
  - Supervised spatial filtering for class separation
  - Log-variance feature extraction
  - CSP component analysis

### Stage 3: Classification & Cross-Session Evaluation
- **Training Strategy:**
  - Within-session: 5-fold CV on Session 1
  - Cross-session: Train on Session 1, test on Sessions 2 & 3
- **Classifiers:**
  - CSP + LDA (Linear Discriminant Analysis)
  - CSP + SVM (Support Vector Machine)
- **Performance Metrics:**
  - Accuracy, F1-score, confusion matrices
  - Session transfer degradation analysis
  - Statistical significance testing

### Stage 4: Comprehensive Visualizations
- Power spectral density by class and session
- Within vs cross-session accuracy comparison
- Subject-wise performance heatmap
- Feature stability across sessions
- Confusion matrices per session

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd BCI-fundamentals
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Basic Usage

**Run the complete pipeline:**
```python
python bci_motor_imagery_pipeline.py
```

This will:
1. Download/generate the dataset
2. Preprocess all sessions
3. Extract features (spectral + CSP)
4. Train and evaluate classifiers
5. Generate visualizations
6. Create summary report

**Results will be saved to `./results/` directory.**

### Custom Configuration

Edit the `CONFIG` dictionary in `bci_motor_imagery_pipeline.py`:

```python
CONFIG = {
    # Preprocessing
    'filter_low': 0.5,      # High-pass filter (Hz)
    'filter_high': 40.0,    # Low-pass filter (Hz)
    'target_sfreq': 250,    # Downsampled frequency (Hz)

    # Feature extraction
    'csp_components': 4,    # Number of CSP components

    # Classification
    'cv_folds': 5,          # Cross-validation folds
    'classifiers': ['lda', 'svm'],

    # ... more options
}
```

### Advanced Usage

**Run individual pipeline stages:**

```python
from bci_motor_imagery_pipeline import BCIMultiSessionPipeline

# Create pipeline
pipeline = BCIMultiSessionPipeline()

# Run specific stages
pipeline.download_dataset()
pipeline.load_dataset()
pipeline.inspect_data_quality()
pipeline.preprocess_data()
pipeline.extract_spectral_features()
pipeline.extract_csp_features()
pipeline.train_and_evaluate()
pipeline.create_visualizations()
pipeline.generate_summary_report()
```

## 📊 Output Files

All results are saved to `./results/` directory:

### CSV Reports
- `spectral_features.csv` - Band power for all trials
- `csp_explained_variance.csv` - CSP component variance
- `within_session_performance.csv` - Within-session accuracy
- `cross_session_performance.csv` - Cross-session transfer results

### Visualizations (PNG)
- `psd_by_class_session.png` - Power spectral density
- `accuracy_within_vs_cross_session.png` - Performance comparison
- `subject_wise_performance_heatmap.png` - Subject × session heatmap
- `feature_stability_across_sessions.png` - Feature consistency
- `confusion_matrices_per_session.png` - Classification confusion matrices

### Summary Report
- `SUMMARY_REPORT.txt` - Complete analysis summary with key findings

## 🔬 Scientific Background

### Motor Imagery BCI

Motor imagery (MI) is the mental rehearsal of a movement without actual execution. During MI:
- **Event-Related Desynchronization (ERD)** occurs in mu (8-12 Hz) and beta (12-30 Hz) bands
- ERD is localized to contralateral motor cortex (left hand → right hemisphere)
- This creates distinctive spatial-spectral patterns that can be classified

### Common Spatial Pattern (CSP)

CSP is a supervised spatial filtering technique that:
1. Finds linear combinations of EEG channels
2. Maximizes variance ratio between two classes
3. Produces features robust to volume conduction

**Why CSP works:**
- Motor imagery creates focal cortical activity
- CSP enhances signal from motor cortex while suppressing other sources
- Log-variance of CSP-filtered signals is highly discriminative

### Cross-Session Challenge

**Key Scientific Question:** How well do BCI classifiers generalize across days?

Real-world BCIs face:
- Electrode impedance changes
- Subject cognitive state variations
- Environmental factors

This dataset's multi-day protocol enables rigorous evaluation of session-to-session transfer, which is critical for practical BCI systems.

## 📈 Expected Results

**Within-Session Performance (5-fold CV on Session 1):**
- CSP + LDA: 80-90% accuracy
- CSP + SVM: 80-90% accuracy

**Cross-Session Performance (Train Session 1 → Test Sessions 2 & 3):**
- CSP + LDA: 70-80% accuracy
- CSP + SVM: 70-80% accuracy

**Transfer Degradation:** 5-15% accuracy drop

**Comparison to Benchmarks:**
- BCI Competition IV-2a (2008): ~70-75% within-session
- This dataset (2025): Higher accuracy due to improved data quality

## 🛠️ Technical Details

### Preprocessing Pipeline

1. **Band-pass Filter (0.5-40 Hz):**
   - Removes DC drift (< 0.5 Hz)
   - Removes high-frequency noise (> 40 Hz)
   - Butterworth IIR filter, order 4

2. **Common Average Reference (CAR):**
   - Reduces common-mode noise
   - Assumes all electrodes see same noise
   - Enhances spatially-specific signals

3. **Downsampling (1000 → 250 Hz):**
   - Reduces computational load
   - Sufficient for capturing < 40 Hz signals
   - Uses anti-aliasing filter

### CSP Feature Extraction

**Binary CSP (2-class):**
```
1. Compute covariance matrices for each class
2. Solve generalized eigenvalue problem
3. Select top N components (default: 4)
4. Transform trials using CSP filters
5. Extract log-variance features
```

**Multi-class CSP (3-class):**
```
Use one-vs-rest approach:
- CSP for class 0 vs {1, 2}
- CSP for class 1 vs {0, 2}
- CSP for class 2 vs {0, 1}
Concatenate all features
```

### Classification

**LDA (Linear Discriminant Analysis):**
- Assumes Gaussian distributions
- Fast, interpretable
- Works well with CSP features
- Good baseline for BCI

**SVM (Support Vector Machine):**
- RBF kernel with automatic gamma scaling
- More flexible decision boundaries
- Robust to outliers
- May overfit with small datasets

## ⚠️ Important Notes

### Real Dataset Usage

This implementation includes **synthetic data generation** for demonstration. For real analysis:

1. **Download the actual dataset from:**
   https://doi.org/10.25452/figshare.plus.22671172

2. **Place data files in `./data/` directory**

3. **Modify `load_dataset()` method** to load actual .mat or .h5 files:
   ```python
   # For .mat files
   from pymatreader import read_mat
   data = read_mat('path/to/file.mat')

   # For .h5 files
   import h5py
   with h5py.File('path/to/file.h5', 'r') as f:
       data = f['dataset_name'][:]
   ```

### Computational Requirements

- **RAM:** 4-8 GB recommended
- **Runtime:** 5-10 minutes for full pipeline (synthetic data)
- **Storage:** ~500 MB for results

## 📚 References

1. **Dataset Paper:**
   Yang, B., Rong, F., Xie, Y. et al. (2025). "Multi-day high-quality motor imagery BCI dataset." Nature Scientific Data.
   DOI: 10.25452/figshare.plus.22671172

2. **CSP Algorithm:**
   Ramoser, H., Muller-Gerking, J., & Pfurtscheller, G. (2000). "Optimal spatial filtering of single trial EEG during imagined hand movement." IEEE Trans Rehab Eng, 8(4), 441-446.

3. **BCI Review:**
   Lotte, F., et al. (2018). "A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update." Journal of Neural Engineering, 15(3), 031005.

4. **Session Transfer:**
   Shenoy, P., et al. (2006). "Towards adaptive classification for BCI." Journal of Neural Engineering, 3(1), R13.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add EEGNet deep learning classifier
- [ ] Implement Riemannian geometry features
- [ ] Add online adaptation methods
- [ ] Implement transfer learning techniques
- [ ] Add more visualization options

## 📄 License

MIT License - see LICENSE file for details

## 👥 Authors

BCI Research Pipeline - 2025

## 🐛 Issues

For bugs or feature requests, please open an issue on the repository.

---

**Happy BCI Analysis! 🧠⚡**
