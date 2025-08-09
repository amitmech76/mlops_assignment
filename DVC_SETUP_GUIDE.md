# Data Version Control (DVC) Setup Guide

## 🎉 **DVC Successfully Configured!**

Your MLOps project now has comprehensive Data Version Control implemented with DVC. Here's what has been set up:

## 📋 **What's Been Configured**

### **1. DVC Initialization ✅**
- DVC repository initialized in `.dvc/`
- Auto-staging enabled for seamless Git integration
- Local remote storage configured at `E:\mlopsassignment\dvc-storage`

### **2. Data Versioning ✅**
- **Raw Datasets**: `housing_raw.csv`, `iris_raw.csv`
- **Processed Datasets**: `housing_processed.csv`, `iris_processed.csv`
- Data files moved from Git tracking to DVC tracking
- `.dvc` metadata files created for each dataset

### **3. ML Pipeline Definition ✅**
Complete pipeline defined in `dvc.yaml` with 6 stages:
1. **prepare_housing_data** - Download & preprocess housing data
2. **prepare_iris_data** - Download & preprocess iris data  
3. **train_housing_model** - Train California Housing models
4. **train_iris_model** - Train Iris classification models
5. **evaluate_models** - Compare and evaluate all models
6. **validate_deployment** - Test deployment readiness

### **4. Parameter Management ✅**
- `train_housing_params.yaml` - Housing model parameters
- `train_iris_params.yaml` - Iris model parameters
- Parameterized training for reproducible experiments

### **5. Metrics & Visualization ✅**
- `metrics/` directory for JSON metrics storage
- `plots/` directory for visualization outputs
- Model evaluation script with comprehensive reporting

## 🚀 **How to Use DVC**

### **Run Complete Pipeline:**
```bash
# Run entire ML pipeline
.venv\Scripts\python.exe -m dvc repro

# Run specific stage
.venv\Scripts\python.exe -m dvc repro train_housing_model
```

### **Check Pipeline Status:**
```bash
# See what's changed
.venv\Scripts\python.exe -m dvc status

# Show pipeline DAG
.venv\Scripts\python.exe -m dvc dag
```

### **Manage Data Versions:**
```bash
# Push data to remote storage
.venv\Scripts\python.exe -m dvc push

# Pull data from remote storage
.venv\Scripts\python.exe -m dvc pull

# Check data status
.venv\Scripts\python.exe -m dvc data status
```

### **View Metrics and Plots:**
```bash
# Show metrics comparison
.venv\Scripts\python.exe -m dvc metrics show

# Compare different experiments
.venv\Scripts\python.exe -m dvc metrics diff

# View plots
.venv\Scripts\python.exe -m dvc plots show
```

### **Experiment Management:**
```bash
# Create new experiment branch
git checkout -b experiment/new-features
# Modify parameters in train_*_params.yaml
.venv\Scripts\python.exe -m dvc repro
git add . && git commit -m "New experiment"

# Compare with main branch
.venv\Scripts\python.exe -m dvc metrics diff main
```

## 📁 **File Structure After DVC Setup**

```
mlops_assignment/
├── .dvc/                          # DVC configuration
├── data/
│   ├── .gitignore                 # Git ignores data files
│   ├── housing_raw.csv.dvc        # DVC metadata for housing data
│   ├── iris_raw.csv.dvc          # DVC metadata for iris data
│   ├── housing_processed.csv.dvc  # DVC metadata for processed housing data
│   └── iris_processed.csv.dvc    # DVC metadata for processed iris data
├── metrics/                       # JSON metrics storage
├── plots/                         # Visualization outputs
├── dvc.yaml                       # Pipeline definition
├── train_housing_params.yaml      # Housing model parameters
├── train_iris_params.yaml         # Iris model parameters
└── src/
    └── model_evaluation.py        # Model evaluation script
```

## 🔄 **Integration with Existing Workflow**

### **CI/CD Integration**
Your existing GitHub Actions can now include DVC commands:
```yaml
- name: Reproduce DVC pipeline
  run: |
    pip install dvc
    dvc repro
    dvc metrics show
```

### **Docker Integration**
Add DVC to your Dockerfile:
```dockerfile
RUN pip install dvc
COPY dvc.yaml train_*_params.yaml ./
RUN dvc repro --no-commit
```

## 📊 **Benefits You Now Have**

### **🎯 Reproducibility**
- Exact data versions linked to model experiments
- Parameterized training for consistent results
- Complete pipeline tracking from data to deployment

### **🔄 Collaboration**
- Team members get exact same data versions
- Shared remote storage for data artifacts
- Version control for datasets and experiments

### **📈 Experiment Tracking**
- Compare metrics across different experiments
- Visual plots showing model performance evolution
- Easy rollback to previous data/model versions

### **🚀 Automation**
- Automated pipeline execution
- Dependency tracking (only run what changed)
- Integration with existing ML tools (MLflow, Git)

## 🛠️ **Common DVC Commands Reference**

| Command | Purpose |
|---------|---------|
| `dvc repro` | Run pipeline (only changed stages) |
| `dvc repro --force` | Force run entire pipeline |
| `dvc status` | Check what's changed |
| `dvc dag` | Show pipeline dependency graph |
| `dvc metrics show` | Display current metrics |
| `dvc metrics diff` | Compare metrics between versions |
| `dvc plots show` | Generate and show plots |
| `dvc push` | Upload data to remote storage |
| `dvc pull` | Download data from remote storage |
| `dvc add <file>` | Add new file to DVC tracking |

## 🎉 **Next Steps**

1. **Run the pipeline**: `dvc repro` to test everything works
2. **Commit changes**: `git commit -m "Add DVC pipeline"`
3. **Set up cloud storage**: Configure S3/GCS remote for production
4. **Integrate with CI/CD**: Add DVC commands to GitHub Actions
5. **Train your team**: Share this guide with team members

## 📚 **Additional Resources**

- [DVC Documentation](https://dvc.org/doc)
- [DVC Tutorials](https://dvc.org/doc/tutorials)
- [DVC with MLflow](https://dvc.org/doc/use-cases/experiment-tracking)
- [DVC Remote Storage](https://dvc.org/doc/command-reference/remote)

Your MLOps project now has enterprise-grade data version control! 🚀
