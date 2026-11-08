# Titanic Survival Prediction | MLDS 400 Homework 3

A reproducible Titanic survival prediction project implemented in **Python** and **R** using logistic regression. Each implementation runs in its own Docker container, so the grader can reproduce the results without installing Python, R, or their libraries locally.

## 1. What this repository does

Both scripts:

1. Read the Kaggle Titanic `train.csv` and `test.csv` datasets.
2. Print data dimensions, label counts, and missing-value counts.
3. Create `FamilySize = SibSp + Parch + 1`.
4. Use `Pclass`, `Age`, `Fare`, `FamilySize`, `Sex`, and `Embarked` as prediction features.
5. Fit logistic regression on labeled training data and report training and held-out accuracy using an 80/20 stratified split.
6. Refit the model on all labeled `train.csv` rows and generate predictions for the 418 passengers in Kaggle's `test.csv`.

Missing numerical values are imputed with training-set medians; missing categorical values use the most frequent training-set category. Python standardizes numerical variables and one-hot encodes categorical variables. R uses `glm(..., family = binomial())`, with categorical variables encoded as factors.

**Important:** Kaggle's `test.csv` does **not** contain the true `Survived` labels. Therefore, its accuracy cannot be calculated from the provided files. The *held-out accuracy* reported by each script is the out-of-sample performance estimate. The Python and R splits are stratified but are not guaranteed to select identical passengers, so their scores should not be compared as if they used the same holdout rows.

## 2. Requirements

- **Git** to clone this repository.
- **Docker Desktop** (macOS/Windows) or Docker Engine (Linux), installed and running.
- An internet connection for the first Docker image build.
- A free **Kaggle account** to download the Titanic data.
- A terminal supporting the commands below (macOS/Linux shell, or Windows WSL2).

No separate Python, R, Jupyter, or package installation is required on the host computer.

## 3. Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/aliciaisitongliu2027-mlds/mlds400-titanic-survival.git
cd mlds400-titanic-survival
```

**All Docker commands below must be run from the repository root**, the directory containing `README.md`, `.gitignore`, and `src/`.

## 4. Download the dataset (required)

The data is intentionally **not included** in this GitHub repository.

1. Visit the [Kaggle Titanic competition data page](https://www.kaggle.com/competitions/titanic/data).
2. Sign in to Kaggle and accept the competition terms if prompted.
3. Select **Download All** and unzip the downloaded archive.
4. Copy **`train.csv`** and **`test.csv`** into the repository's **`src/data/`** directory. Keep their filenames unchanged. If needed, create that directory with:

   ```bash
   mkdir -p src/data
   ```

5. Verify the files are in the correct location:

   ```bash
   ls src/data/train.csv src/data/test.csv
   ```

Do not copy the datasets into `src/python/` or `src/r/`. The data remains local and is excluded by `.gitignore`.

## 5. Run the Python model with Docker

First, ensure Docker Desktop / Docker Engine is running. From the **repository root**, build the Python image:

```bash
docker build -f src/python/Dockerfile -t titanic-python .
```

Then run the container, mounting the local data folder as read-only:

```bash
docker run --rm -v "$(pwd)/src/data:/app/data:ro" titanic-python
```

The script prints data diagnostics, preprocessing notes, training accuracy, held-out accuracy, and a preview of predictions for Kaggle's test set. A successful run ends with:

```text
Python pipeline completed successfully.
```

The Python image is based on `python:3.12-slim`; exact Python package versions are specified in `src/python/requirements.txt`.

## 6. Run the R model with Docker

From the **same repository root**, build the R image:

```bash
docker build -f src/r/Dockerfile -t titanic-r .
```

Then run it with the same local data folder:

```bash
docker run --rm -v "$(pwd)/src/data:/app/data:ro" titanic-r
```

The R container uses `r-base:4.4.3`. The `install_packages.R` script verifies that the built-in `stats` and `utils` packages are available. There are no additional CRAN dependencies.

The R script prints data diagnostics, preprocessing notes, training accuracy, held-out accuracy, and a preview of predictions. A successful run ends with:

```text
R pipeline completed successfully.
```

## 7. Example results

The following figures were observed when running both Docker containers with the Kaggle Titanic files during development:

| Metric | Python | R |
| --- | ---: | ---: |
| Labeled training dataset | 891 passengers | 891 passengers |
| Kaggle unlabeled test dataset | 418 passengers | 418 passengers |
| Training partition | 712 passengers | 712 passengers |
| Held-out partition | 179 passengers | 179 passengers |
| Training accuracy (80/20 split) | 80.48% | 80.06% |
| Held-out accuracy (80/20 split) | 79.89% | 82.12% |
| Predictions produced for Kaggle test set | 418 | 418 |

The held-out values are estimates from separate splits; they are **not Kaggle competition scores**. Each script also reports an in-sample accuracy after refitting on all labeled data, which is not an independent test score.

## 8. Repository structure

```text
mlds400-titanic-survival/
├── README.md                  # Setup and execution instructions
├── .gitignore                 # Excludes data and temporary files
└── src/
    ├── data/
    │   ├── .gitkeep           # Tracks the otherwise-empty directory
    │   ├── train.csv          # Download locally; NOT committed to Git
    │   └── test.csv           # Download locally; NOT committed to Git
    ├── python/
    │   ├── Dockerfile         # Python container definition
    │   ├── main.py            # Python preprocessing and model
    │   └── requirements.txt   # Pinned Python dependencies
    └── r/
        ├── Dockerfile         # R container definition
        ├── install_packages.R # Checks R packages
        └── main.R             # R preprocessing and model
```

The `.gitignore` file intentionally excludes `src/data/*` except `.gitkeep`. **Do not upload the datasets to GitHub.**

## 9. Troubleshooting

- **`Cannot connect to the Docker daemon`**: Launch Docker Desktop and verify its engine is running before building or running images.
- **`docker: command not found`**: Install Docker Desktop or Docker Engine, reopen the terminal, and retry.
- **`Missing train.csv or test.csv`**: Check that both files are named exactly as shown and live in `src/data/` (not in a nested unzipped folder). Recheck the data download instructions in Section 4.
- **Docker `COPY` fails / required files not found**: Run `docker build` from the repository root, and keep the final `.` in the build command.
- **Changes to Python or R scripts are not reflected**: Rebuild the relevant image with the corresponding `docker build` command before running it again.
- **No prediction CSV appears**: This assignment prints the first ten test predictions to the terminal; the scripts do not write a submission CSV file.

## 10. Data source

[Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data). Data must be downloaded separately by each user and is not distributed with this repository.
