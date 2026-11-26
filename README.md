# Modelling UK Government Debt, Bond Markets, and Wealth Transfer: An AI Agent-Based Simulation

MSc Artificial Intelligence — Capstone Project  
University of Roehampton

## 1. Project overview

This project implements a reinforcement learning model within a simplified agent-based simulation of the UK government bond market. It investigates how adaptive financial decisions and different risk-aversion configurations influence bond allocation, coupon income and wealth distribution.

The model uses Python and the Mesa modelling framework. Commercial banks, institutional investors and households use independent tabular Q-learning, while the government agent follows predefined behavioural rules. The simulation uses historical economic observations from 1997 to 2025.

The Jupyter notebook is the main research artefact. No separate web application is required to execute it.

## 2. Project files

Keep the notebook and three original input datasets in the same working directory when running the project.

```text
Project/
├── README.md
├── UK_Debt_RL_Project(1).ipynb
├── series-230626.csv
├── BoE-Database_export.csv
├── API_GBR_DS2_en_csv_v2_448313.csv
├── master_dataset(1).csv
├── Risk_Preference_Experiment_Results(1).csv
├── Risk_Preference_Statistical_Summary.csv
├── Risk_Preference_Verification.csv
├── Detailed_Accounting_Verification(1).csv
└── Q_Learning_Training_Results(1).csv
```

The `(1)` suffix on some CSV files reflects the downloaded filenames. When the notebook regenerates these files, it uses the same names without that suffix.

### Input datasets

| File | Description |
| --- | --- |
| `series-230626.csv` | UK government borrowing data |
| `BoE-Database_export.csv` | Bank of England government bond yield data |
| `API_GBR_DS2_en_csv_v2_448313.csv` | World Bank macroeconomic indicators for the UK |

The notebook cleans and merges the original datasets by year to produce `master_dataset.csv`. The merged dataset contains 29 annual observations and eight columns: year, government borrowing, bond yield, population, GDP, inflation, unemployment and government expenditure.

## 3. Software requirements

- Python 3
- Google Colab or Jupyter Notebook
- Mesa 3.5.2
- NumPy, Pandas, Matplotlib, Seaborn, SciPy and NetworkX

The first notebook cell installs Mesa and most dependencies. If NetworkX is not already installed, install it as well. For a local Python environment, the dependencies can be installed using:

```bash
python -m pip install mesa==3.5.2 numpy pandas matplotlib seaborn scipy networkx notebook
```

## 4. How to run the project

### Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `UK_Debt_RL_Project(1).ipynb` and open it.
3. Use the Files panel to upload the three original CSV input datasets to the current runtime's working directory (normally `/content`). Keep their filenames exactly as listed above.
4. Select **Runtime > Run all** to execute the notebook in order. If NetworkX is missing, run `!pip install networkx` before the imports cell.
5. Allow the training and evaluation cells to finish. Training runs 300 episodes in total.
6. Review the verification checks, statistical tables and graphs displayed in the notebook.
7. Run the final export cell (saved execution number 58) to generate the experimental results CSV files. Download these from the Files panel if required.

### Local Jupyter Notebook

Install the dependencies listed in Section 3, open a terminal in the project directory, and run:

```bash
jupyter notebook
```

Open `UK_Debt_RL_Project(1).ipynb` and execute the cells from top to bottom. The original datasets must be accessible using the relative filenames in the notebook. If the dependencies are already installed in the active environment, the initial installation cell can be skipped.

The supplied notebook and CSV results can also be inspected without retraining. The trained Q-tables are held in memory during execution; the CSV exports contain results rather than saved policy checkpoints. A fresh runtime requires training again to reproduce policy evaluation.

## 5. Model architecture

| Agent type | Number | Decision-making approach |
| --- | ---: | --- |
| Government | 1 | Predefined behavioural rules |
| Commercial banks | 5 | Tabular Q-learning |
| Institutional investors | 20 | Tabular Q-learning |
| Households | 100 | Tabular Q-learning |
| **Total** | **126** | **125 learning agents** |

At each annual simulation step, the model updates historical economic conditions, issues bonds, collects investment decisions, allocates available bonds proportionally to demand, pays coupons, updates Q-values during training and records the resulting financial and learning metrics.

Historical non-negative borrowing values are normalised using the median positive borrowing value to adjust bond issuance. The model uses a fixed bond price of £100, a base issuance amount of £100,000 and a coupon rate of 3%.

### Reinforcement learning design

- **Algorithm:** independent tabular Q-learning, with a separate Q-table for each financial agent.
- **State:** bond yield, cash-to-wealth ratio, bond-holdings-to-wealth ratio, GDP and inflation. Each feature is discretised into three categories, giving up to 243 theoretical state combinations before accounting for dependencies and reachability.
- **Actions:** banks and investors choose investment fractions of 0%, 5%, 10%, 15% or 20%. Bank investments are additionally limited to £20,000 per step. Households choose from eight predefined consumption and savings-investment combinations.
- **Action selection:** epsilon-greedy exploration during training and greedy selection during evaluation, with random selection among equally valued actions.
- **Reward:** change in financial wealth minus a risk penalty based on bond exposure, risk aversion and the absolute yield movement. Households additionally receive consumption utility equal to 2% of their consumption.
- **Q-value update:** `Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s_next,a_next)) - Q(s,a)]`. The future value is zero at the terminal step.

## 6. Experimental configuration

| Parameter | Value |
| --- | --- |
| Historical period | 1997–2025 |
| Annual steps per episode | 29 |
| Learning rate (alpha) | 0.10 |
| Discount factor (gamma) | 0.95 |
| Initial epsilon | 1.0 |
| Minimum epsilon | 0.05 |
| Epsilon decay per episode | 0.995 |
| Risk-aversion coefficients | High: 0.10; Medium: 0.05; Low: 0.01 |
| Training episodes per scenario | 100 |
| Total training episodes | 300 |
| Evaluation runs per scenario | 10 |
| Total evaluation runs | 30 |
| Evaluation epsilon | 0.0 (learning disabled) |
| Base random seed | 42 |

Evaluation seeds: `1, 10, 42, 100, 500, 1000, 2026, 2027, 2028, 2029`.

The financial environment is reset between training episodes while the agents' Q-tables are retained within each risk-aversion scenario. Evaluation uses the trained policies without updating the Q-tables.

## 7. Verification and experimental results

The notebook includes individual agent tests, proportional allocation testing, Q-table immutability testing and financial accounting verification.

The supplied experimental results record:

- 300 training episodes across three risk-aversion scenarios.
- 30 evaluation runs, with financial accounting verification passed in all 30.
- Five detailed accounting checks passed: bond holdings reconciliation, government cash reconciliation, coupon income reconciliation, government debt reconciliation and non-negative agent balances.

### Output files

| Generated file | Contents |
| --- | --- |
| `master_dataset.csv` | Cleaned and merged historical data |
| `Risk_Preference_Experiment_Results.csv` | Individual evaluation results, including bond holdings, coupon income, wealth, government debt and accounting status |
| `Risk_Preference_Statistical_Summary.csv` | Mean, standard deviation, minimum and maximum by risk-aversion scenario (multi-row CSV header) |
| `Risk_Preference_Verification.csv` | Number of evaluation runs and accounting status for each scenario |
| `Detailed_Accounting_Verification.csv` | Detailed results of five accounting checks |
| `Q_Learning_Training_Results.csv` | Per-episode rewards, epsilon, learning-step counts and Q-table state counts |

For learning performance, use the training results CSV. The evaluation reward fields are zero because the reward-accumulating learning update is skipped when learning is disabled; those fields should not be interpreted as measured evaluation returns.

## 8. Limitations and intended use

This is an exploratory computational research model, not a prediction of actual UK government debt, investment returns or household wealth. The financial accounting tests establish internal consistency rather than real-world predictive validity.

The current experiments reuse the same historical economic sequence for training and evaluation. They do not establish policy generalisation to unseen economic conditions or superiority to a rule-based baseline. Training demonstrates Q-learning activity but does not establish convergence; epsilon remains approximately 0.606 after 100 decay steps. The bank investment cap restricts the effective action choices, and the reward design and timing of next-state observations could be refined in future work.

The notebook and exported results are provided for academic assessment and reproducibility.
