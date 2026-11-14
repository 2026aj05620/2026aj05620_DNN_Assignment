# DNN Programming Assignment 2: RNN, LSTM, GRU, and Transformer

**Student ID:** 2026aj05620

## Environment
- **Platform:** Kaggle (GPU T4 x2 / P100) and Google Colab (T4 GPU)
                This is due to the issue in running P2 need to go to Kaggle
- **Python:** 3.10+
- **Core Libraries:** PyTorch 2.x, NumPy, Pandas, Matplotlib, scikit-learn
- **Other:** `datasets` (Hugging Face) for dataset utilities.

## How to Run
1. Ensure you are in the root directory (`2026aj05620_DNN_Assignment/`).
2. Open `problem1/problem1.ipynb`. Run the notebook from top to bottom (Runtime -> Restart and Run All). The notebook will automatically download the UCI HAR dataset into `problem1/data/`.
3. Open `problem2/problem2.ipynb`. Run the notebook from top to bottom. It will automatically download the Tiny Shakespeare dataset into `problem2/data/`.
4. All plots will be saved to the respective `figures/` directories, and metrics will be appended to the shared `results.csv` in the root folder.

## Total Run Time on T4 GPU
- **`problem1.ipynb`:** ~15 minutes (includes 3 RNN models × 3 seeds, adding problem sweep, and ablations).
- **`problem2.ipynb`:** ~15 minutes (includes HAR Transformer training and 3000 training steps for the Language Model).
- **Total:** ~30 minutes.

## My Variant & Assumptions
Because the "Individualisation" variants table was not included in the assignment PDF, I made the following reasonable assumptions:
- **Hidden Size:** `HIDDEN = 64` (matches the fixed parameter derivation required in P1.2).
- **Seeds:** `SEEDS = [0, 1, 2]`.
- **Individual Ablation (P1.5c):** My student ID ends in 0 (even), so I performed the **Gradient Clipping** ablation (norm 1.0 vs none) for the vanilla RNN at T=200.
- **Hand-checkable example (P2.1):** The instructor's 3-token example was missing, so I created a 3-token, 1-dimensional example with manually calculated attention weights to verify the code.

## Known Issues / What Does Not Work Perfectly
- **Vanilla RNN at T=200/400 (P1.4):** The model completely fails to learn, stalling at the 0.167 MSE baseline due to catastrophic vanishing gradients (as proven in the P1.5a gradient plots).
- **LSTM at T=50 (P1.4):** The LSTM failed to escape a local minimum at short sequence lengths (MSE ≈ 0.12–0.17), whereas the GRU solved it easily (MSE < 0.01).
- **Transformer HAR Positional Encoding (P2.3):** Surprisingly, the ablation showed that using *no* positional encoding yielded slightly higher accuracy than sinusoidal or learned encodings. This is documented in the report.
- **No-Mask LM Ablation (P2.5c):** As expected, removing the causal mask causes the model to "cheat" and learn an identity mapping, resulting in an infinite loop of the character 'e' during generation.