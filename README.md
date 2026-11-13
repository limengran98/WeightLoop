<div align="center">

# WeightLoop

**Weight-Space Programming with Experimental Feedback**

Anonymous code artifact · Python 3.10+ · MIT

</div>

## ⚙️ Installation

```bash
python -m pip install -e '.[scientific]'
python -m unittest discover -s tests -v
python scripts/check_release.py --links
```

The functional editor uses NumPy. The `scientific` extra installs SciPy for the static fitting program.

## ▶️ Synthetic example

```bash
python examples/feedback_demo.py --output outputs/demo
```

The example creates synthetic feedback, selects an update, saves native model parameters and checks prediction replay after reload. Choose a fresh output directory for each run.

## 🧩 Code interfaces

- `weightloop/functional.py`: output-weight and hidden-bias proposals for a 128-unit final hidden layer.
- `weightloop/selection.py`: feedback-only cross-validation and update selection.
- `programs/static_reference.py`: fixed regularized fitting and the Adam control on supplied native response arrays.
- `programs/STATIC_REFERENCE.json`: implementation settings and dependency description.
- `programs/official_*.py`: model-specific generated proposal code.
- `programs/hepg2/`: supplied-rule and free-design proposal programs.
- `tests/`: numerical, input-validation, feedback-isolation and persistence checks.

`choose_update` accepts feedback preactivations, measured outcomes, original output weights, output bias, hidden biases and integer feedback-fold IDs. It returns the selected native parameter vector and a selection report. The vector contains 128 output weights followed by 128 hidden biases. Each target vector must align with its selected rows.

The lower-level programs are hosted numerical functions. Generated programs use an injected NumPy namespace. Scientific fitting uses dataset preprocessing, native model checkpoints and model-specific response adapters supplied by the caller.

## 📄 License

[MIT](LICENSE).
