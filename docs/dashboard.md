# PTM Dashboard - Interactive Dear PyGUI Frontend

An interactive dashboard for exploring the Prolog Tsetlin Machine without needing to wrestle with command lines or read massive papers on cellular automata.

## Features

The dashboard provides:

- **Hyperparameter Exploration**: Adjust clause count, states per action, specificity, threshold, random seed, and training epochs in real-time
- **Visual Training Feedback**: Watch training progress and see accuracy metrics immediately
- **Inference Results Table**: Compare predictions against targets with visual match indicators
- **Model Export**: Export trained models as portable `.ptm` artifacts with metadata
- **Clause State Visualization**: Preview internal clause states (planned enhancement)

## Quick Start

### Prerequisites

```bash
pip install dearpygui
```

Make sure you have the Prolog Tsetlin Machine package available:

```bash
export PYTHONPATH=/path/to/prolog_tsetlin/python
```

### Running the Dashboard

```bash
python dashboard.py
```

## Layout

The dashboard consists of five main windows:

1. **Hyperparameters** (top-left): Configure model parameters and initiate training
2. **Training Log** (top-center): View training progress and diagnostic messages
3. **Inference Results** (bottom-center): See prediction vs target comparisons
4. **Export Model** (bottom-left): Export trained models to `.ptm` format
5. **Clause States** (top-right): Visualize internal model state (after training)

## Hyperparameters Explained

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| Number of Clauses | Number of pattern-recognition units in the TM | 10-100 |
| States per Action | TA state memory depth | 50-200 |
| Specificity | Controls reward/penalty balance (>1.0) | 2.0-10.0 |
| Threshold | Voting threshold for prediction | 5-50 |
| Random Seed | Reproducibility control | Any integer |
| Epochs | Training iterations over data | 50-500 |

## Example Workflow

1. Launch the dashboard
2. Adjust hyperparameters in the left panel
3. Click **Train** to train on the built-in XOR problem
4. Review accuracy in the Training Log
5. Check predictions in the Inference Results table
6. Export your trained model using the **Export .ptm** button

## Exported Artifacts

The dashboard exports standard PTM `.ptm` artifacts that can be:
- Inspected with `ptmrt inspect <file.ptm>`
- Verified with `ptmrt verify <file.ptm>`
- Run with `ptmrt run <file.ptm> <input>`
- Used in production inference pipelines

## Future Enhancements

Planned features for future iterations:

- **Custom Data Loading**: Import CSV/JSON datasets
- **Real-time Clause Visualization**: Show which literals each clause includes
- **Feature Importance**: Display learned feature weights
- **Multi-class Support**: Handle multi-class classification problems
- **Benchmarking Tools**: Compare different hyperparameter configurations
- **Class II Integration**: Visualize Logic program consolidation
- **Live JSONL Streaming**: Connect to benchmark output streams

## Technical Notes

- The dashboard uses the scalar binary Tsetlin Machine reference implementation
- Training is performed on the CPU (intentionally not optimized for exploration)
- All exports conform to the `ptm.model.v1` artifact schema
- Cross-platform font handling supports Windows, macOS, and Linux

## Contributing

This is an exploratory tool for the Prolog Tsetlin Machine project. Feel free to suggest improvements or submit enhancements that help visitors explore the system's capabilities.
