# Connecticut Drought Analysis

pandas + matplotlib analysis of U.S. Drought Monitor weekly data for Connecticut's 9 planning regions (2000–2026).

- `data/ct_drought_monitor_2000_2026.csv`: raw export
- `analysis.ipynb`: notebook (load → inspect → clean → charts → summary), saved with outputs

## Run it in VS Code
1. Install Python 3 and the VS Code **Jupyter** extension.
2. `pip install -r requirements.txt`
3. Open `analysis.ipynb`, pick your Python as the kernel, and click **Run All**.

## Drought forecast model (`drought_rf/`)

A random forest that predicts streamflow drought at Connecticut stream gauges 0, 7, 14 and 30 days ahead, after Heldmyer et al. (2026, HESS). Start with [`drought_rf/HOW_TO_USE.md`](drought_rf/HOW_TO_USE.md).
