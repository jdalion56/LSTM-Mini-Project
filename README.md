# LSTM-Mini-Project

A proof-of-concept LSTM model for daily rainfall–runoff prediction on the CAMELS-US dataset, with a calibrated **GR4J + degree-day snow** physical baseline and a **linear regression** floor for comparison. All three models are trained, validated, and evaluated on the same basin and the same train/val/test split for a clean head-to-head.

## Results

Single basin **13340600** (NF Clearwater River, ID — snow-dominated, 3354 km²), test period 2000-10-01 → 2010-09-30:

| Metric | LSTM | GR4J + Snow | Linear (floor) | Threshold |
|---|---|---|---|---|
| NSE | **0.773** | 0.639 | 0.390 | > 0.70 |
| KGE | **0.870** | 0.731 | 0.434 | > 0.70 |
| RMSE (mm/day) | **1.257** | 1.582 | 2.058 | lower |
| PBIAS | 4.55% | 0.86% | 2.83% | < ±10% |

The LSTM beats the GR4J physical baseline by **+0.13 NSE** and the linear floor by **+0.38 NSE**, justifying the data-driven complexity. See [`DEVLOG.md`](DEVLOG.md) for the troubleshooting and refactor history (validation-set sliding-window leakage, overfitting countermeasures, GR4J bounds saturation, etc.).

## Folder Structure

```
LSTM-Mini-Project/
├── .gitignore
├── README.md
├── DEVLOG.md                                 ← troubleshooting & refactor log
├── Data/                                     ← place CAMELS data here (gitignored)
│   ├── basin_timeseries_v1p2_metForcing_obsFlow/
│   │   └── basin_dataset_public_v1p2/
│   │       ├── basin_mean_forcing/daymet/<HUC2>/<basin_id>_lump_cida_forcing_leap.txt
│   │       └── usgs_streamflow/<HUC2>/<basin_id>_streamflow_qc.txt
│   ├── camels_attributes_v2.0/               ← .txt attribute files
│   ├── basin_set_full_res/                   ← shapefiles (HCDN_nhru_final_671.shp)
│   └── cb_2018_us_state_5m/                  ← US state shapefile (for the basin map)
└── scripts/
    ├── lstm_gr4j_rainfall_runoff.ipynb       ← main notebook (LSTM + GR4J + Linear)
    ├── gr4j.py                               ← GR4J + degree-day snow module
    └── ...                                   ← earlier iterations
```

## Setup

### 1. Dependencies

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib geopandas
```

Tested on Python 3.13, PyTorch 2.9 (Apple Silicon MPS / CUDA / CPU all supported via auto-detection).

### 2. Data

Download CAMELS-US from <https://ral.ucar.edu/solutions/products/camels>  or <https://zenodo.org/records/15529996> and place the contents inside the `Data/` folder following the structure above. The US state shapefile (used only for the basin location map) can be downloaded from the [US Census TIGER/Line shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html) — it is optional; the rest of the pipeline runs without it.

### 3. Path configuration

The notebook reads the data root from the `CAMELS_DATA_ROOT` environment variable, falling back to `"../Data"` (relative to `scripts/`).

- **Default works** if you put data in `LSTM-Mini-Project/Data/` — no setup needed.
- **Override** if your data lives elsewhere:
  ```bash
  export CAMELS_DATA_ROOT=/path/to/your/CAMELS-US-data
  ```
  Add the line to `~/.zshrc` (or `~/.bashrc`) to make it permanent.

## Running

```bash
cd scripts/
jupyter lab lstm_gr4j_rainfall_runoff.ipynb
```

Or open the notebook directly in VSCode. Re-running all cells reproduces the metrics in the **Results** table above.

## References

- Kratzert, F. et al. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. *Hydrology and Earth System Sciences*, 22(11), 6005–6022.
- Kratzert, F. et al. (2019). Toward Improved Predictions in Ungauged Basins: Exploiting the Power of Machine Learning. *Water Resources Research*, 55(12).
- Perrin, C., Michel, C., Andreassian, V. (2003). Improvement of a parsimonious model for streamflow simulation. *Journal of Hydrology*, 279(1-4), 275–289.
- Addor, N. et al. (2017). The CAMELS data set: catchment attributes and meteorology for large-sample studies. *Hydrology and Earth System Sciences*, 21(10).
- Newman, A. J., Clark, M. P., Sampson, K., Wood, A., Hay, L. E., Bock, A., Viger, R., Blodgett, D., Brekke, L., Arnold, J. R., Hopson, T. and Duan, Q.: Development of a large-sample watershed-scale hydrometeorological dataset for the contiguous USA: dataset characteristics and assessment of regional variability in hydrologic model performance, Hydrology and Earth System Sciences, 19, 209–223, doi:10.5194/hess-19-209-2015, 2015.
- Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and meteorology for large-sample studies, Hydrology and Earth System Sciences, doi:10.5194/hess-2017-169, 2017.

## Acknowledgments

Code implementation, refactoring, and documentation were assisted by Claude (Anthropic). The project design, modeling decisions, troubleshooting direction, and result interpretation are my own work. AI-assisted commits are attributed via `Co-Authored-By` in the git history.
