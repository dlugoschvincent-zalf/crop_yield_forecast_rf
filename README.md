## Setting up a Conda Environment for Crop Yield Forecasting

This document outlines the steps to set up a Conda environment for running the crop yield forecasting project, which uses random forest models to predict yields in German NUTS3 regions.

### Prerequisites

* **Anaconda or Miniconda:**  Ensure you have either Anaconda or Miniconda installed on your system. If not, you can download them from:
    * **Anaconda:** https://www.anaconda.com/products/distribution
    * **Miniconda:** https://docs.conda.io/en/latest/miniconda.html

### Steps

1. **Navigate to the Project Directory:** Open your terminal or command prompt and navigate to the directory containing the `crop_yield_forecast.yml` file.

2. **Create the Conda Environment:** Execute the following command to create a new Conda environment using the specifications defined in the YAML file:

   ```bash
   conda env create -f crop_yield_forecast.yml
   ```

   This command will:
    * Create a new environment (likely named "crop-yield" or similar, depending on the YAML file).
    * Install all the necessary packages and their dependencies, including libraries like scikit-learn (for random forests), pandas (for data manipulation), etc. 

3. **Activate the Environment:** Activate the newly created environment using:

   ```bash
   conda activate  crop_yield_forecast
   ```
4. **Verify Installation (Optional):** 
   To confirm the environment is set up correctly, you can optionally list the installed packages:
   ```bash
   conda list
   ```

### Running the Project

With the environment activated, you should be able to run the `random_forest.py` script:

```bash
python random_forest.py
```
