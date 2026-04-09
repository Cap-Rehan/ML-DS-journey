# Machine Learning Scripts

A collection of Python scripts explaining various machine learning concepts and Python fundamentals.

## Repository Structure

### Section 1: Introduction
*Not included in this repository.*

### Section 2: Linear Regression
*   **`Section-2/movie_revenue.py`**: Performs linear regression on movie revenue data using pandas, matplotlib, and scikit-learn.
*   **`Section-2/cost_revenue_clean.csv`**: Dataset used for the movie revenue analysis.

### Section 3: Python Fundamentals (in `python/` directory)
*   **`python/Section-3.py`**: A revision of Python programming concepts with a focus on pandas DataFrames and Series. Covers reading data, accessing, adding, deleting, and operating on columns.
*   **`python/S3-Functions.py`**: A tutorial on Python functions, covering their anatomy, parameters, arguments, return values, and dynamic typing.
*   **`python/S3-LSD_case_study.py`**: A case study on the effect of LSD on math scores using pandas, matplotlib, and scikit-learn.
*   **`python/lsd_math_score_data.csv`**: Dataset used for the LSD case study.
*   **`python/S3-Modules.py`**: A tutorial on Python modules and packages, explaining `import`, `as`, and `from ... import`.
*   **`python/life.py`**: A helper module used to demonstrate imports in `S3-Modules.py`.
*   **`python/S3-Objects.py`**: An introduction to Python objects, covering attributes, methods, and the difference between variables and objects.

### Section 4: Gradient Descent
*   **`Section-4/Gradient_Descent.py`**: Explains Gradient Descent from scratch with visualization of the cost function and its derivative.
*   **`Section-4/Grad_Desc_2nd.py`**: Continues with Gradient Descent, introducing Mean Squared Error (MSE) as a cost function for linear regression and exploring RSS vs. MSE.

### Section 5: Case Study - California Housing
*   **`Section-5/California_Housing.py`**: A comprehensive machine learning workflow using the California Housing dataset. Covers data exploration, visualization (including `seaborn`), feature engineering, and modeling with `statsmodels` and `scikit-learn`.
*   **`Section-5/allcorr_plot.pdf`**: A visualization output showing the correlation between features in the California Housing dataset.

## Setup

To run these scripts, you need to have Python installed.
You can install the necessary libraries using pip:

```bash
pip install pandas matplotlib scikit-learn numpy sympy statsmodels seaborn
```

After installing the libraries, you can run the Python scripts from your terminal. For example:

```bash
python Section-2/movie_revenue.py
```

## Tools
*   **`ML-DS.code-workspace`**: VS Code workspace configuration for this project.
