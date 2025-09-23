# Machine Learning Scripts

A collection of Python scripts explaining various machine learning concepts and Python fundamentals.

## Scripts

### Section 1: just intro of Machine Learning & Data Science... not included here

### Section 2: Linear Regression

*   **`Section-2/movie_revenue.py`**: Performs linear regression on movie revenue data using pandas, matplotlib, and scikit-learn. It reads a CSV, plots the data, applies a linear regression model, and evaluates it.

### Section 3: Python Fundamentals

*   **`python/Section-3.py`**: A revision of Python programming concepts with a focus on pandas DataFrames and Series. It covers reading data, accessing, adding, deleting, and operating on columns.
*   **`python/S3-Functions.py`**: A tutorial on Python functions, covering their anatomy, parameters vs. arguments, positional and keyword arguments, return values, and dynamic typing.
*   **`python/S3-LSD_case_study.py`**: A case study on the effect of LSD on math scores. It uses pandas, matplotlib, and scikit-learn to perform linear regression, calculate the R-squared value, and visualize the results.
*   **`python/S3-Modules.py`**: A tutorial on Python modules and packages, explaining how to use `import`, `as`, and `from ... import`.
*   **`python/S3-Objects.py`**: An introduction to Python objects, covering attributes, methods, the difference between variables and objects, and dynamic typing.

### Section 4: Gradient Descent

*   **`Section-4/Gradient_Descent.py`**: Explains the concept of Gradient Descent with a simple cost function. It visualizes the cost function and its derivative, and then implements the gradient descent algorithm from scratch. It also explores the effect of the initial guess on the result.
*   **`Section-4/Grad_Desc_2nd.py`**: Continues with Gradient Descent, introducing Mean Squared Error (MSE) as a cost function for linear regression. It explains RSS vs. MSE and demonstrates their implementation. It also covers data reshaping for scikit-learn.

### Section 5: Case Study - California Housing

*   **`Section-5/California_Housing.py`**: A comprehensive case study that walks through the entire machine learning workflow using the California Housing dataset. It covers data exploration, visualization, feature engineering, and building a linear regression model with `statsmodels` and `scikit-learn`.

## Setup

To run these scripts, you need to have Python installed.
You can install the necessary libraries using pip:

```bash
pip install pandas matplotlib scikit-learn numpy sympy statsmodels
```

After installing the libraries, you can run the Python scripts from your terminal. For example:

```bash
python Section-2/movie_revenue.py
```
