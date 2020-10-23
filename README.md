# Counterfeit-Bills-Detection-ML-Visualization

Given data set that provides 4 features per bill and whether the bill was genuine or counterfeit.  Using Machine Learning algorithms and the measurements provided, I built a predictor to determine whether a bill is genuine or counterfeit in `machineLearningAlgorithms.py`  In addition, a descriptive statistical analysis and pair plot is created for visualization on the data set interactions in statisticalAnalysis_visualization.py

## Algorithms
- Perceptron
- Logistic Regression
- Support Vector Machine
- Decision Tree Learning
- Random Forest
- K-Nearest Neighbor 

## Results
* Best Prediction was K-Nearest Neighbor with  `Accuracy 100%` & `Combined Accuracy 99.93%`

## Visualization
- Pair Plot
- Cross Covariance 

## Requirements
- Python 3 
### Packages
- [`numpy`](http://www.numpy.org/) version 1.16.5 or +  
- [`pandas`](https://pandas.pydata.org/) version 1.0.0 or +  
- [`scipy`](http://www.scipy.org/) version 1.3.1 or +
- [`matplotlib`](http://matplotlib.org/) version 1.3 or +
- [`scikit-learn`](http://scikit-learn.or) version 0.21.3 or +
- [`seaborn`](https://seaborn.pydata.org/) version 0.9.0 or +
* I used the [`Anaconda`](https://www.anaconda.com/products/individual) Environment to install these packages with additional, with Jupyter Notebook and Spyder IDE in addtion
* Other method to just download packages is [`Miniconda`](https://docs.conda.io/en/latest/miniconda.html)

## Data Set Bill Features `data_banknote_authentication.txt`
1. variance of Wavelet Transformed image (continuous)
2. skewness of Wavelet Transformed image (continuous)
3. curtosis of Wavelet Transformed image (continuous)
4. entropy of image (continuous)
5. class (integer)

