This project runs a Support Vector Machine model to classify breast cancer tumors using Breast Cancer Wisconsin (Diagnostic) dataset. You can find the dataset 
[here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)). This project is done from scratch and without use of any ML packages. Feel free to load your own datasets to train your own models. Two training methods are provided: (1) training using barrier interior point method with Newton steps, for optimization of quadratic programs and (2) ellipsoid method. Information on the details of implementations of these algorithms is provided in `Training Documentation.ipynb` notebook.

<!--- <a href="http://tensorlayer.readthedocs.io">--->
<div align="center">
	<img src="data.jpeg" width="50%" height="10%"/>
</div>
</a>

To clone the project use:
```bash
git clone https://github.com/RezaSoleymanifar/Softmargin-SVM-training-from-scratch.git
```

To install required packages run:

```bash
pip install -r requirements.txt
```

- Start training using interior point method:

```bash
python train_ipm.py regularization_factor max_iterations 'seed_file.npy'
```
`regularization_factor` can be any non-negative real number and determines the amount of regularization of the parameters of the model, `max_iterations` determines the total number of iterations of the interior point method, and `'seed_file.npy'` initializes the quadratic program. If no seed is available use `'None'` instead and default initialization is used. At each 10 iterations model parameters is saved to `best_seed.npy` and can be used for prediction and initialization.

- To train SVM using ellipsoid method:
```bash
python train_elps.py regularization_factor max_iterations 'seed_file.npy'
```

Model parameters here are saved to `best_seed_elps.npy`.

- To make predictions:

```bash
python predict.py 'seed_file.npy'
```
An accuracy score is reported and predictions are save to `predictions.csv` file.

__Results__: This model able to achieve an accuracy score of 96% using interior point method and 94% using ellipsoid method.

### Data Description


1) ID number 
2) Diagnosis (M = malignant, B = benign) 
3-32) 

Ten real-valued features are computed for each cell nucleus: 

- radius (mean of distances from center to points on the perimeter) 
- texture (standard deviation of gray-scale values) 
- perimeter 
- area 
- smoothness (local variation in radius lengths) 
- compactness (perimeter^2 / area - 1.0) 
- concavity (severity of concave portions of the contour) 
- concave points (number of concave portions of the contour) 
- symmetry 
- fractal dimension ("coastline approximation" - 1)
