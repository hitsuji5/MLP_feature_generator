# MLP_feature_generator
## Usage

### construct the model
```python
from mlp_feature_generator import MLPFeatureGenerator

model = MLPFeatureGenerator()
```
### load and create input features
```python
dir0 = 'data/female/sober'
dir1 = 'data/female/drunk'
X, y = model.get_input_features(dir0, dir1)
```
X: neighboring 9 frames (default) of Mel log spectrogram

### train the model
```python
model.fit(X, y)
```

### evaluate the model
```python
model.evaluate(X, y)
```
return accuracy

### convert data into NN features
```python
hout1, hout2, p = model.convert(X)
```
hout1: 1st hidden layer output

hout2: 2nd hidden layer output

p: probability of predicted label
