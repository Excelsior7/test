```python
import torch
```

***
### LINEAR REGRESSION MODEL IMPLEMENTATION 
***

**DESIGN MATRIX : X**      
**LABELS : Y**


- Data generating process : $ y = Ax + b + \epsilon$

    - input : $ x \in \mathbb{R}^{2}$
    - output : $ y \in \mathbb{R}$
    - $\epsilon  \sim \mathcal{N}(0,1)$ 
    
    - Parameters :
        - $ A \in \mathbb{R}^{1 \times 2}$
        - $ b \in \mathbb{R}$

### *DATA*


```python
A = torch.tensor([4.0,7.0]);
b = torch.tensor(7.0);
```


```python
X = torch.rand(100,2, dtype=torch.float32);
Y = (X @ A) + b + torch.distributions.normal.Normal(0,1).sample((100,));
Y = Y.reshape((X.shape[0],1));

X[:10],Y[:10]
```




    (tensor([[0.7551, 0.2889],
             [0.2451, 0.7929],
             [0.9558, 0.3290],
             [0.6788, 0.4839],
             [0.6988, 0.2789],
             [0.3290, 0.5911],
             [0.6061, 0.4946],
             [0.9630, 0.4950],
             [0.0860, 0.0548],
             [0.8782, 0.0891]]),
     tensor([[11.3543],
             [14.1888],
             [14.4696],
             [12.3760],
             [12.4557],
             [12.8839],
             [13.4916],
             [14.3634],
             [ 7.6436],
             [10.1428]]))




```python
batch_size = 10;

tensor_dataset = torch.utils.data.TensorDataset(X,Y);
data_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size, shuffle=True);
```

### *LOSS*


```python
loss = torch.nn.MSELoss();
```

***   
### TRAINING I 
#### Without using *torch.nn.Sequential*

### *MODEL*


```python
model = torch.nn.Linear(2,1);
```

### *OPTIMIZATION ALGORITHM*


```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01);
```

***


```python
num_epochs = 10;

for epoch in range(num_epochs):
    for X_batch,Y_batch in data_loader:
        model_loss = loss(model(X_batch), Y_batch);
        model_loss.backward();
        optimizer.step();
        optimizer.zero_grad();
    
    print(f'epoch #{epoch+1}, loss {loss(model(X), Y)}');
```

    epoch #1, loss 92.0
    epoch #2, loss 49.66971206665039
    epoch #3, loss 27.17513656616211
    epoch #4, loss 15.177179336547852
    epoch #5, loss 8.782000541687012
    epoch #6, loss 5.377230644226074
    epoch #7, loss 3.5595459938049316
    epoch #8, loss 2.5957305431365967
    epoch #9, loss 2.0655875205993652
    epoch #10, loss 1.7824013233184814


***
### TRAINING II
#### Using torch.nn.Sequential

### *MODEL*


```python
modelS = torch.nn.Sequential(torch.nn.Linear(2,1));
```

### *OPTIMIZATION ALGORITHM*



```python
optimizerS = torch.optim.SGD(modelS.parameters(), lr=0.01);
```

***


```python
num_epochs = 10;

for epoch in range(num_epochs):
    for X_batch,Y_batch in data_loader:
        model_loss = loss(modelS(X_batch), Y_batch);
        model_loss.backward();
        optimizerS.step();
        optimizerS.zero_grad();
    
    print(f'epoch #{epoch+1}, loss {loss(modelS(X), Y)}');
```

    epoch #1, loss 51.292503356933594
    epoch #2, loss 28.016447067260742
    epoch #3, loss 15.599756240844727
    epoch #4, loss 8.993050575256348
    epoch #5, loss 5.4934611320495605
    epoch #6, loss 3.6124825477600098
    epoch #7, loss 2.6174347400665283
    epoch #8, loss 2.0772430896759033
    epoch #9, loss 1.7918494939804077
    epoch #10, loss 1.6288971900939941

