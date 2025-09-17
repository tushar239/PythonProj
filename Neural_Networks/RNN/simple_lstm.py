# 0) Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# 1) Your scaled data (copy your array here)
training_set_scaled = np.array([
 [0.08581368], [0.09701243], [0.09433366], [0.09156187], [0.07984225],
 [0.0643277 ], [0.0585423 ], [0.06568569], [0.06109085], [0.06639259],
 [0.0614257 ], [0.07474514], [0.02797827], [0.02379269], [0.02409033],
 [0.0159238 ], [0.01078949], [0.00967334], [0.01642607], [0.02100231],
 [0.02280676], [0.02273235], [0.02810849], [0.03212665], [0.0433812 ],
 [0.04475779], [0.04790163], [0.0440695 ], [0.04648783], [0.04745517],
 [0.04873875], [0.03936305], [0.04137213], [0.04034898], [0.04784582],
 [0.04325099], [0.04356723], [0.04286033], [0.04602277], [0.05398467],
 [0.05738894], [0.05714711], [0.05569611], [0.04421832], [0.04514845],
 [0.04605997], [0.04412531], [0.03675869], [0.04486941], [0.05065481],
 [0.05214302], [0.05612397], [0.05818885], [0.06540665], [0.06882953],
 [0.07243843], [0.07993526], [0.07846566], [0.08034452], [0.08497656],
 [0.08627874], [0.08471612], [0.07454052], [0.07883771], [0.07238262],
 [0.06663442], [0.06315574], [0.06782499], [0.06823424], [0.07601012],
 [0.08082819]
], dtype=np.float32)

# (If your data is already scaled like above, you can skip re-scaling.
#  Otherwise use MinMaxScaler to scale raw values to 0-1.)

# ********* 2) Create sliding windows (sequence inputs) *********
'''
X is 3-D array
Horizontal (T1…T10) → timesteps (sequence length = 10).
Vertical (F1…F5) → features per timestep (like stock OHLCV: Open, High, Low, Close, Volume).
X= (samples/batch-size, timesteps, features)

input_shape = (10, 5)
which means “10 time steps, each with 5 features.”

LSTM Layer (with units = 5)
            ┌── U1
            ├── U2
Input seq → ├── U3
            ├── U4
            └── U5
            
Each unit (U1…U5) is like a memory cell that learns temporal patterns.
All timesteps (T1–T6) flow into each unit.
The hidden state at the last timestep (or all timesteps if return_sequences=True) becomes the output.

example of X (training input) and y (training output)
X =
[
    Timestamp1 - 3 samples in each timestamp
    [
        [f1, f2, f3, f4, f5],  - each sample has 5 features. In our example, there is only one feature.
        [f1, f2, f3, f4, f5],
        [f1, f2, f3, f4, f5]
    ],
    
    Timestamp2
    [
        [f1, f2, f3, f4, f5], 
        [f1, f2, f3, f4, f5],
        [f1, f2, f3, f4, f5]
    ],
    
    Timestamp3
    [
        [f1, f2, f3, f4, f5], 
        [f1, f2, f3, f4, f5],
        [f1, f2, f3, f4, f5]
    ]
]

In our example, after every 10 samples, 11th sample is considered as output.
So, 0-9 samples are kept in X and then 10th sample is kept in y
10-19 samples are kept in X and then 20th sample is kept in y
and so on
'''
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        input_array = data[i:i+seq_len]
        X.append(input_array)
        output_array = data[i+seq_len]
        y.append(output_array)
    X = np.array(X)   # shape -> (samples, seq_len, 1)
    y = np.array(y)   # shape -> (samples, 1)
    return X, y

seq_len = 10  # how many past steps to use to predict next step
X, y = create_sequences(training_set_scaled, seq_len)
print("X shape:", X.shape)   # (samples/batch-size, timesteps, features) - (61, 10, 1) (61, 10, 1)
print("y shape:", y.shape)   # (samples, 1) (61, 1)
print(X)
'''
[[[0.08581368]
  [0.09701243]
  [0.09433366]
  [0.09156187]
  [0.07984225]
  [0.0643277 ]
  [0.0585423 ]
  [0.06568569]
  [0.06109085]
  [0.06639259]]

 [[0.09701243]
  [0.09433366]
  [0.09156187]
  [0.07984225]
  [0.0643277 ]
  [0.0585423 ]
  [0.06568569]
  [0.06109085]
  [0.06639259]
  [0.0614257 ]]

 [[0.09433366]
  [0.09156187]
  [0.07984225]
  [0.0643277 ]
  [0.0585423 ]
  [0.06568569]
  [0.06109085]
  [0.06639259]
  [0.0614257 ]
  [0.07474514]]

 [[0.09156187]
  [0.07984225]
  [0.0643277 ]
  [0.0585423 ]
  [0.06568569]
  [0.06109085]
  [0.06639259]
  [0.0614257 ]
  [0.07474514]
  [0.02797827]]

 [[0.07984225]
  [0.0643277 ]
  [0.0585423 ]
  [0.06568569]
  [0.06109085]
  [0.06639259]
  [0.0614257 ]
  [0.07474514]
  [0.02797827]
  [0.02379269]]

 [[0.0643277 ]
  [0.0585423 ]
  [0.06568569]
  [0.06109085]
  [0.06639259]
  [0.0614257 ]
  [0.07474514]
  [0.02797827]
  [0.02379269]
  [0.02409033]]

 [[0.0585423 ]
  [0.06568569]
  [0.06109085]
  [0.06639259]
  [0.0614257 ]
  [0.07474514]
  [0.02797827]
  [0.02379269]
  [0.02409033]
  [0.0159238 ]]

 [[0.06568569]
  [0.06109085]
  [0.06639259]
  [0.0614257 ]
  [0.07474514]
  [0.02797827]
  [0.02379269]
  [0.02409033]
  [0.0159238 ]
  [0.01078949]]

 [[0.06109085]
  [0.06639259]
  [0.0614257 ]
  [0.07474514]
  [0.02797827]
  [0.02379269]
  [0.02409033]
  [0.0159238 ]
  [0.01078949]
  [0.00967334]]

 [[0.06639259]
  [0.0614257 ]
  [0.07474514]
  [0.02797827]
  [0.02379269]
  [0.02409033]
  [0.0159238 ]
  [0.01078949]
  [0.00967334]
  [0.01642607]]

 [[0.0614257 ]
  [0.07474514]
  [0.02797827]
  [0.02379269]
  [0.02409033]
  [0.0159238 ]
  [0.01078949]
  [0.00967334]
  [0.01642607]
  [0.02100231]]

 [[0.07474514]
  [0.02797827]
  [0.02379269]
  [0.02409033]
  [0.0159238 ]
  [0.01078949]
  [0.00967334]
  [0.01642607]
  [0.02100231]
  [0.02280676]]

 [[0.02797827]
  [0.02379269]
  [0.02409033]
  [0.0159238 ]
  [0.01078949]
  [0.00967334]
  [0.01642607]
  [0.02100231]
  [0.02280676]
  [0.02273235]]

 [[0.02379269]
  [0.02409033]
  [0.0159238 ]
  [0.01078949]
  [0.00967334]
  [0.01642607]
  [0.02100231]
  [0.02280676]
  [0.02273235]
  [0.02810849]]

 [[0.02409033]
  [0.0159238 ]
  [0.01078949]
  [0.00967334]
  [0.01642607]
  [0.02100231]
  [0.02280676]
  [0.02273235]
  [0.02810849]
  [0.03212665]]

 [[0.0159238 ]
  [0.01078949]
  [0.00967334]
  [0.01642607]
  [0.02100231]
  [0.02280676]
  [0.02273235]
  [0.02810849]
  [0.03212665]
  [0.0433812 ]]

 [[0.01078949]
  [0.00967334]
  [0.01642607]
  [0.02100231]
  [0.02280676]
  [0.02273235]
  [0.02810849]
  [0.03212665]
  [0.0433812 ]
  [0.04475779]]

 [[0.00967334]
  [0.01642607]
  [0.02100231]
  [0.02280676]
  [0.02273235]
  [0.02810849]
  [0.03212665]
  [0.0433812 ]
  [0.04475779]
  [0.04790163]]

 [[0.01642607]
  [0.02100231]
  [0.02280676]
  [0.02273235]
  [0.02810849]
  [0.03212665]
  [0.0433812 ]
  [0.04475779]
  [0.04790163]
  [0.0440695 ]]

 [[0.02100231]
  [0.02280676]
  [0.02273235]
  [0.02810849]
  [0.03212665]
  [0.0433812 ]
  [0.04475779]
  [0.04790163]
  [0.0440695 ]
  [0.04648783]]

 [[0.02280676]
  [0.02273235]
  [0.02810849]
  [0.03212665]
  [0.0433812 ]
  [0.04475779]
  [0.04790163]
  [0.0440695 ]
  [0.04648783]
  [0.04745517]]

 [[0.02273235]
  [0.02810849]
  [0.03212665]
  [0.0433812 ]
  [0.04475779]
  [0.04790163]
  [0.0440695 ]
  [0.04648783]
  [0.04745517]
  [0.04873875]]

 [[0.02810849]
  [0.03212665]
  [0.0433812 ]
  [0.04475779]
  [0.04790163]
  [0.0440695 ]
  [0.04648783]
  [0.04745517]
  [0.04873875]
  [0.03936305]]

 [[0.03212665]
  [0.0433812 ]
  [0.04475779]
  [0.04790163]
  [0.0440695 ]
  [0.04648783]
  [0.04745517]
  [0.04873875]
  [0.03936305]
  [0.04137213]]

 [[0.0433812 ]
  [0.04475779]
  [0.04790163]
  [0.0440695 ]
  [0.04648783]
  [0.04745517]
  [0.04873875]
  [0.03936305]
  [0.04137213]
  [0.04034898]]

 [[0.04475779]
  [0.04790163]
  [0.0440695 ]
  [0.04648783]
  [0.04745517]
  [0.04873875]
  [0.03936305]
  [0.04137213]
  [0.04034898]
  [0.04784582]]

 [[0.04790163]
  [0.0440695 ]
  [0.04648783]
  [0.04745517]
  [0.04873875]
  [0.03936305]
  [0.04137213]
  [0.04034898]
  [0.04784582]
  [0.04325099]]

 [[0.0440695 ]
  [0.04648783]
  [0.04745517]
  [0.04873875]
  [0.03936305]
  [0.04137213]
  [0.04034898]
  [0.04784582]
  [0.04325099]
  [0.04356723]]

 [[0.04648783]
  [0.04745517]
  [0.04873875]
  [0.03936305]
  [0.04137213]
  [0.04034898]
  [0.04784582]
  [0.04325099]
  [0.04356723]
  [0.04286033]]

 [[0.04745517]
  [0.04873875]
  [0.03936305]
  [0.04137213]
  [0.04034898]
  [0.04784582]
  [0.04325099]
  [0.04356723]
  [0.04286033]
  [0.04602277]]

 [[0.04873875]
  [0.03936305]
  [0.04137213]
  [0.04034898]
  [0.04784582]
  [0.04325099]
  [0.04356723]
  [0.04286033]
  [0.04602277]
  [0.05398467]]

 [[0.03936305]
  [0.04137213]
  [0.04034898]
  [0.04784582]
  [0.04325099]
  [0.04356723]
  [0.04286033]
  [0.04602277]
  [0.05398467]
  [0.05738894]]

 [[0.04137213]
  [0.04034898]
  [0.04784582]
  [0.04325099]
  [0.04356723]
  [0.04286033]
  [0.04602277]
  [0.05398467]
  [0.05738894]
  [0.05714711]]

 [[0.04034898]
  [0.04784582]
  [0.04325099]
  [0.04356723]
  [0.04286033]
  [0.04602277]
  [0.05398467]
  [0.05738894]
  [0.05714711]
  [0.05569611]]

 [[0.04784582]
  [0.04325099]
  [0.04356723]
  [0.04286033]
  [0.04602277]
  [0.05398467]
  [0.05738894]
  [0.05714711]
  [0.05569611]
  [0.04421832]]

 [[0.04325099]
  [0.04356723]
  [0.04286033]
  [0.04602277]
  [0.05398467]
  [0.05738894]
  [0.05714711]
  [0.05569611]
  [0.04421832]
  [0.04514845]]

 [[0.04356723]
  [0.04286033]
  [0.04602277]
  [0.05398467]
  [0.05738894]
  [0.05714711]
  [0.05569611]
  [0.04421832]
  [0.04514845]
  [0.04605997]]

 [[0.04286033]
  [0.04602277]
  [0.05398467]
  [0.05738894]
  [0.05714711]
  [0.05569611]
  [0.04421832]
  [0.04514845]
  [0.04605997]
  [0.04412531]]

 [[0.04602277]
  [0.05398467]
  [0.05738894]
  [0.05714711]
  [0.05569611]
  [0.04421832]
  [0.04514845]
  [0.04605997]
  [0.04412531]
  [0.03675869]]

 [[0.05398467]
  [0.05738894]
  [0.05714711]
  [0.05569611]
  [0.04421832]
  [0.04514845]
  [0.04605997]
  [0.04412531]
  [0.03675869]
  [0.04486941]]

 [[0.05738894]
  [0.05714711]
  [0.05569611]
  [0.04421832]
  [0.04514845]
  [0.04605997]
  [0.04412531]
  [0.03675869]
  [0.04486941]
  [0.05065481]]

 [[0.05714711]
  [0.05569611]
  [0.04421832]
  [0.04514845]
  [0.04605997]
  [0.04412531]
  [0.03675869]
  [0.04486941]
  [0.05065481]
  [0.05214302]]

 [[0.05569611]
  [0.04421832]
  [0.04514845]
  [0.04605997]
  [0.04412531]
  [0.03675869]
  [0.04486941]
  [0.05065481]
  [0.05214302]
  [0.05612397]]

 [[0.04421832]
  [0.04514845]
  [0.04605997]
  [0.04412531]
  [0.03675869]
  [0.04486941]
  [0.05065481]
  [0.05214302]
  [0.05612397]
  [0.05818885]]

 [[0.04514845]
  [0.04605997]
  [0.04412531]
  [0.03675869]
  [0.04486941]
  [0.05065481]
  [0.05214302]
  [0.05612397]
  [0.05818885]
  [0.06540665]]

 [[0.04605997]
  [0.04412531]
  [0.03675869]
  [0.04486941]
  [0.05065481]
  [0.05214302]
  [0.05612397]
  [0.05818885]
  [0.06540665]
  [0.06882953]]

 [[0.04412531]
  [0.03675869]
  [0.04486941]
  [0.05065481]
  [0.05214302]
  [0.05612397]
  [0.05818885]
  [0.06540665]
  [0.06882953]
  [0.07243843]]

 [[0.03675869]
  [0.04486941]
  [0.05065481]
  [0.05214302]
  [0.05612397]
  [0.05818885]
  [0.06540665]
  [0.06882953]
  [0.07243843]
  [0.07993526]]

 [[0.04486941]
  [0.05065481]
  [0.05214302]
  [0.05612397]
  [0.05818885]
  [0.06540665]
  [0.06882953]
  [0.07243843]
  [0.07993526]
  [0.07846566]]

 [[0.05065481]
  [0.05214302]
  [0.05612397]
  [0.05818885]
  [0.06540665]
  [0.06882953]
  [0.07243843]
  [0.07993526]
  [0.07846566]
  [0.08034452]]

 [[0.05214302]
  [0.05612397]
  [0.05818885]
  [0.06540665]
  [0.06882953]
  [0.07243843]
  [0.07993526]
  [0.07846566]
  [0.08034452]
  [0.08497656]]

 [[0.05612397]
  [0.05818885]
  [0.06540665]
  [0.06882953]
  [0.07243843]
  [0.07993526]
  [0.07846566]
  [0.08034452]
  [0.08497656]
  [0.08627874]]

 [[0.05818885]
  [0.06540665]
  [0.06882953]
  [0.07243843]
  [0.07993526]
  [0.07846566]
  [0.08034452]
  [0.08497656]
  [0.08627874]
  [0.08471612]]

 [[0.06540665]
  [0.06882953]
  [0.07243843]
  [0.07993526]
  [0.07846566]
  [0.08034452]
  [0.08497656]
  [0.08627874]
  [0.08471612]
  [0.07454052]]

 [[0.06882953]
  [0.07243843]
  [0.07993526]
  [0.07846566]
  [0.08034452]
  [0.08497656]
  [0.08627874]
  [0.08471612]
  [0.07454052]
  [0.07883771]]

 [[0.07243843]
  [0.07993526]
  [0.07846566]
  [0.08034452]
  [0.08497656]
  [0.08627874]
  [0.08471612]
  [0.07454052]
  [0.07883771]
  [0.07238262]]

 [[0.07993526]
  [0.07846566]
  [0.08034452]
  [0.08497656]
  [0.08627874]
  [0.08471612]
  [0.07454052]
  [0.07883771]
  [0.07238262]
  [0.06663442]]

 [[0.07846566]
  [0.08034452]
  [0.08497656]
  [0.08627874]
  [0.08471612]
  [0.07454052]
  [0.07883771]
  [0.07238262]
  [0.06663442]
  [0.06315574]]

 [[0.08034452]
  [0.08497656]
  [0.08627874]
  [0.08471612]
  [0.07454052]
  [0.07883771]
  [0.07238262]
  [0.06663442]
  [0.06315574]
  [0.06782499]]

 [[0.08497656]
  [0.08627874]
  [0.08471612]
  [0.07454052]
  [0.07883771]
  [0.07238262]
  [0.06663442]
  [0.06315574]
  [0.06782499]
  [0.06823424]]

 [[0.08627874]
  [0.08471612]
  [0.07454052]
  [0.07883771]
  [0.07238262]
  [0.06663442]
  [0.06315574]
  [0.06782499]
  [0.06823424]
  [0.07601012]]]
'''
print(y)
'''
[[0.0614257 ]
 [0.07474514]
 [0.02797827]
 [0.02379269]
 [0.02409033]
 [0.0159238 ]
 [0.01078949]
 [0.00967334]
 [0.01642607]
 [0.02100231]
 [0.02280676]
 [0.02273235]
 [0.02810849]
 [0.03212665]
 [0.0433812 ]
 [0.04475779]
 [0.04790163]
 [0.0440695 ]
 [0.04648783]
 [0.04745517]
 [0.04873875]
 [0.03936305]
 [0.04137213]
 [0.04034898]
 [0.04784582]
 [0.04325099]
 [0.04356723]
 [0.04286033]
 [0.04602277]
 [0.05398467]
 [0.05738894]
 [0.05714711]
 [0.05569611]
 [0.04421832]
 [0.04514845]
 [0.04605997]
 [0.04412531]
 [0.03675869]
 [0.04486941]
 [0.05065481]
 [0.05214302]
 [0.05612397]
 [0.05818885]
 [0.06540665]
 [0.06882953]
 [0.07243843]
 [0.07993526]
 [0.07846566]
 [0.08034452]
 [0.08497656]
 [0.08627874]
 [0.08471612]
 [0.07454052]
 [0.07883771]
 [0.07238262]
 [0.06663442]
 [0.06315574]
 [0.06782499]
 [0.06823424]
 [0.07601012]
 [0.08082819]]
'''


# *********** 3) Train / validation split ********
# Use the first 80% for training, remaining 20% for validation (time-series split — do not shuffle).
print(len(X)) # 61
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

print("Train:", X_train.shape, y_train.shape) # Train: (48, 10, 1) (48, 1)
print("Val:", X_val.shape, y_val.shape) # Val: (13, 10, 1) (13, 1)


tf.random.set_seed(42)
# input_shape=(timesteps, features)
# As per my understanding, RNN has just one hidden layer which is spread over timesteps.
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(seq_len, 1)),  # 50 units - 50 memory cells - 50 neurons in hidden layer
    tf.keras.layers.Dense(1)                             # output one value
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

'''
The model.summary() output provides a textual visualization of the layers, their order, output shapes, and the number of parameters, 
which helps in understanding the model's structure.
'''
model.summary()

# ********** 4) Build a simple LSTM model ********
# A vanilla LSTM (single layer) for sequence→one prediction.
'''
Notes
return_sequences=True is required if stacking multiple LSTM layers (so intermediate LSTM returns full sequence).
Use units (50 here) as capacity — raise to 100–200 if you have lots of data.
'''
tf.random.set_seed(42)

'''
Here units=50 means:
Each timestep’s input (shape (10,1)) is processed by 50 memory cells. seq_len=10 here.
The final output of the LSTM layer is a vector of shape (50,).
These 50 outputs are passed into the Dense(1) layer to predict one value.

Choosing number of units:
Small datasets → 10–50 units (to avoid overfitting).
Larger datasets / complex patterns → 100–200 units.
Very large datasets (NLP, stock prediction, etc.) → 256–512+ units.
Often tuned by trial and error + monitoring validation loss.

Number of units = number of LSTM neurons/memory cells = dimensionality of hidden state vector.
'''
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(seq_len, 1)),  # 50 units - 50 memory cells - 50 neurons in hidden layer
    tf.keras.layers.Dense(1)                             # output one value
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(model.summary())

# ******** 5) Train with Early Stopping *******
#es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8,
    #callbacks=[es],
    verbose=2
)
'''
# Plot the training history:
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.xlabel('epoch'); plt.ylabel('mse'); plt.title('Training history')
plt.show()
'''
# ***** 6) Predict & inverse-transform (if needed) *****
'''
If your data was scaled with a MinMaxScaler, you must inverse-transform predictions to original scale. 
In your case training_set_scaled is already scaled; if you want original values, keep the scaler used earlier. 
Below we just predict in scaled space:
'''
preds_val = model.predict(X_val)      # shape (n_val, 1)
print(preds_val)

'''
[[0.07557997]
 [0.07594149]
 [0.07798918]
 [0.07915293]
 [0.07816914]
 [0.0710742 ]
 [0.06959301]
 [0.06545059]
 [0.06031918]
 [0.05627084]
 [0.05774506]
 [0.05972669]
 [0.06583084]]
'''
preds_val_2D = np.reshape(preds_val, (1, len(preds_val)))
print(preds_val_2D)
'''
[[[0.07556766 0.07574021 0.07762866 0.0788864  0.07799482 0.07086985 0.06901561 0.06505804 0.060059   0.05604178 0.05762809 0.06001235 0.0663316 ]]]
'''
y_val_2D = np.reshape(y_val, (1, len(y_val)))
print(y_val_2D)
'''
[[[0.08034452 0.08497656 0.08627874 0.08471612 0.07454052 0.07883771 0.07238262 0.06663442 0.06315574 0.06782499 0.06823424 0.07601012 0.08082819]]]
'''
actual_and_predicated_side_by_side = np.concatenate((y_val_2D, preds_val_2D), axis=0)
print(actual_and_predicated_side_by_side)
'''
Actual
[[[0.08034452 0.08497656 0.08627874 0.08471612 0.07454052 0.07883771
   0.07238262 0.06663442 0.06315574 0.06782499 0.06823424 0.07601012
   0.08082819]]
Predicted
 [[0.0753973  0.07570183 0.07758592 0.07877119 0.07789376 0.0710189
   0.06922377 0.0651882  0.06017476 0.05612967 0.05749686 0.05964653
   0.06576078]]]
'''
'''
# Example: compare last 20 actual vs predicted
n_show = min(20, len(preds_val))
plt.plot(y_val[-n_show:], label='actual')
plt.plot(preds_val[-n_show:], label='predicted')
plt.legend(); plt.title('Val: actual vs predicted (scaled)');
plt.show()
'''
'''
# To forecast the next step after the end of the series:
last_sequence = training_set_scaled[-seq_len:]   # shape (seq_len, 1)
last_sequence = last_sequence.reshape((1, seq_len, 1))
next_pred = model.predict(last_sequence)         # predicted scaled value for next time step
print("Next step (scaled):", next_pred.ravel()[0])
'''

'''
8) Tips, common pitfalls & tuning

Shapes:                 LSTM expects (samples, timesteps, features). For 1D time series features=1.
Scaling:                Scale features (MinMax or StandardScaler). Remember to inverse-transform predicted values for reporting in original units.
Overfitting:            Use Dropout, smaller networks, or early stopping. Also consider more data/augmentation.
Batch size:             Typical values: 8–128. If using stateful=True, batch handling gets stricter.
Activation:             LSTM internals use tanh + gates (default). Don’t replace gates unless you know what you’re doing.
Loss:                   For regression use mse or mae. For classification use crossentropy and appropriate output/activation.
Seed determinism:       tf.random.set_seed for reproducible runs (still hardware-dependent).
Hardware:               use GPU for faster training (pip install tensorflow-gpu or a TF build with GPU support).
Exploding gradients:    If you see NaNs/huge updates, use tf.keras.optimizers.Adam(clipnorm=1.0) or clipvalue.
Long sequences:         If sequences are very long, consider truncating, hierarchical RNNs, or Transformer-based models.
'''

