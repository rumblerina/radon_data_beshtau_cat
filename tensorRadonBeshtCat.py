import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.dates import DateFormatter
import numpy as np
from numpy.core.arrayprint import DatetimeFormat
from numpy.lib.histograms import _ravel_and_check_weights
import pandas as pd
from pandas.io.formats.format import Datetime64Formatter
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import warnings
import datetime as dt
warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#dataset
column_names = ['time', 'temp', 'pressure0', 'dp', 'airhumidity', 'winddir', 'windspd10', 'weathercnd', 'Td', 'mm precip12', 'ur/h', 'usv/h']
catcols = ['winddir', 'weathercnd']
raw_dataset = pd.read_csv("D:\\Users\\User\\Desktop\\radon stuff\\beshtau_weather.csv", names = column_names, na_values='?', sep =',', skiprows = 1)
ds = raw_dataset.copy()
ds = ds.drop('ur/h', 1)
ds = ds.dropna()
timestamps_orig = ds.pop('time')
#onehot encoding
ds = pd.get_dummies(ds, columns = catcols, dummy_na=False)
#print(pred_set)

#function for collapsing categoricals
def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

#split into test and train sets:
ds_trn = ds.sample(frac = 0.8, random_state = 0)
ds_tst = ds.drop(ds_trn.index)
# print(ds_tst)

#split features from labels, label - value we're looking for
train_features = ds_trn.copy()
test_features = ds_tst.copy()
train_labels = train_features.pop('usv/h')
test_labels = test_features.pop('usv/h')

# #check out the graphs
# #sns.pairplot(train_dataset[['usv/h', 't', 'p0', 'humid', 'windspd', 'Td', 'rrr', 'mmprecip24', 'mmprecip48']], diag_kind='kde')
# #plt.show()
#normalize values:
#train_features = np.asarray(train_features).astype('float32') 
#ALWAYS CHECK FOR ',' INSTEAD OF '.' IN DATA!
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
first = np.array(train_features[:1])

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 0.2])
    plt.xlabel('Epoch')
    plt.ylabel('Error [t]')
    plt.legend()
    plt.grid(True)

#define the model
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(256, activation='sigmoid'),
        layers.Dense(256, activation='sigmoid'),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model

# callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 50)

#Full DNN model
dnn_model = build_and_compile_model(normalizer)
#dnn_model.summary()
history = dnn_model.fit(
    train_features, train_labels,
    batch_size = 256,
    validation_split=0.2,
    verbose=1, epochs=2500,
    # callbacks = [callback]
    )

# plot_loss(history)
# plt.show()

#Predictions
# predictions = dnn_model.predict(test_features).flatten()
# a = plt.axes(aspect='equal')
# plt.scatter(test_labels, predictions)
# plt.xlabel('True Values [usv/h]')
# plt.ylabel('Predictions [usv/h]')
# plt.show()

#Error distribution for predictions
# error = predictions - test_labels
# plt.hist(error, bins=25)
# plt.xlabel('Prediction Error [usv/h]')
# _ = plt.ylabel('Count')
# plt.show()

#Show original plot
# y_orig = ds['usv/h']
# plt.plot(timestamps_orig,y_orig)
# plt.gcf().autofmt_xdate()
# plt.xticks(timestamps_orig[::56])
# plt.xlabel("Date")
# plt.ylabel("Dose rate, usv/h")
# plt.title("Actual doserate at mt. Beshtau in 2018-19")
# plt.ylim([0.4, 0.85])
# plt.show()
 
#define the savitzky-golay smoothing algorithm
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError('window_size and order have to be type int')
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError('Window size must be a positive odd number')
    if window_size < order + 2:
        raise TypeError('Window_size is too small for the polynomials order')
    order_range = range(order+1)
    half_window = (window_size - 1) // 2
    #precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode = 'valid')
#Make predictions for 2020
p_column_names = ['time', 'temp', 'pressure0', 'dp', 'airhumidity', 'winddir', 'windspd10', 'weathercnd', 'Td', 'mm precip12']
ds_pred = pd.read_csv("D:\\Users\\User\\Desktop\\radon stuff\\beshtau2020w.csv", names = p_column_names, na_values='?', sep =',', skiprows = 1)
timestamps = ds_pred.pop('time')
ds_pred = pd.get_dummies(ds_pred, columns = catcols, dummy_na=False)
hs_pred = list(ds_pred.columns)
hs_trn = list(ds_trn.columns)
hs_diff = [x for x in hs_trn if x not in hs_pred]
hs_diff.remove('usv/h')
ds_pred = np.asarray(ds_pred).astype(np.float32)
print(ds_pred)
print(hs_pred)

predix = dnn_model.predict(ds_pred).flatten()
timestamps = timestamps[:-1]
pred_smoothed = savitzky_golay(predix, 25, 2)
predix = predix[:-1]
output1 = pd.DataFrame(predix, index=timestamps, columns = ['prediction'])
output2 = pd.DataFrame(pred_smoothed, index=timestamps, columns = ['smoothed_p'])
outs = [output1, output2]
output = pd.concat(outs, 1)
output.to_csv('outputcat.csv')
#plt.plot(timestamps,predix, color = 'red')
plt.plot(timestamps,predix, color = 'turquoise')
plt.plot(timestamps,pred_smoothed, color = 'blue')
plt.gcf().autofmt_xdate()
plt.xticks(timestamps[::56])
plt.xlabel("Date")
plt.ylabel("Dose rate, usv/h")
plt.title('Predicted dose rate at mt. Beshtau in 2019-20')
#plt.savefig('prediction.svg', format = 'svg', dpi = 1200)
plt.ylim([0.35, 0.7])
plt.show()

# #sns.pairplot(raw_dataset, kind = 'reg', plot_kws = {'line_kws':{'color':'blue'}})
# #plt.show()