import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import Tools

numFeatures = 10
numEpochs = 10
dataset = Tools.makeDataSet(numFeatures).values
print("Size of dataset is {}".format(dataset.shape))

# split into input (X) and output (Y) variables
X = dataset[:,0:numFeatures].astype(float)
#X = numpy.empty([dataset.shape[0], dataset.shape[1]-1])
Y = dataset[:,numFeatures]

seed = 7
numpy.random.seed(seed)
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(int(numFeatures/2), input_dim=numFeatures, kernel_initializer='normal', activation='relu'))
    model.add(Dense(int(numFeatures/4), input_dim=int(numFeatures/2), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=numEpochs, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
