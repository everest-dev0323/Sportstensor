import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pylab as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json

# Make the fonts bigger
plt.rc('figure', figsize=(14, 7))
plt.rc('font', family='sans-serif', weight='bold', size=15)

ROOT_DIR = './data'
LEAGUES_PATH = './leagues.json'
features1 = ['HomeTeam', 'AwayTeam', 'HTeamEloScore', 'ATeamEloScore', 'HTdaysSinceLastMatch',
            'ATdaysSinceLastMatch', 'HTW_rate', 'ATW_rate', 'ATD_rate', 'HTD_rate', 
            '7_HTW_rate', '12_HTW_rate', '7_ATW_rate', '12_ATW_rate', 
            '7_HTD_rate', '12_HTD_rate', '7_ATD_rate', '12_ATD_rate',
            '7_HTL_rate', '12_HTL_rate', '7_ATL_rate', '12_ATL_rate',
            '5_HTHW_rate', '5_ATAW_rate', 'ODDS1', 'ODDSX', 'ODDS2']

features2 = ['HomeTeam', 'AwayTeam', 'HTeamEloScore', 'ATeamEloScore', 'HTdaysSinceLastMatch',
            'ATdaysSinceLastMatch', 'HTW_rate', 'ATW_rate',
            '7_HTW_rate', '12_HTW_rate', '7_ATW_rate', '12_ATW_rate', 
            '7_HTL_rate', '12_HTL_rate', '7_ATL_rate', '12_ATL_rate',
            '5_HTHW_rate', '5_ATAW_rate', 'ODDS1', 'ODDS2']

#Functions to manipulate data for use in the model
def revert_yoh(Y):
    Y_new = np.empty([Y.shape[0],Y.shape[1]], dtype="<U1")
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if (Y[i, j] == 0):
                Y_new[i, j]= 'H'
            elif (Y[i, j] == 1):
                Y_new[i, j]= 'A'
            elif (Y[i, j] == 2):
                Y_new[i, j]='D'
    return Y_new

def one_hot_y(Y):
    Y = np.array(Y)  # Convert Y to a NumPy array
    Y_new = np.zeros((Y.shape[0],3))
    for i in range(Y.shape[0]):
        if (Y[i] == 'H'):
            Y_new[i]=[1,0,0]
        elif (Y[i] == 'A'):
            Y_new[i]=[0,1,0]
        elif (Y[i] == 'D'):
            Y_new[i]=[0,0,1]
    return Y_new

def time_step(a_prev):
    a_prev = a_prev[np.newaxis, ...]
    prev_f = a_prev.shape[2] # match feature length
    input_step = 10 # sequence length
    Ttot = int(a_prev.shape[1] / input_step)
    step = 0
    a_new = np.zeros((Ttot, input_step, prev_f))
    for i in range(Ttot):
        step += input_step
        for j in range(step-input_step,step):
            for k in range(prev_f):
                a_new[i,j-input_step*i,k] = a_prev[0,j,k]
    return a_new

# Load league information
with open(LEAGUES_PATH, 'r') as file:
    leagues = json.load(file)
scores = []
for league in leagues:
    if league['is_active'] == False:
        continue
    data_path = f"./match_infos/{league['folder_name']}/preprocess_data.csv"
    model_path = f"./models/{league['folder_name']}/label_encoder.joblib"
    E0_data = pd.read_csv(data_path, encoding='utf-8')
    le = joblib.load(model_path)
    encoded_teams = set(le.classes_)
    E0_data["HomeTeam"] = E0_data["HomeTeam"].apply(
            lambda x: le.transform([x])[0] if x in encoded_teams else None
    )
    E0_data["AwayTeam"] = E0_data["AwayTeam"].apply(
        lambda x: le.transform([x])[0] if x in encoded_teams else None
    )
    if league['can_draw']:
        E0_data = E0_data[(E0_data['ODDS1'] != '-') & (E0_data['ODDSX'] != '-') & (E0_data['ODDS2'] != '-')]
        X = E0_data[features1].fillna(0)
    else:
        E0_data = E0_data[(E0_data['ODDS1'] != '-') & (E0_data['ODDS2'] != '-') & (E0_data['FTR'] != 'D')]
        X = E0_data[features2].fillna(0)
    Y = E0_data[['FTR']].to_numpy().ravel()

    #XY preprocessing
    imputer = SimpleImputer()
    X_imputed = imputer.fit_transform(X)
    Y = one_hot_y(Y)

    x_train, x_test, y_train, y_test = train_test_split(X_imputed, Y, shuffle=False, test_size=0.2)

    # x_train = X_imputed
    # y_train = Y

    #Setup XY to have 10 game steps
    x_train = time_step(x_train)
    y_train = time_step(y_train)
    y_train = np.moveaxis(y_train, 0, 1)

    x_test = time_step(x_test)
    y_test = time_step(y_test)
    y_test = np.moveaxis(y_test, 0, 1)

    Tx= x_train.shape[1] #Time steps
    Ty= y_train.shape[0] #Time Steps

    num_features = x_train.shape[2] #Features per step

    # Create and Setup Model
    fbmodel = tf.keras.Sequential()
    inputs = tf.keras.Input(shape=(Tx, num_features))
    outputs = []

    class TimeStepSlice(tf.keras.layers.Layer):
        def __init__(self, t, **kwargs):
            super(TimeStepSlice, self).__init__(**kwargs)
            self.t = t

        def call(self, inputs):
            return inputs[:, self.t, :]

    for t in range(Ty):
        x = TimeStepSlice(t)(inputs)
        x = tf.keras.layers.Reshape((1, num_features))(x)
        x = tf.keras.layers.LSTM(units=64, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-1))(x)
        x = tf.keras.layers.Dropout(rate=0.8)(x)
        out = tf.keras.layers.Dense(3, activation='softmax')(x)
        
        outputs.append(out)
        
    fbmodel = tf.keras.Model(inputs=inputs, outputs=outputs)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    fbmodel.summary()

    fbmodel.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.5)
    )
    #Train Model
    history = fbmodel.fit(
        x = x_train, 
        y = list(y_train),
        epochs=1000,
        batch_size=64,
        validation_split=0.2,
        verbose=2,
        shuffle=False,
        callbacks=[early_stopping]
    )

    fbmodel.save(f"./models/{league['folder_name']}/model.keras")
    # fbmodel = tf.keras.models.load_model("./models/model.keras", custom_objects={'TimeStepSlice': TimeStepSlice})

    # Model Metrics Data Setup
    y_pred_train = fbmodel.predict(x_train)
    y_pred_train = np.asarray(y_pred_train)
    y_predm_train = np.argmax(y_pred_train, axis=2)
    y_trainm = np.argmax(y_train, axis=2)

    # Revert one-hot encoding for the predictions and actual results
    y_predm_train = revert_yoh(y_predm_train).ravel()
    y_trainm = revert_yoh(y_trainm).ravel()

    y_pred_test = fbmodel.predict(x_test)
    y_pred_test = np.asarray(y_pred_test)
    y_pred_confidences = np.max(y_pred_test, axis=2)
    y_predm_test = np.argmax(y_pred_test, axis=2)
    y_testm = np.argmax(y_test, axis=2)

    # Revert one-hot encoding for the predictions and actual results
    y_predm_test = revert_yoh(y_predm_test).ravel()
    y_testm = revert_yoh(y_testm).ravel()

    t_cnt = 0
    cnt = 0
    for pred, test, confidence in zip(y_predm_test, y_testm, y_pred_confidences):
        for p, t, c in zip(pred, test, confidence):
            if c >= 0.4:
                t_cnt += 1
                if p == t:
                    cnt += 1
            print(f"Predicted: {p}, Result: {t}, Confidence: {c * 100:.2f}%")

    print(f'Total matches: {len(x_test)*15}, High matches: {t_cnt}, Correct matches: {cnt}, Accuracy: {cnt/t_cnt if not t_cnt == 0 else 0}')
    score_test = accuracy_score(y_testm, y_predm_test)
    # Model Metrics
    print(f"{league['league_name']} - Train Score: ", accuracy_score(y_trainm, y_predm_train))
    print(f"{league['league_name']} - Test Score: ", score_test)
    scores.append(score_test)
    # Create a figure and axis
    plt.figure(figsize=(12, 6))

    # Plot predictions vs true values
    # plt.plot(y_predm_train, label='Predicted (Training)', color='blue', linestyle='-', marker='o')
    # plt.plot(y_trainm, label='Actual (Training)', color='red', linestyle='-', marker='x')

    plt.plot(y_predm_test, label='Predicted (Testing)', color='blue', linestyle='-', marker='o')
    plt.plot(y_testm, label='Actual (Testing)', color='red', linestyle='-', marker='x')

    # plt.plot(history.history['loss'], label='Loss (Training)', color='blue', linestyle='-', marker='o')
    # plt.plot(history.history['val_loss'], label='Val Loss (Training)', color='red', linestyle='-', marker='x')

    # Add labels and title
    plt.xlabel('Sample Index')
    plt.ylabel('Class Label')
    plt.title('Training Predictions vs Actual Labels')
    plt.legend()

    # Show the plot
    # plt.show()

print(scores)
print(np.array(scores).mean())