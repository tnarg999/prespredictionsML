__author__ = 'Grant.Hugh'

import keras as k
import numpy as np

# my_data = np.genfromtxt('GlobalElections_MODProject.csv', delimiter=',')
# print(my_data)

file = open('GlobalElections_MODProject.csv', 'r')
totalLines = file.readlines()
header = totalLines[0]
dataLines = totalLines[1:]

allData = []
presData = []
senData = []
houseData = []

myStates = ['Wisconsin', 'Washington', 'Utah', 'Vermont', 'South Dakota', 'South Carolina', 'Pennsylvania', 'Oregon', 'Oklahoma', 'Ohio', 'North Dakota', 'New York', 'North Carolina', 'New Hampshire', 'Nevada', 'Missouri', 'Maryland', 'Louisiana', 'Kentucky', 'Kansas', 'Iowa', 'Indiana', 'Illinois', 'Hawaii', 'Idaho', 'Georgia', 'Florida', 'Connecticut', 'Colorado', 'California', 'Arkansas', 'Arizona', 'Alaska', 'Alabama', 'Wyoming', 'Delaware', 'West Virginia', 'Virginia', 'Tennessee', 'Texas', 'Rhode Island', 'New Mexico', 'New Jersey', 'Nebraska', 'Montana', 'Mississippi', 'Minnesota', 'Massachusetts', 'Michigan', 'Maine']

categoryList = []
lenData = 3;

for line in dataLines:
    splitLine = line.split(',')
    imLine = splitLine[0:8]
    if(imLine[2] not in categoryList):
        categoryList.append(imLine[2])
    for i in range(0,8):
        if imLine[i] == '':
            imLine[i] = 0
    imLine[3] = int(imLine[3]) / 10000;
    imLine[5] = int(imLine[5]) / 10000;
    imLine[7] = int(imLine[7]) / 10000;
    imLine[2] = categoryList.index(imLine[2]) + 1
    allData.append(imLine)

print(categoryList)

for state in allData:
    if(state[1] == '1'):
        houseData.append(state)
    elif(state[1] == '2'):
        senData.append(state)
    elif(state[1] == '5'):
        presData.append(state)

presData1988 = []
presData1992 = []
presData1996 = []
presData2000 = []

def processState(state):
    del state[0:2]
    del state[1:3]
    del state[2]

for state in presData:
    if(state[0] == '1988'):
        processState(state)
        presData1988.append(state)
    if(state[0] == '1992'):
        processState(state)
        presData1992.append(state)
    if(state[0] == '1996'):
        processState(state)
        presData1996.append(state)
    if(state[0] == '2000'):
        processState(state)
        presData2000.append(state)

presData1988.sort(key=lambda x: x[0])
presData1992.sort(key=lambda x: x[0])
presData1996.sort(key=lambda x: x[0])
presData2000.sort(key=lambda x: x[0])

statePresDataX = []
statePresDataY = []
oneStatePresDataX = []
oneStatePresDataY = []
dWin = [1, 0]
rWin = [0, 1]

for ii in range(0, len(presData1988)):
    XList = []
    YList = []
    oneXList = []
    oneYList = []
    XList.append(presData1988[ii])
    XList.append(presData1992[ii])
    XList.append(presData1996[ii])
    YList.append(presData1992[ii])
    YList.append(presData1996[ii])
    YList.append(presData2000[ii])
    statePresDataX.append(XList)
    statePresDataY.append(YList)

    if(presData1988[ii][0] > presData1988[ii][1]):
        oneXList.append(dWin)
    else:
        oneXList.append(rWin)

    if(presData1992[ii][0] > presData1992[ii][1]):
        oneXList.append(dWin)
        oneYList.append(dWin)
    else:
        oneXList.append(rWin)
        oneYList.append(rWin)

    if(presData1996[ii][0] > presData1996[ii][1]):
        oneXList.append(dWin)
        oneYList.append(dWin)
    else:
        oneXList.append(rWin)
        oneYList.append(rWin)

    if(presData2000[ii][0] > presData2000[ii][1]):
        oneYList.append(dWin)
    else:
        oneYList.append(rWin)

    oneStatePresDataX.append(oneXList)
    oneStatePresDataY.append(oneYList)

print(oneStatePresDataX)

# onepresmodel = k.models.Sequential();
# onepresmodel.add(k.layers.LSTM(input_shape=(3, lenData), units=512, activation='relu', return_sequences = True));
# onepresmodel.add(k.layers.LSTM(512, activation='sigmoid', return_sequences = True));
# onepresmodel.add(k.layers.LSTM(512, activation='sigmoid', return_sequences = True));
# onepresmodel.add(k.layers.Dense(lenData, activation='softmax'));
# onepresmodel.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy']);
#
# onepresmodel.fit(np.asarray(oneStatePresDataX).astype(np.float), np.asarray(oneStatePresDataY).astype(np.float), epochs=10, verbose=1);

presmodel = k.models.Sequential();
presmodel.add(k.layers.LSTM(input_shape=(3, lenData), units=512, activation='relu', return_sequences = True));
presmodel.add(k.layers.LSTM(512, activation='relu', return_sequences = True));
presmodel.add(k.layers.LSTM(512, activation='relu', return_sequences = True));
presmodel.add(k.layers.Dense(lenData, activation='linear'));
presmodel.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy']);

presmodel.fit(np.asarray(statePresDataX).astype(np.float), np.asarray(statePresDataY).astype(np.float), batch_size=1, epochs=10, verbose=1);
for ii in range (0, 50):
    # print(statePresDataX[ii])
    # oneprediction = onepresmodel.predict(np.asarray([oneStatePresDataX[ii]]))
    average2000 = [0,0]
    for jj in range(0, 100):
        valprediction = presmodel.predict(np.asarray([statePresDataX[ii]]))

        prediction2000 = valprediction[0][0][1:] * 10000
        prediction2004 = valprediction[0][1][1:] * 10000
        prediction2008 = valprediction[0][2][1:] * 10000
        average2000 = [x + y for x, y in zip(average2000, prediction2004)]

    print([x/100 for x in average2000])
    # print(myStates[ii], 2004, [prediction2004[4], prediction2004[6]])
    # print(myStates[ii], 2008, [prediction2008[4], prediction2008[6]])
    # print(myStates[ii], 2012, [prediction2012[4], prediction2012[6]])






