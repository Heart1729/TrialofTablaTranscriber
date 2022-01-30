import flask
from tkinter import filedialog
import numpy as np
import pandas as pd
import csv
import librosa
import tensorflow as tf
print('library imported')
model = tf.keras.models.load_model('TTM_model.h5')
model.summary()

def predict_audio(audio):
    header = 'ChromaSTFT RMS SpectralCentroid SpectralBandwidth Rolloff ZeroCrossingRate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    file = open('predict_file.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    taalfile = audio
    #print('stored in taalfile')
    y, sr = librosa.load(taalfile, mono=True, duration=30)
    rms = librosa.feature.rms(y=y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f' {np.mean(chroma)} {np.mean(rms)} {np.mean(spec_centroid)} {np.mean(spec_bandwidth)} {np.mean(rolloff)} {np.mean(zcr)} '    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    #to_append += f' {t}'
    file = open('predict_file.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
    predict_file = pd.read_csv("predict_file.csv")
    X_predict = predict_file.drop('label', axis=1)
    prediction = 0
    prediction_val = 0
    taals = ['addhatrital','bhajani','dadra','deepchandi','ektal','jhaptal','rupak','trital']
    pred = model.predict(X_predict)
    for i in range(0, 8):
      if (pred[0][i]*100>prediction):
        prediction = pred[0][i]*100
        prediction_val = i
    print('This audio sounds to be ',prediction,'% belonging to ',taals[prediction_val],' taal!')
    return str(round(prediction, 2)),taals[prediction_val].capitalize()



app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET','POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        audio = flask.request.form['audio']
        print(type(audio))
        prediction, prediction_name = predict_audio(audio)
        return flask.render_template('main.html',
                                     prediction="This music sounds "+prediction+"% like:",
                                     prediction_name=prediction_name,
                                     audio=audio
                                     )
        
    #return(flask.render_template('main.html'))


if __name__ == '__main__':
    app.run()
