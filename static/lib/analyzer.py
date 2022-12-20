import cv2, librosa, myprosody, nltk, subprocess, joblib, re, string, pickle, dlib, imutils
import speech_recognition as sr
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import pandas as pd
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from static.lib.gaze_tracking import GazeTracking
from fer import FER
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv


class Analyzer:
    def __init__(self, with_file=False, app_root="/"):
        self.model = None
        self.app_root = app_root
        command = "ffmpeg -y -i static/clip.webm -ab 160k -ac 2 -ar 44100 -vn static/myprosody_old/myprosody/dataset/audioFiles/audio.wav"
        subprocess.call(command, shell=True)
        self.verdict = "unidentified"
        recognizer = sr.Recognizer()
        try:
            audioFile = sr.AudioFile("static/myprosody_old/myprosody/dataset/audioFiles/audio.wav")
        except:
            self.verdict("failed to transcript audio because audio is not clear")
        with audioFile as source:
            data = recognizer.record(source)
        self.transcript = recognizer.recognize_google(data, key=None)
        self.features = pd.DataFrame(columns=["gazeCenter", "gazeNotCenter", "emotion", "rate_of_speech", "articulation_rate", "balance", "f0_mean", "f0_std", "f0_median", "f0_min", "f0_max", "f0_quantile25", "f0_quan75", "sentiment"])
        self.features.loc[0] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        self.transcript_classifier = pickle.load(open("static/model/sentiment.sav", "rb"))

        custom_tokens = self.remove_noise(word_tokenize(self.transcript))

        self.sentiment = self.transcript_classifier.classify(dict([token, True] for token in custom_tokens))
        self.features.at[0, 'sentiment'] = 0 if self.sentiment == "Negative" else 1

        try:
            self.prosodies = myprosody.mysptotal("audio", r""+self.app_root+"/myprosody_old/myprosody")
        except:
            self.verdict = "failed to extract prosodic features because audio is not clear"
        
        if self.prosodies == None:
            return

        signal, sample_rate = librosa.load("static/myprosody_old/myprosody/dataset/audioFiles/" + "audio.wav", sr=44100)

        rmss = librosa.feature.rms(y=signal)
        dfm = pd.DataFrame(np.transpose(rmss), columns=["rms"])
        summary = dfm.describe()
        pdz = {}
        for index, row in summary.iterrows():
            if index != "count":
                pdz[index+"_"+"rms"] = row["rms"]
        self.rms = pd.DataFrame(pdz, index=[0,])

        MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=2048, n_mfcc=13)
        dfm = pd.DataFrame(np.transpose(MFCCs), columns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"])
        summary = dfm.describe()
        columns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]
        pdz = {}
        for index, row in summary.iterrows():
            if index != "count":
                for col in columns:
                    pdz[index+"_"+col] = row[col]
        self.mfcc = pd.DataFrame(pdz, index=[0,])

        self.features = pd.concat([self.features, self.mfcc], axis=1)
        self.features['blink_rate'] = 0
        self.features['average_pause_duration'] = 0

        self.features.at[0, "rate_of_speech"] = self.prosodies.iloc[0]["rate_of_speech"]
        self.features.at[0, "articulation_rate"] = self.prosodies.iloc[0]["articulation_rate"]
        self.features.at[0, "balance"] = self.prosodies.iloc[0]["balance"]
        self.features.at[0, "f0_mean"] = self.prosodies.iloc[0]["f0_mean"]
        self.features.at[0, "f0_std"] = self.prosodies.iloc[0]["f0_std"]
        self.features.at[0, "f0_median"] = self.prosodies.iloc[0]["f0_median"]
        self.features.at[0, "f0_min"] = self.prosodies.iloc[0]["f0_min"]
        self.features.at[0, "f0_max"] = self.prosodies.iloc[0]["f0_max"]
        self.features.at[0, "f0_quantile25"] = self.prosodies.iloc[0]["f0_quantile25"]
        self.features.at[0, "f0_quan75"] = self.prosodies.iloc[0]["f0_quan75"]
        if float(self.prosodies.iloc[0]["number_of_pauses"]) > 0:
            self.features.at[0, "average_pause_duration"] = (float(self.prosodies.iloc[0]["original_duration"])-float(self.prosodies.iloc[0]["speaking_duration"]))/float(self.prosodies.iloc[0]["number_of_pauses"])

        self.get_visual_features()
        self.features.at[0, "blink_rate"] = self.blink_rate
        self.features.at[0, "gazeCenter"] = self.gaze_center
        self.features.at[0, "gazeNotCenter"] = self.gaze_not_center
        self.features.at[0, "emotion"] = self.emotion

    def remove_noise(self, tweet_tokens, stop_words = ()):

        cleaned_tokens = []

        for token, tag in pos_tag(tweet_tokens):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            token = re.sub("(@[A-Za-z0-9_]+)","", token)

            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        return cleaned_tokens

    def get_all_words(self, cleaned_tokens_list):
        for tokens in cleaned_tokens_list:
            for token in tokens:
                yield token

    def get_tweets_for_model(self, cleaned_tokens_list):
        for tweet_tokens in cleaned_tokens_list:
            yield dict([token, True] for token in tweet_tokens)
 
    # defining a function to calculate the EAR
    def calculate_EAR(self, eye):
    
        # calculate the vertical distances
        y1 = dist.euclidean(eye[1], eye[5])
        y2 = dist.euclidean(eye[2], eye[4])
    
        # calculate the horizontal distance
        x1 = dist.euclidean(eye[0], eye[3])
    
        # calculate the EAR
        EAR = (y1+y2) / x1
        return EAR

    def get_visual_features(self):
        self.emotion = 0
        self.gaze_center = 0
        self.gaze_not_center = 0
        gaze = GazeTracking()
        emotion_detector = FER(mtcnn=True)
        cam = cv2.VideoCapture("static/clip.webm")
        # count the number of frames
        frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cam.get(cv2.CAP_PROP_FPS)
        
        # calculate duration of the video
        seconds = round(frames / fps)
        self.screen_time = seconds
         # Variables
        blink_thresh = 0.45
        succ_frame = 2
        count_frame = 0
        
        # Eye landmarks
        (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        
        # Initializing the Models for Landmark and
        # face Detection
        detector = dlib.get_frontal_face_detector()
        landmark_predict = dlib.shape_predictor(
            'static/model/shape_predictor_68_face_landmarks.dat')
        blink_count = 0
        val = 0
        count = 0
        while 1:
        
            # If the video is finished then reset it
            # to the start
            if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(
                    cv2.CAP_PROP_FRAME_COUNT):
                # cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # blink_count = 0
                break
        
            else:
                is_okay, frame = cam.read()
                if not is_okay:
                    break
                frame = imutils.resize(frame, width=640)
                count = count + 1
                if count < 10 and count > 0:
                    continue
                count = 0
                if self.emotion == 0:
                    dominant_emotion, emotion_score = emotion_detector.top_emotion(frame)
                    if dominant_emotion != "neutral":
                        self.emotion = 1

                gaze.refresh(frame)

                frame = gaze.annotated_frame()

                if gaze.is_blinking() or gaze.is_center():
                    self.gaze_center = 1
                elif gaze.is_right() or gaze.is_left():
                    self.gaze_not_center = 1
        
                # converting frame to gray scale to
                # pass to detector
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
                # detecting the faces
                faces = detector(img_gray)
                for face in faces:
        
                    # landmark detection
                    shape = landmark_predict(img_gray, face)
        
                    # converting the shape class directly
                    # to a list of (x,y) coordinates
                    shape = face_utils.shape_to_np(shape)
        
                    # parsing the landmarks list to extract
                    # lefteye and righteye landmarks--#
                    lefteye = shape[L_start: L_end]
                    righteye = shape[R_start:R_end]
        
                    # Calculate the EAR
                    left_EAR = self.calculate_EAR(lefteye)
                    right_EAR = self.calculate_EAR(righteye)
        
                    # Avg of left and right eye EAR
                    avg = (left_EAR+right_EAR)/2
                    if avg < blink_thresh:
                        val = 1
                        # blink_count += 1
                        count_frame += 1  # incrementing the frame count
                    else:
                        if count_frame >= succ_frame:
                            blink_count += val
                            val = 0
                            cv2.putText(frame, 'Blink Detected ' + str(blink_count), (30, 30),
                                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                        else:
                            count_frame = 0
        
                #cv2.imshow("Video", frame)
                #if cv2.waitKey(5) & 0xFF == ord('q'):
                #    break
        
        cam.release()
        self.blink_count = blink_count
        # result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
        #                      "format=duration", "-of",
        #                      "default=noprint_wrappers=1:nokey=1", "static/clip.webm"],
        # stdout=subprocess.PIPE,
        # stderr=subprocess.STDOUT)
        # self.screen_time = float(result.stdout)
        self.blink_rate = 0
        if self.screen_time > 0:
            self.blink_rate = self.blink_count/(self.screen_time/60)
        #cv2.destroyAllWindows()

    def load_model(self, model="elm"):
        self.model = model
        if model=="elm":
            with open('static/model/elm.npy', 'rb') as f:
                self.input_weights = np.load(f)
                self.output_weights = np.load(f)
                self.biases = np.load(f)
            self.scaler = joblib.load("static/model/elmscaler.sav")
        else:
            self.classifier = pickle.load(open("static/model/" + model + ".sav", 'rb'))

    def predict(self):
        if self.model == None:
            self.load_model(model="mlpoptimized")
        if self.model == "elm":
            features = self.scaler.transform(self.features)
            out = self.hidden_nodes(features, self.input_weights)
            out = np.dot(out, self.output_weights)
            newPrediction = pd.DataFrame(data=out[0:,0:],index=[out[0:,0]])
            newPrediction.reset_index(drop=True, inplace=True)
            newPrediction.drop(0, axis=1, inplace=True)
            newPrediction = newPrediction.round(0).astype(int)
            if newPrediction.iloc[0, 0] > 1:
                newPrediction.at[0, 1] = 1
            if newPrediction.iloc[0, 0] < 0:
                newPrediction.at[0, 1] = 0
            if str(newPrediction.iloc[0, 0]) == "1":
                self.verdict = "truth"
            else:
                self.verdict = "lie"
        else:
            visual = self.features.loc[:, ["gazeCenter", "gazeNotCenter", "emotion", "blink_rate"]]
            lexical = self.features.loc[:, ["sentiment"]]
            prediction = self.classifier.predict(pd.concat([self.mfcc, visual, lexical], axis=1))
            if str(prediction[0]) == "1":
                self.verdict = "truth"
            else:
                self.verdict = "lie"
        return self.verdict

    def relu(self, x):
        return np.maximum(x, 0, x)

    def hidden_nodes(self, X, input_weights):
        G = np.dot(X, input_weights)
        G = G + self.biases
        H = self.relu(G)
        return H
        