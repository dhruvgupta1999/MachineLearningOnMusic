import numpy as np
import glob
import csv
import librosa
#the final code for creating labelled features and writing to csv file 


def rmse(arr,axis):
    return np.sqrt(np.mean(np.square(arr),axis=axis))
    



songlist=list(glob.iglob('./beatles110mb/mp3s-32k/**/*.mp3', recursive=True))
# songlist=list(glob.iglob('./beatles110mb/mp3s-32k/Rubber_Soul/04-Nowhere_Man.mp3', recursive=True))

for audio_path in songlist[0:1]:
    print(audio_path)
    chord_file_path=list(glob.iglob('./beatles110mb/chordlabs/' + audio_path[23:-4]+'.lab', recursive=True))
    print(chord_file_path)
    y, sr = librosa.load(audio_path)    
    # y_harmonic, y_percussive = librosa.effects.hpss(y)
    Chroma_feature = librosa.feature.chroma_cqt(y=y, sr=sr) #feature for ml model
    # print(Chroma_feature)
    # print("dimensions of chroma_feature",Chroma_feature.shape)
#     tempo,beats=librosa.beat.beat_track(y_percussive,sr=sr,trim=True)
#     beats_wrt_time=librosa.frames_to_time(beats, sr=sr)
    with open(chord_file_path[0]) as tmp:  #taking the first string cuz right now it is a list
        lab_reader = csv.reader(tmp, delimiter=" ")
        chordlab_list=list(zip(*lab_reader))
        tmp=[float(i) if float(i)>=0.0 else 0.001 for i in chordlab_list[0]] #reading the first column(time column)
        tmp.append(float(chordlab_list[1][-1]))
        
        # print("tmp.shape",tmp.shape)
        
        
        chord_change_time_list=np.array(tmp)
#         print(chord_change_time_list.shape)
        chord_change_time_list=librosa.time_to_frames(chord_change_time_list,sr=sr)# util.sync works on beats as frames
        C_sync = librosa.util.sync(Chroma_feature,chord_change_time_list, aggregate=rmse) #Everything is now synced with the beats
        print("C_sync shape is:",C_sync.shape)
#         chordlab_list[1] #the labels
       
        label_as_string=[str(i) for i in chordlab_list[2]]
        # print("tmp is:",tmp)
        # print("temp is:",temp)
        # listing=['N','C','C#','D','D#','E','F','F#','G','G#','A','A#','B','C:min','C#:min','D:min','D#:min','E:min','F:min','F#:min','G:min','G#:min','A:min','A#:min','B:min']
        list_feature_and_label=[]
        listing=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

        final_data_list=[]
        count=-1
        for i in label_as_string:
            count+=1
            if i in listing and count<C_sync.shape[1]:
                print(i)
                for j in range(0,12):
                    list_feature_and_label.append(C_sync[j][count])
                list_feature_and_label.append(float(listing.index(i)+1))
                final_data_list.append(list_feature_and_label)
            list_feature_and_label=[]
        # count=0
#         print(alll)
#         print(np.array(alll).shape)
#         alll=np.array(alll)
#         np.savetxt("all.csv", alll,delimiter=",")
        with open("major_only_beatle_set(1).csv", "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerows(final_data_list)

