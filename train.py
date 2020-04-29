from train_models import train_models
emotions = ['anger','anticipation','disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust']
train_models('2018-E-c-En-train.txt',emotions,'./GoogleNews-vectors-negative300.bin','./models/')