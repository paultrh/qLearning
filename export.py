import pickle

# read python dict back from the file
qtable = []
with open('qtable.pickle', 'rb') as f:
    qtable = pickle.load(f)


for key, values in qtable.items():
    for action_score in values:
        if action_score == -300:
            print(key, values)
