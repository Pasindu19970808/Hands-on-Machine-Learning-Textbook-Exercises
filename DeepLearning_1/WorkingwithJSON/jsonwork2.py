import json

#json.load loads a file, while json.loads loads a string

with open("C:\\Users\\ASUS\\Desktop\\Hands on ML\\Hands-on-Machine-Learning-Textbook-Exercises\\DeepLearning_1\\WorkingwithJSON\\states.json") as f:
    data = json.load(f) 

for state in data["states"]:
    del(state["area_codes"])

#dumps dumps to a json string, while dump dumps to a json file
with open("C:\\Users\\ASUS\\Desktop\\Hands on ML\\Hands-on-Machine-Learning-Textbook-Exercises\\DeepLearning_1\\WorkingwithJSON\\new_states.json","w") as f:
    json.dump(data, f, indent = 2)