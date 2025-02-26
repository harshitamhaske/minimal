import pickle

file_path = ''

with open(file_path, 'rb') as f:
    policy = pickle.load(f, encoding='latin1')

if isinstance(policy, dict):
    for key, value in policy.items():
        print(f"{key}:")
        if isinstance(value, list):
            for array in value:
                print(array) 
        else:
            print(value)  
else:
    if isinstance(policy, list):
        for array in policy:
            print(array)  
    else:
        print(policy) 
