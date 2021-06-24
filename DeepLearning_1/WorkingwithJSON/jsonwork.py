import json

people_string = '''{
    "people" : [
        {
            "name" : "John", 
            "phone" : "615-555-7164",
            "emails" : ["abc@yahoo.com","def@yahoo.com"],
            "has_license": false
        },
        {
            "name" : "Jane", 
            "phone" : "615-555-7777",
            "emails" : null,
            "has_license": true
        }
    ]
}
'''

#loading to json format
#region JSON to PYTHON translations
'''
When we load a string using json.loads, the following translations occur between JSON and Python
(JSON -> Python)
object : dict
array : list
string : str
number(int) : int
number(real) : float
true : True
false: False
null : None
'''
#endregion


data = json.loads(people_string)

#for  person in data["people"]:
#    print(person)

#converting from python object to json string 

for person in data["people"]:
    del person["phone"]

new_string = json.dumps(data, indent = 2)

print(new_string)


#How to load kson files to python objects