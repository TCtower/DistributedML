import json

f = open("data_setting_4.json", )

data = json.load(f)
print(data["data-setting"])


f.close()
