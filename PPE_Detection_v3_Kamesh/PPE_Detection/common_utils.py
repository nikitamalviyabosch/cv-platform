import json

class utilities:
    def load_json(self, json_path):
        with open(json_path) as json_file:
            config_file = json.load(json_file)
        return config_file
    
    def getConfigParam(self,strKey, dict):
            try:
                if strKey in dict.keys():
                    return dict[strKey]
                else:
                    return ''
            except:
                print(str(strKey) + "not available")

    def read_classes_file(self,file_path):
        with open(file_path, 'r') as file:
            # Read lines from the file and strip any extra whitespace
            class_list = [line.strip() for line in file]
        return class_list