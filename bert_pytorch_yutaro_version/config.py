from attrdict import AttrDict
import json

# set configuration file
config_file = "./weights/bert_config.json"

# get configuration of file
json_file = open(config_file, "r")
config = json.load(json_file)

# set dictionary variable to object variable
config = AttrDict(config)
