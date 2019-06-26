import json
import traceback


def open_config(config_path,keylist):
    send = {}
    try:
        data = json.load(open(config_path,'r'))
        send = [data[each] for each in keylist if each in data]
        return send
    except Exception as e:
        print("Error at Utils-->open_config {}".format(traceback.format_exc()))
