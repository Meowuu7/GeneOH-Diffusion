import json

def get_sh_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    test_sh_fn = "data/grab/result/test_exported_sh.sh"
    wf = open(test_sh_fn, "w")
    for key in data:
        val = data[key]
        print(key, type(val), val)    
        if isinstance(val, bool):
            if val == True:
                wf.write(f"export {key}=\"--{key}\"\n")
            else:
                wf.write(f"export {key}=\"\"\n")
        elif isinstance(val, str):
            wf.write(f"export {key}=\"{val}\"\n")
        else:
            wf.write(f"export {key}={val}\n")
    
    wf.close()
    
    # sh = data['sh']
    # return sh
    

if __name__=='__main__':
    json_file = "data/grab/result/args_spatial.json"
    get_sh_from_json(json_file=json_file)
