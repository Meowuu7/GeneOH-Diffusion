import json

def get_sh_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    # test_sh_fn = "data/grab/result/test_exported_sh.sh"
    # test_sh_fn = "scripts/val_examples/predict_grab_rndseed_14_bak.sh"
    # test_sh_fn = "scripts/val_examples/predict_grab_rndseed_14_test.sh"
    # test_sh_fn = "scripts/val_examples/predict_hoi4d_rndseed_toycar_inst3.sh"
    test_sh_fn = "scripts/val_examples/predict_hoi4d_rndseed_toycar_inst3_spatial.sh"
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
    

# args ho
if __name__=='__main__':
    json_file = "data/grab/result/args_jts_only.json"
    json_file = "/data1/xueyi/eval_save/HOI_Rigid/ToyCar/args_hoi4d.json"
    json_file = "/data1/xueyi/eval_save/HOI_Rigid/ToyCar/args_hoi4d_spatial.json"
    get_sh_from_json(json_file=json_file)
