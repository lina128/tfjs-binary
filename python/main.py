import subprocess

mystr = b'1,2,3'
cmd = "../index"
url = "https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/model.json"
dummy_inputs = "0"
subprocess.run([cmd, url, dummy_inputs], input=mystr)