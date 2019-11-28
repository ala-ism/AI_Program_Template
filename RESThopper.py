#%%
#import packages
import requests
import rhino3dm
import base64
import compute_rhino3d.Util
import json

#%%
#Set URL and Auth
compute_rhino3d.Util.url="http://127.0.0.1:8081/"
compute_rhino3d.Util.authToken="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJwIjoiUEtDUyM3IiwiYyI6IkFFU18yNTZfQ0JDIiwiYjY0aXYiOiJaSU8zQ1FOMlhvSVBJay9Mcnoya3B3PT0iLCJiNjRjdCI6ImRyWjV1MG5yMHRjVkhHMTlHWmFJOUFTbDdScnZoMEVwMldHelJydTB3VWhnQzdneHVwNFpxV1lzZGkxbkNYSWJhZ1psYmxRQmRLVUZ6RlplRU1ra2dELytkTFppdFNJWDNKN1kxMnpoM1orR3BMSVIwTHM3UGU1NXkyMDQwL0wyVnJnNVJSK1JPbndGaWJYemFQQ3Mrekh6dit6aTJnNCtNcWljVmltaFhFOGRpb2RZUFhHemhXc0h2KzhXQ2NTeWpZN29QMDZsV0o3OFZFY0V1ZHlYWHc9PSIsImlhdCI6MTU3NDA5MDEyN30.i8XbIRs0fEC4H7dMB2SVurm1IL3s5olF_NQeTg5SZ3I"

#%%
#Create URL for POST request
POST_URL=compute_rhino3d.Util.url+"grasshopper"
gh_data=open("RESTHopper_test.ghx", mode="r", encoding="utf-8-sig").read()
print("gh_data= ",gh_data)

#%%
#Encoding 1/3
data_bytes=gh_data.encode("utf-8")
print("databytes= ",data_bytes)
type(data_bytes)
#%%
#Encoding 2/3
encoded=base64.b64encode(data_bytes)
print(encoded)
type(encoded)
#%%
#Encoding 3/3
decoded=encoded.decode("utf-8")
print(decoded)
type(decoded)
#%%
#Create a json file for parameters used as input 
jsonfile={
    "algo":decoded, 
    "pointer":None, 
    "values":[
                {"ParamName":"RH_IN:Count","InnerTree":{"{0; }":[{"type":"System.Integer", "data":"34" }]}},
                {"ParamName":"RH_IN:Size","InnerTree":{"{0; }":[{"type":"System.Double", "data":"47.0"}]}},
                {"ParamName":"RH_IN:Outer","InnerTree":{"{0; }":[{"type":"System.Double", "data":"27.0"}]}},
                {"ParamName":"RH_IN:Inner","InnerTree":{"{0; }":[{"type":"System.Double", "data":"8.0"}]}}
            ]
        }
#%%
#Send a POST request through REST API and running some sanity checks
response= requests.post(POST_URL, json=jsonfile)
print("response= ", response)
print("response.content= ", response.content)
res=response.content.decode("utf-8")
res=json.loads(res)
print("res= ", res)
values=res['values']

#%%
#Creating a Rhino output file
model=rhino3dm.File3dm()
#%%
#Getting the output elements
for val in values:
    paramName=val["ParamName"]
    print ('paramName= ', paramName)
    InnerTree=val['InnerTree']
    for key, innerVals in InnerTree.items():
        print('key= ', key)
        print("InnverVals=", innerVals)
        for innerVal in innerVals:
            data=json.loads(innerVal["data"])
            print("data=", data)
            geo=rhino3dm.CommonObject.Decode(data)
            print("geo= ",geo)
            model.Objects.Add(geo)
#%%
#Putting the output in a Rhino file
model.Write("output_AI.3dm")