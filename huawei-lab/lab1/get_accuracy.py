from obs import ObsClient
import json
import csv

obsClient = ObsClient(
    access_key_id='S1ERIN0FTHOI77QGUJ1V',    
    secret_access_key='LZ2cJWrzEy2vnWT9DEAo8UCOgUyJeR77ekZ3pV3b',    
    server='https://obs.cn-east-3.myhuaweicloud.com'
)

try:
    resp = obsClient.listObjects('bucket-6576','speechsepecification/output/infer-result-1ae2c7fa-a34a-48a5-af4c-bdbad2f88973') 

    if resp.status < 300: 
        all = 0
        correct = 0
        write_file = csv.DictWriter(open('test.csv','w',newline='',encoding = 'utf-8'),['file_tag','tag','score'])
        write_file.writeheader()
        for info in resp.body['contents']:
            if info['key'].endswith('txt'):
                all += 1
                type = info['key'].split('/')[-1].split('_')[0]
                data = obsClient.getObject('bucket-6576',str(info['key']),loadStreamInMemory=True)
                data = data['body']['buffer'].decode('utf-8')
                data = json.loads(data)
                if type == data['predicted_label']:
                    correct += 1
                write_file.writerow({'file_tag':type,'tag': data['predicted_label'],'score': data["score"]})
        print("accuracy: ", correct / all)
    else: 
        print('errorCode:', resp.errorCode) 
        print('errorMessage:', resp.errorMessage)
except:
    import traceback
    print(traceback.format_exc())

obsClient.close()