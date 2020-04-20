import json
import csv
file_path= 'C:/Users/amitb/Documents/Ganesha/VIBRAINT Project/'
json_input = file_path+'gsample.json'
csv_output = file_path+'DataFile.csv'
with open(json_input) as json_file:
    data = json.load(json_file)
    with open(csv_output, "w") as csv_file:
        myFields = ['messageId','version','recordId','userId','numberOfTracks','eventType','dataSetId', 'selectionId', 'timezone', 'timestamp', 'messageType', 'eventType']
        writer = csv.DictWriter(csv_file, fieldnames=myFields)    
        writer.writeheader()
        data1 = data['messageData']
        data2 = data1['dataSets']
        messageId = data['messageId']
        version = data1['version']
        recordId = data1['recordId']
        userId = data1['userId']
        numberOfTracks = data1['numberOfTracks']
        eventType = data1['eventType']
        #dataSetId = data2['dataSetId'] 
        selectionId = data1['selectionId']
        timezone = data1['timezone'] 
        timestamp = data1['timestamp']
        messageType = data['messageType'] 
        eventType = data1['eventType']
        writer.writerow({'messageId' : messageId, 'version': version, 'recordId' : recordId, 'userId': userId,'numberOfTracks': numberOfTracks, 'eventType': eventType, 'selectionId': selectionId,'timezone': timezone, 'timestamp': timestamp, 'messageType': messageType, 'eventType': eventType})