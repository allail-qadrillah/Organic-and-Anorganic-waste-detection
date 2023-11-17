organik = ['AMPAS TEBU', 'RANTING KAYU', 'DAUN', 'KULIT TELUR']
detections = ['KULIT TELUR']

for detection in detections:
    if detection in organik:
        print('ROTATE SERVO')