import csv
import re

# Your provided data as a string
data = "51615,3,6632,(SpineBase,Tracked,0.1407465,-0.01618211,2.642958,277.981,209.2025),(SpineMid,Tracked,0.1439389,0.3106374,2.622563,278.6025,163.4418),(Neck,Tracked,0.1460119,0.6271127,2.589385,279.2148,117.7276),(Head,Tracked,0.1527646,0.7744268,2.579439,280.2905,96.18451),(ShoulderLeft,Tracked,-0.02793079,0.5089267,2.602411,254.4908,135.0054),(ElbowLeft,Tracked,-0.0806906,0.2667878,2.630191,247.1744,169.7095),(WristLeft,Tracked,-0.1039654,0.02940379,2.608654,243.8165,202.8201),(HandLeft,Tracked,-0.09606399,-0.01941682,2.603497,244.9015,209.6922),(ShoulderRight,Tracked,0.3167299,0.4868576,2.554402,304.0968,136.7744),(ElbowRight,Tracked,0.3628582,0.2413872,2.567245,310.4086,172.384),(WristRight,Tracked,0.3858157,0.004776413,2.5557,313.9223,206.2689),(HandRight,Tracked,0.382229,-0.05734983,2.545668,313.6241,215.2357),(HipLeft,Tracked,0.06238811,-0.01422033,2.611025,267.2059,208.9539),(KneeLeft,Tracked,0.01087089,-0.3142282,2.683355,259.9275,249.9646),(AnkleLeft,Tracked,-0.01424084,-0.6091588,2.712693,256.5067,289.6372),(FootLeft,Tracked,-0.03621031,-0.67559,2.622143,253.3506,301.9038),(HipRight,Tracked,0.2149325,-0.0176756,2.597917,288.8086,209.4533),(KneeRight,Tracked,0.278372,-0.3420982,2.653026,297.0161,254.3634),(AnkleRight,Tracked,0.3131567,-0.6173937,2.701226,301.1605,291.1808),(FootRight,Tracked,0.3379418,-0.6711916,2.607368,306.2398,301.8926),(SpineShoulder,Tracked,0.1456582,0.5494619,2.599994,279.0636,129.1566),(HandTipLeft,Tracked,-0.0850715,-0.09645323,2.613563,246.4958,220.4976),(ThumbLeft,Tracked,-0.06424066,0.01071023,2.573286,249.2806,205.4288),(HandTipRight,Tracked,0.3846655,-0.1263849,2.531365,314.3,225.3091),(ThumbRight,Tracked,0.3427438,-0.06380382,2.546667,307.8882,216.161)"

# Extracting the first three numbers
first_three = data.split('(')[0].strip().split(',')

# Extracting the data within parentheses
matches = re.findall(r'\((.*?)\)', data)

# Writing to CSV
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write header
    writer.writerow(['ID1', 'ID2', 'ID3', 'Joint', 'TrackingState', 'X', 'Y', 'Z', 'X2', 'Y2'])

    # Write the first three IDs
    for match in matches:
        row = first_three + match.split(',')
        writer.writerow(row)