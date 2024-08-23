import pandas as pd
import matplotlib.pyplot as plt

rel_Cols = [[' Step Length(cm) L', ' Step Length(cm) R'], [' Stride Length(cm) L', ' Stride Length(cm) R'],
             [' Step Time(sec) L', ' Step Time(sec) R'], [' Double Supp % Cycle R', ' Double Supp % Cycle L'],
             [' Swing Time(sec) L', ' Swing Time(sec) R']]
filePaths = ['/Users/lukakoll/Desktop/SimCenterData/Parkinsonian(luka1).txt', '/Users/lukakoll/Desktop/SimCenterData/Frailty(Dr. Johnson1).txt',
             '/Users/lukakoll/Desktop/SimCenterData/Frailty(Dr. Iannone1).txt', '/Users/lukakoll/Desktop/SimCenterData/Hemiplegic(Sriharsha1).txt',
             '/Users/lukakoll/Desktop/SimCenterData/Hemiplegic(Basam1).txt', '/Users/lukakoll/Desktop/SimCenterData/Normal(Basam1).txt', 
             '/Users/lukakoll/Desktop/SimCenterData/Normal(Andrew1).txt', '/Users/lukakoll/Desktop/SimCenterData/Normal(Elizabeth1).txt'
             ]
for [attribute1, attribute2] in rel_Cols:
    data1 = {}
    data2 = {}
    
    for i in range(0, len(filePaths)):
        df = pd.read_csv(filePaths[i])
        data1[filePaths[i].rstrip(".txt").lstrip("Users/lukakoll/Desktop/SimCenterData/")] = df.loc[0, attribute1]
        print(f'{attribute1}: {df.loc[0, attribute1]}')

    for i in range(0, len(filePaths)):
        df = pd.read_csv(filePaths[i])
        data2[filePaths[i].rstrip(".txt").lstrip("Users/lukakoll/Desktop/SimCenterData/")] = df.loc[0, attribute2]
        print(f'{attribute2}: {df.loc[0, attribute2]}')
    
    categories = list(data1.keys())
    values1 = list(data1.values())
    values2 = list(data2.values())

    positions = range(len(categories))

    plt.figure(figsize=(20, 8))

    plt.bar(positions, values1, label=attribute1)

    plt.bar(positions, values2, bottom=values1, label=attribute2)
    plt.title(f'{attribute1.strip()} and {attribute2.strip()} comparison', fontsize=14)
    plt.xlabel("condition", fontsize=12)
    plt.xticks(range(len(data1)), list(data1.keys()), fontsize=10)

    ax = plt.gca()  # Get the current axis
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 2 == 0:
            label.set_y(label.get_position()[1] - 0.05)

    plt.legend()
    plt.show()
    plt.waitforbuttonpress  

    


