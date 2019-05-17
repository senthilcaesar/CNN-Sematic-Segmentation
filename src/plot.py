import pandas
import matplotlib.pyplot as plt

f = open('DWIlog_99percentile.csv')
for line in f:
    data_header = line.rstrip().split(';')
    break
    
column_header = data_header
data_pandas = pandas.read_csv('DWIlog_99percentile.csv', sep = ';', names=column_header)
SO_loss = list(map(float, data_pandas.final_op_loss.tolist()[1:]))

L = len(SO_loss)
plt.xlim(0, L - 1)
plt.ylim(0, 1)

X = list(range(0, L))
Y = [abs(x) for x in SO_loss] 

plt.plot(X, Y, marker='o', markerfacecolor='red', color='blue', linestyle=':');
plt.xlabel('No of epochs', size=12)
plt.ylabel('Dice Coefficient', size=12)   
plt.grid(linestyle='dotted')
plt.show()
