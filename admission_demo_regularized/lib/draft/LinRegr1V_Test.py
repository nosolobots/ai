import matplotlib.pyplot as plt

def comp_parcial_tetha0(tetha0, tetha1, test_data_x, test_data_y):
    sum = 0.0
    for i in range(len(test_data_x)):
        sum += (tetha0 + tetha1*test_data_x[i]) - test_data_y[i]

    return sum

def comp_parcial_tetha1(tetha0, tetha1, test_data_x, test_data_y):    
    sum = 0.0

    for i in range(len(test_data_x)):
        sum += ((tetha0 + tetha1*test_data_x[i]) - test_data_y[i])*test_data_x[i]

    return sum

def comp_MCE(tetha0, tetha1, test_data_x, test_data_y):    
    sum = 0.0

    for i in range(len(test_data_x)):
        sum += ((tetha0 + tetha1*test_data_x[i]) - test_data_y[i])**2

    return sum/len(test_data_x)    

test_data_x = [1.5, 2, 2.5, 3, 3.5, 4]
test_data_y = [1.5, 2, 2.5, 3, 3.5, 4]

alpha = 0.01
epsilon = 1E-7

tetha0 = 0
tetha1 = 0

while True:
    temp0 = tetha0 - (alpha/len(test_data_x))*comp_parcial_tetha0(tetha0, tetha1, test_data_x, test_data_y)
    temp1 = tetha1 - (alpha/len(test_data_x))*comp_parcial_tetha1(tetha0, tetha1, test_data_x, test_data_y)
    if abs(tetha0-temp0)<epsilon and abs(tetha1-temp1)<epsilon:
        break
    tetha0 = temp0
    tetha1 = temp1

print("\n---> tetha0:", tetha0, ", tetha1:", tetha1)
print("\n---> MCE =", comp_MCE(tetha0, tetha1, test_data_x, test_data_y))
x = []
h = []
for i in range(6):
    x.append(i)
    h.append(tetha0 + tetha1*i)

plt.plot(test_data_x, test_data_y, 'ro', x, h, 'b')
plt.axis([0, 5, 0, 5])
plt.show()