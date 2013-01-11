import matplotlib.pyplot as plt

def f(x): 
    return_list = []
    previous = 0
    for i in x:
        previous+=i
        return_list.append(previous)
    return return_list

def f1(x): 
    return_list = []
    previous = 1
    beta = 0.5
    for i in x:
        previous*=(beta**i)
        return_list.append(previous)
    return return_list

def temp():
    data_points = 1000.
    x = [i/data_points for i in range(int(data_points))]
    x.reverse()
    y = f1(x)
    plt.semilogx(range(len(x)),y)
    plt.show()
    
temp()