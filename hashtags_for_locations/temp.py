#import matplotlib.pyplot as plt
#
#def f(x): 
#    return_list = []
#    previous = 0
#    for i in x:
#        previous+=i
#        return_list.append(previous)
#    return return_list
#
#def f1(x): 
#    return_list = []
#    previous = 1
#    beta = 0.5
#    for i in x:
#        previous*=(beta**i)
#        return_list.append(previous)
#    return return_list
#
#def temp():
#    data_points = 1000.
#    x = [i/data_points for i in range(int(data_points))]
#    x.reverse()
#    y = f1(x)
#    plt.semilogx(range(len(x)),y)
#    plt.show()
#    
#temp()

#from library.file_io import FileIO
#f_name = '/mnt/chevron/kykamath/data_from_dfs/geo/hashtags//2011-09-01_2011-11-01/360_120/100/linear_regression'
#for data in FileIO.iterateJsonFromFile(f_name, remove_params_dict=True):
#    print data.keys()

import cjson
data = '[["faze3",0.285714285714],["terriblenamesforavagina",0.285714285714],["cgi2011",0.142857142857],' + \
                                        '["dudesthatsaynohomo",0.142857142857],["takewallstreet",0.142857142857]]'
mf_hashtag_to_value = dict(cjson.decode(data))
def accuracy(hashtags, mf_hashtag_to_value): return set(hashtags).intersection(set(mf_hashtag_to_value.keys()))/5.
def impact(hashtags, mf_hashtag_to_value): 
    total = sum(mf_hashtag_to_value.values())
    val = sum([mf_hashtag_to_value.get(h, 0.0) for h in hashtags])
    return total/val
print mf_hashtag_to_value
print sum(mf_hashtag_to_value.values())