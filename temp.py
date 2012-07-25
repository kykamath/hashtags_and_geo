from itertools import combinations

ltuo_valid_iid_and_focus_lid = [[1,'a'], [2,'b'], [3,'c'], [4,'d']]
for (valid_iid1, focus_lid1), (valid_iid2, focus_lid2) in combinations(ltuo_valid_iid_and_focus_lid, 2):
    print focus_lid1, focus_lid2, valid_iid1-valid_iid2
    print focus_lid2, focus_lid1, valid_iid2-valid_iid1
    
val = {}
val[12]= 12
val[2] = 12
for k, v in val.iteritems():
    print v
    