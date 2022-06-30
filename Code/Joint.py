#--*--coding:utf-8--*--
from pulp import *
import math
def distance(x,y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

def distance_hm(x,y):
    if x[0]==y[0] and x[1]==y[1]:return 0
    return 1


def min_distance(x,loc_set):
    minD=80
    for loc in loc_set:
        minD=min(minD,distance(loc,x))
    return minD

def pulp_joint(dm,epsilon,eps_g,dataset,loc_set,fir_pro):

    d_eps=epsilon/eps_g


    model = LpProblem("joint", LpMinimize)

    n=len(loc_set)
    # dm=0
    # for i in range(n):
    #     dm+=err[i]*fir_pro[i]

    # for i in range(n):
    #     print(i, min_distance(loc_set[i], loc_set))
    # dm=8.24
    # dm=2.39
    # print(dm)
    pro_min=0
    pro_max=1

    p = pulp.LpVariable.dicts("pro", (range(n), range(n)), lowBound = pro_min,upBound=pro_max,cat=LpContinuous)
    xo = pulp.LpVariable.dicts("x(o)", range(n), lowBound=0, upBound=50,cat=LpContinuous)



    # model += lpSum([[fir_pro[s] * p[s][o]*distance(loc_set[s],loc_set[o]) for s in range(n)] for o in range(n)]), "总效用损失"
    model += lpSum(
        fir_pro[s] * p[s][o] * distance_hm(loc_set[s], loc_set[o]) for o in range(n) for s in range(n)), "总效用损失"


    for i in range(n):
        model += lpSum([p[i][item] for item in range(n)]) == 1


    for o in range(n):
        for s in range(n):
            for ss in range(n):
                if distance(loc_set[s],loc_set[ss])<=d_eps:
                    model += p[s][o]-math.exp(eps_g*distance(loc_set[s],loc_set[ss]))*p[ss][o] <= 0


    for o in range(n):
        for ss in range(n):
            model += lpSum([fir_pro[s] * p[s][o] * distance(loc_set[s], loc_set[ss]) for s in range(n)]) - xo[o] >= 0
    # for o in range(n):
        # model += xo[o] <= 80
        # model += xo[o] >= 0


    model += lpSum([xo[o] for o in range(n)]) >= dm
    # model += lpSum([[fir_pro[s]*p[s][o]*min_distance(loc_set[s],loc_set) for s in range(n)] for o in range(n)])>=dm,"误差约束"


    model.solve()




    # print(model)
    print(":", LpStatus[model.status])
    i=0
    array=[[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            array[i][j]=p[i][j].value()
    print("pro=",array)
    xo_array=[]
    for i in range(n):
        xo_array.append(xo[i].value())
    print("xo=",xo_array)
    # for arr in array:
    #     print(arr)
        # print(v.name, "=", v.varValue)
    print("Qloss ", value(model.objective))
    return array

