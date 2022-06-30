# /usr/bin/env python
# --*--coding:utf-8--*--
# from __future__ import division
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import Domain_Partition as qu
import SC1 as SC1
import copy
import SC2 as SC2
from DPIVE import *
fir_pro = [0.0043335808056442675, 0.004997436325052889, 0.004482487708706725, 0.00593378702413151, 0.004787080440321329, 0.004732256498500137, 0.005897219331343113, 0.004100288292199329, 0.004033062306351729, 0.004942885702298939, 0.005698611147169671, 0.00433605333033881, 0.005816448561766392, 0.004838966665201451, 0.005820275189591764, 0.005294310232056265, 0.005249544513468861, 0.004345862947450193, 0.005671800860211339, 0.005120696242003786, 0.0047554565290314375, 0.005155194144194754, 0.005474904977842727, 0.005480673908085349, 0.004107186152213814, 0.004414379469111339, 0.005410620608503747, 0.005749264429975706, 0.0058876968535609125, 0.005319418814495649, 0.00513039239789115, 0.005663794699662232, 0.004418842743064668, 0.004064682471739925, 0.004393567182673486, 0.005687958180107989, 0.005134741625955735, 0.00538559841615058, 0.004195299601168858, 0.004192965112425178, 0.005364608682937654, 0.00489045978938392, 0.004166119136847488, 0.00411787993650427, 0.005907607396332376, 0.005015735075727645, 0.004624659705168721, 0.0046477564457318545, 0.004342671213892046, 0.0051187502650797655, 0.004836657837757147, 0.005491612695740479]

def _distance(zi, xh):
    return math.sqrt((zi[0] - xh[0]) ** 2 + (zi[1] - xh[1]) ** 2)
def init_pop(N,epsilon,Em,Sets,dataset):

    print("start")
    init_slove = [[] for i in range(N)]
    count=0
    Mids=[[] for i in range(N)]
    Ks = []
    DPPC_Error = []
    DPPC_QLoss = []
    for loc_set in Sets:
        print(len(loc_set))
        fir_pro1 = []
        for s in loc_set:
            ind = dataset.index(s)
            fir_pro1.append(fir_pro[ind])
        sc = DPIVE(loc_set, fir_pro1, epsilon, Em)
        Error, QLoss, base_k,bag_list ,mid= sc.DPPC()
        DPPC_Error.append(Error)
        DPPC_QLoss.append(QLoss)
        init_slove[0].append(bag_list)
        Mids[0].append(mid)
        Ks.append(base_k)
        print(len(bag_list),bag_list)
        print(len(mid),mid)
    print(Ks)
    # print("init_slove[0]",init_slove[0])
    # f = open("./da1.txt","w",encoding='utf-8')
    # f.write(str(init_slove[0]))
    # Ks = [20]
    for i in range(1,N):
        print(i)
        for loc_set in Sets:
            t = Sets.index(loc_set)
            if 2*(Ks[t]+1) <= len(loc_set):
                ra = random.randint(0,100)
                if ra > 50 :
                    k0 = Ks[t]+1
                else:
                    k0 = Ks[t]
            else:
                k0 = Ks[t]
            # print(k0,Ks[t])
            mid_array = []
            # temp = random.randint(1,k0)
            # print("temp",temp)
            index = random.sample(range(0, len(loc_set)), k0)
            for j in index:
                mid_array.append(loc_set[j])
            res, mid = SC1.kmeans_once(loc_set, mid_array)
            list, mid_ = SC2.kmeans1(loc_set, mid, epsilon, Em,dataset)
            init_slove[i].append(list)
            Mids[i].append(mid_)
    for i in range(N):
        print(i)

        sens_list, ep_list = get_sens_ep(init_slove[i], epsilon, Em,dataset)

        j = 0
        for area in Sets:
            release_pro_set = []
            for l in range(len(area)):
                loc = area[l]
                index = dataset.index(loc)
                release_pro_set.append(SC2.computer_pro(loc, sens_list[index], ep_list[index], area))
            obj_value = [SC2.compute_QLoss(release_pro_set,area,dataset), -SC2.compute_Error(release_pro_set,area,dataset)]
            init_slove[i][j].append(obj_value)
            j+=1
    result = []
    for i in range(N):
        result.append(init_slove[i][0])
    return result,Ks[0],Mids[0]
def get_sens_ep(set, epsilon,Em,dataset):
    sens_list = [0 for i in range(len(dataset))]
    ep_list = [0 for i in range(len(dataset))]
    for i in range(len(set)):
        # print(set[i])
        for j in range(len(set[i])):
            index=[dataset.index(loc) for loc in set[i][j]]

            pro = []
            for t in index:
                pro.append(fir_pro[t])
            E_fi = SC1.compute_E_Phi(set[i][j], pro,dataset)
            if E_fi >= math.exp(epsilon) * Em:
                eps_g = epsilon
            else:
                eps_g = math.log(E_fi/Em)
            D = SC1._min_cycle(set[i][j])
            # if D>10:
            #     print(D,set[i][j])
            for ind in index:
                sens_list[ind] = D
                ep_list[ind] = eps_g
    return sens_list, ep_list
def init_Single(N,epsilon,Em,dataset,loc_set,k):
    init_slove = [[] for i in range(N)]
    Mids = [[] for i in range(N)]
    count = 0
    for i in range(N):
        print(i)
        mid_array = []
        index = random.sample(range(0, len(loc_set)), k)
        for j in index:
            mid_array.append(loc_set[j])
        list, mid_ = SC2.kmeans1(loc_set, mid_array, epsilon, Em,dataset)
        init_slove[i]= list
        Mids[i].append(mid_)
    # print("init_solve:",init_slove)
    for i in range(N):
        print(i)
        # sens_list, ep_list = SC1.get_sens_ep(init_slove[i], epsilon, Em, dataset)
        fir_pro_child2 = []
        for loc in loc_set:
            ind = dataset.index(loc)
            fir_pro_child2.append(fir_pro[ind])
        # print("init_slove[i]:",init_slove[i])
        sens_list, ep_list = SC2.get_sens_ep2(init_slove[i], epsilon, Em, fir_pro_child2, loc_set)
        # print("sens_list:",sens_list)
        release_pro_set = []
        for l in range(len(loc_set)):
            loc = loc_set[l]
            release_pro_set.append(SC2.computer_pro(loc, sens_list[l], ep_list[l], loc_set))
        obj_value1 = [SC2.compute_QLoss(release_pro_set, loc_set, dataset),
                      -SC2.compute_Error_child_1(release_pro_set, loc_set, dataset, fir_pro_child2)]
        init_slove[i].append(obj_value1)
        # print(obj_value1)
    return init_slove
def init_Qk(Sets,dataset,epsilon, Em):
    print("init QKmeans")
    K = []
    DPPC_Error=[]
    DPPC_QLoss = []
    for set in Sets:
        fir_pro1 = []
        for s in set:
            ind = dataset.index(s)
            fir_pro1.append(fir_pro[ind])
        sc = DPIVE(set, fir_pro1, epsilon, Em)
        Error, QLoss, base_k = sc.DPPC()
        DPPC_Error.append(Error)
        DPPC_QLoss.append(QLoss)
        print("DPPC Error and QLoss is: ", Error, QLoss)
        print('DPPC k is:', base_k)
        K.append(base_k)
    return K
def non_sort(population, M):
    # print('population',population)
    N = len(population)
    # print(N)
    front = [[]]
    n = [0 for i in range(0, N)]
    S = [[] for i in range(0, N)]
    for i in range(0, N):
        n[i] = 0
        S[i] = []
        for j in range(0, N):
            dom_less = 0
            dom_equal = 0
            dom_more = 0
            for l in range(0, M):
                if (population[i][-1][l] < population[j][-1][l]):
                    dom_more = dom_more + 1
                elif (population[i][-1][l] == population[j][-1][l]):
                    dom_equal = dom_equal + 1
                else:
                    dom_less = dom_less + 1
            if dom_more == 0 and dom_equal != M:
                n[i] = n[i] + 1
            elif dom_less == 0 and dom_equal != M:
                S[i].append(j)
    for i in range(N):
        if n[i] == 0:
            population[i].append([0])
            if i not in front[0]:
                front[0].append(i)
    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    population[q].append([i + 1])
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)
    index_of_front = []
    front = np.delete(front, len(front) - 1, axis=0)
    for i in range(N):
        index_of_front.append(population[i][-1][0])
    index_of_front = np.argsort(index_of_front, axis=0)
    sorted_by_front = [[] for i in range(0, len(index_of_front))]
    for i in range(0, len(index_of_front)):
        sorted_by_front[i] = population[index_of_front[i]]
    z = [[] for i in range(N)]
    current_index = 0

    for front_rank in range(0, len(front)):
        n = 0
        y = [[] for i in range(len(front[front_rank]))]
        previous_index = current_index
        for i in range(0, len(front[front_rank])):
            y[i] = sorted_by_front[current_index + i]
            n = n + 1
        current_index = current_index + n
        t = -2
        for i in range(0, M):
            index_of_objective = []
            for j in range(len(front[front_rank])):
                index_of_objective.append(y[j][t][i])
            index_of_objective = np.argsort(index_of_objective)
            sorted_by_objective = [[] for k in range(0, len(index_of_objective))]
            for j in range(0, len(index_of_objective)):
                sorted_by_objective[j] = y[index_of_objective[j]][0:t + 1]


            f_max = sorted_by_objective[len(index_of_objective) - 1][-1][i]
            f_min = sorted_by_objective[0][-1][i]


            if (len(y) != 1):
                y[index_of_objective[len(index_of_objective) - 1]].append([math.inf])
                y[index_of_objective[0]].append([math.inf])
            else:
                y[index_of_objective[0]].append([math.inf])
            for j in range(1, len(index_of_objective) - 1):
                next_obj = sorted_by_objective[j + 1][-1][i]
                previous_obj = sorted_by_objective[j - 1][-1][i]
                if (f_max - f_min == 0):
                    y[index_of_objective[j]].append([math.inf])
                else:
                    y[index_of_objective[j]].append([(next_obj - previous_obj) / (f_max - f_min)])
            t = t - 1
        for i in range(len(y)):
            y[i][-1][0] = y[i][-1][0] + y[i][-2][0]
            del (y[i][-2])
        z[previous_index:current_index] = y
    return z
def tour_select(population, pool, tour):
    pop = np.shape(population)[0]
    candidate = [[] for i in range(0, tour)]
    individual_rank = [[] for i in range(0, tour)]
    individual_distance = [[] for i in range(0, tour)]
    f = [[] for i in range(pool)]
    for i in range(0, pool):
        candidate = random.sample(range(pop),2)
        for j in range(0, tour):
            individual_rank[j] = population[candidate[j]][-2][0]
            individual_distance[j] = population[candidate[j]][-1][0]
        min_candidate = []
        max_candidate = []
        for j in range(0, tour):
            if (individual_rank[j] == min(individual_rank)):
                min_candidate.append(candidate[j])
            if (individual_distance[j] == max(individual_distance)):
                max_candidate.append(candidate[j])
        if (len(min_candidate) != 1):
            if (len(max_candidate) != 1):
                f[i] = population[min_candidate[0]]
            else:
                f[i] = population[candidate[np.argmax(individual_distance)]]
        else:
            f[i] = population[candidate[np.argmin(individual_rank)]]
    return f
def generic_operate(parent_pop,k,set,dataset,mid):
    A = 5
    m = np.shape(parent_pop)[0]
    offspring = []
    for i in range(m):
        # print(i)
        count=0
        flag=False
        while count < 3:
            parent_list = random.sample(range(0, m), A)
            parent_accross = []
            for j in range(A):
                parent_accross.append(parent_pop[parent_list[j]])
            point = []
            mid_array = []
            parent_accross.sort(key=lambda x: (-x[-2][0], x[-1]))
            for j in range(len(parent_accross)):
                for part in range(len(parent_accross[j]) - 3):
                    point.append(np.average(parent_accross[j][part], axis=0).tolist())
            index = random.sample(range(0, len(point)), len(parent_accross[-1]) - 3)
            for j in index:
                mid_array.append(point[j])
            if len(point)<len(mid_array)*2:
                point = set
            res, mid = SC2.kmeans_once(point, mid_array)
            child, mid_ = SC2.kmeans1(set, mid, epsilon, Em,dataset)
            count += 1
        if count == 50:
            print("Error:this cross is not good")
        fir_pro_child1 = []
        for loc in set:
            ind = dataset.index(loc)
            fir_pro_child1.append(fir_pro[ind])
        sens_list, ep_list = SC2.get_sens_ep2(child,epsilon,Em,fir_pro_child1,set)
        release_pro_set = []
        for l in range(len(set)):
            loc = set[l]
            release_pro_set.append(SC2.computer_pro(loc, sens_list[l], ep_list[l], set))
        obj_value0 = [SC2.compute_QLoss(release_pro_set,set,dataset), -SC2.compute_Error_child(release_pro_set,set,dataset,fir_pro_child1)]
        child.append(obj_value0)
        offspring.append(child)
        rate = np.random.random()
        if rate<0.1:
            flag=False
            count = 0
            while  count < 3:
                mid_array_1 = []
                k0 = int(len(mid) * 0.5)
                count = 0
                while count < k0:
                    index_ = random.sample(range(0, len(set)), 1)
                    if set[index_[0]] not in mid:
                        mid_array_1.append(set[index_[0]])
                        count += 1
                index_ = random.sample(range(0, len(mid)), len(mid)-k0)
                for j in index_:
                    mid_array_1.append(mid[j])
                child1, mid_1 = SC2.kmeans1(set, mid_array_1,epsilon,Em,dataset)
                count+=1
            if count == 50:
                print("Error:this change is not good")
            fir_pro_child2 = []
            for loc in set:
                ind = dataset.index(loc)
                fir_pro_child2.append(fir_pro[ind])
            sens_list, ep_list = SC2.get_sens_ep2(child1, epsilon, Em, fir_pro_child2, set)
            release_pro_set = []
            for l in range(len(set)):
                loc = set[l]
                release_pro_set.append(SC2.computer_pro(loc, sens_list[l], ep_list[l], set))
            obj_value1 = [SC2.compute_QLoss(release_pro_set, set, dataset), -SC2.compute_Error_child(release_pro_set, set, dataset,fir_pro_child2)]
            child1.append(obj_value1)
            offspring.append(child1)
    return offspring
def replace(mid_population, N):
    m = np.shape(mid_population)[0]
    f = [[] for i in range(N)]

    index = []
    mid_population1 = []
    yongjidu = []
    for i in range(m):
        yongjidu.append(mid_population[i][-1][0])
    yong_index = np.argsort(yongjidu).tolist()
    yong_index.sort(reverse=True)
    for i in range(m):
        mid_population1.append(mid_population[yong_index[i]])
    for i in range(m):
        index.append(mid_population1[i][-2][0])
    index = np.argsort(index, axis=0)

    # for i in range(m):
    #     index.append(mid_population[i][-2][0])
    # index = np.argsort(index, axis=0)
    # for i in range(len(index)):
    #     mid_population1.append(mid_population[index[i]])
    # popu = []
    # rank = mid_population1[i][-2][0]
    # for i in range(1,len(mid_population1)):
    #     if rank == mid_population1[i][-2][0]:
    #         popu

    # sorted_population = [[] for i in range(0, len(index))]
    # for i in range(0, len(index)):
    #     sorted_population[i] = mid_population[index[i]]
    sorted_population = [[] for i in range(0, len(index))]
    for i in range(0, len(index)):
        sorted_population[i] = mid_population1[index[i]]
    max_rank = sorted_population[-1][-2][0]
    previouis_index = 0
    current_index = 0
    for i in range(0, int(max_rank)):
        for j in sorted_population:
            if j[-2][0] == i:
                current_index = current_index + 1
        if current_index > N:
            remaining = N - previouis_index
            temp_pop = sorted_population[previouis_index:current_index]
            index_ = []
            for i in range(len(temp_pop)):
                index_.append(-temp_pop[i][-1][0])
            index_ = np.argsort(index_, axis=0)
            for j in range(0, remaining):
                f[previouis_index + j] = temp_pop[index_[j]]
            break

        elif current_index < N:
            f[previouis_index:current_index] = sorted_population[previouis_index:current_index]
        else:
            f[previouis_index:current_index] = sorted_population[previouis_index:current_index]
            break
        previouis_index = current_index
    return f
def cal_HV_single(population):
    m = np.shape(population)[0]
    point = []
    angle = []
    for pop in population:
        point.append(pop[-1])
    point = np.array(point)
    max_x = np.max(point[:, 0])
    max_y = np.max(point[:, 1])
    for i in range(m):
        if (point[i][0] - max_x == 0):
            angle.append(math.inf)
        else:
            angle.append((point[i][1] - max_y) / (point[i][0] - max_x))
    index = np.argsort(angle)
    area = 0
    for i in range(1, m - 1):
        area += np.abs(point[index[i + 1], 0] - point[index[i], 0]) * np.abs(point[index[i], 1] - point[index[0], 1])
    return area
def cal_HV(population):
    m = np.shape(population)[0]
    point = []
    angle = []
    for pop in population:
        point.append(pop[-3])
    point = np.array(point)
    max_x = np.max(point[:,0])
    max_y = np.max(point[:,1])
    for i in range(m):
        if (point[i][0] - max_x == 0):
            angle.append(math.inf)
        else:
            angle.append((point[i][1] - max_y) / (point[i][0] - max_x))
    index = np.argsort(angle)
    area = 0
    for i in range(1,m - 1):
        area += np.abs(point[index[i + 1], 0] - point[index[i], 0]) * np.abs(point[index[i], 1] - point[index[0], 1])
    return area
def single_obj(population,epsilon, Em,dataset,N, k,set):
    # print('population',population)
    population1=copy.deepcopy(population)
    alpha_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    Single_QLoss=[]
    Single_Error=[]
    count=0
    alpha=alpha_list[0]
    popu = []
    Popus = []
    while(True):
        F = []
        count1 = 0
        while count1<10:
            print('alpha:',alpha)
            F_min = alpha * population1[0][-1][0] + (1-alpha) * population1[0][-1][1]
            F_min_1 = -10000
            for i in range(1,N):
                F_cur=alpha*population1[i][-1][0]+(1-alpha)*population1[i][-1][1]
                if F_min > F_cur:
                    F_min_1 = F_min
                    F_min=F_cur
                if abs(F_min_1-F_min) < 1:
                    print(F_min,F_min_1)
                    F.append([F_min, population1[i][-1][0], population1[i][-1][1]])
                    popu.append(population1)
                    count1+=1
                    break
            population1 = init_Single(N,epsilon,Em,dataset,set,k)
        f_min=F[0][0]
        f_index = 0
        for j in range(1,len(F)):
            f_cur = F[j][0]
            if f_min > f_cur:
                f_min = f_cur
                f_index = j
        count += 1
        Popus.append(popu[f_index])
        Single_QLoss.append(F[f_index][1])
        Single_Error.append(F[f_index][2])
        if count > len(alpha_list) - 1:
            min_Ql = min(Single_QLoss)
            ind_i = Single_QLoss.index(min_Ql)
            print("Single objective is completed")
            return Single_QLoss, Single_Error,Popus[ind_i]
        alpha = alpha_list[count]
def get_data(n):
    count = 0
    dataset = qu.get_data('GeoLife.txt')
    qu.draw(dataset)
    N = len(dataset)
    print(N,dataset)
    print(len(fir_pro))
    Deep = math.floor(math.log2(N / n))
    Sets = qu.huafen(dataset, count, Deep)
    return Sets,dataset
def QK(dataset,Sets, epsilon, Em):
    set = Sets[0]
    fir_pro1 = []
    for s in set:
        ind = dataset.index(s)
        fir_pro1.append(fir_pro[ind])
    sc = DPIVE(set, fir_pro1, epsilon, Em)
    Error1, QLoss, base_k,init_solve,mid = sc.DPPC()
    # print("sen_list:",sens_list)
    # release_pro_set =[]
    # for l in range(len(area)):
    #     loc = area[l]
    #     index = area.index(loc)
    #     # print(index, sens_list[index], ep_list[index])
    #     release_pro_set.append(SC1.computer_pro(loc, sens_list[index], ep_list[index], area))
    # Error = -SC1.compute_Error(Error_re, Error_set, dataset)
    # print("QK:",QLoss,Error)
    return QLoss,Error1,base_k,init_solve
def get_result(population):
    x = []
    y = []
    x1=[]
    y1=[]
    min_ql = 10000
    ind = 0
    for pop in population:
        cur_ql = pop[-3][0]
        if pop[-2][0] <= 0 and pop[-1][0]>=0.07:
            x.append(pop[-3][0])
            y.append(pop[-3][1])
        if min_ql > cur_ql:
            min_ql = cur_ql
            ind = population.index(pop)
    m = len(x)
    return x,y,population[ind]
def draw(dataset):
    X = []
    Y = []
    for data in dataset:
        X.append(data[0])
        Y.append(data[1])
    plt.scatter(X,Y)
    plt.show()
if __name__ == '__main__':
    n = 60
    Sets, dataset = get_data(n)
    draw(dataset)
    N = 50
    M = 2
    max_gene = 500
    pool = round(N / 2)
    tour = 2
    epsilon_list = [0.5]
    Em_list = [0.1]
    HV = []
    for epsilon in epsilon_list:
        for Em in Em_list:
            population,k,mid= init_pop(N, epsilon, Em,Sets,dataset)
            print('init completed')
            DPPC_QLoss, DPPC_Error,k0,bag_list= QK(dataset,Sets, epsilon, Em)
            print("QK:",DPPC_QLoss, DPPC_Error)
            print("QKmeans")
            Single_QLoss, Single_Error,Single_popu = single_obj(population,epsilon, Em,dataset,N, k,Sets[0])
            HV_single = cal_HV_single(Single_popu)
            print(HV_single)
            print('Single objective Error and QLoss is:', Single_QLoss, Single_Error)
            population = non_sort(population, M)
            for i in range(max_gene):
                parent_pop = tour_select(population, pool, tour)
                offspring_pop = generic_operate(parent_pop,k,Sets[0],dataset,mid)
                for j in population:
                    del j[-2:]
                mid_population = population
                for pop in offspring_pop:
                    mid_population.append(pop)
                mid_population = non_sort(mid_population, M)
                population = replace(mid_population, N)
                hv = cal_HV(population)
                HV.append([i,hv])
                if (i + 1) % 50 == 0:
                    x,y,re_area = get_result(population)
                    # print(len(x))
                    print('epilson,Em:1111',epsilon,Em,len(x))
                    fig, ax = plt.subplots()
                    plt.scatter(x,y, c='white',edgecolors='r',marker='o',label='Geo-MOEA')
                    plt.scatter([DPPC_QLoss], [-DPPC_Error], c='b', marker='*', label='Qk-means')
                    plt.scatter(Single_QLoss, Single_Error, c='g', marker='^', label='PSO')
                    plt.xlabel('QLoss',fontsize=12)
                    plt.ylabel('ExpErr',fontsize=12)
                    # plt.legend(loc='best')
                    plt.legend(loc=0, numpoints=1)
                    leg = plt.gca().get_legend()
                    ltext = leg.get_texts()
                    plt.setp(ltext, fontsize=17)
                    plt.tick_params(labelsize=12)
                    labels = ax.get_xticklabels() + ax.get_yticklabels()
                    plt.show()
                    HV = np.array(HV)
                    plt.scatter(HV[:,0],HV[:,1],c='blue',marker='o',s= 30,label='HV')
                    plt.xlabel('iter')
                    plt.ylabel('HV')
                    plt.legend(loc='best')
                    plt.show()
                    HV = HV.tolist()
                    file = open('./min.txt',"a")
                    min_x = min(x)
                    in_x = x.index(min_x)
                    min_y = min(y)
                    single_x = min(Single_QLoss)
                    single_y = min(Single_Error)
                    print('x,y:',min_x,min_y)
                    print("QK:",DPPC_QLoss,-DPPC_Error)
                    L = abs(DPPC_QLoss-min_x)/abs(min_x)
                    E = abs(-DPPC_Error-min_y)/abs(min_y)
                    sl = abs(single_x-min_x)/abs(min_x)
                    se  = abs(single_y-min_y)/abs(min_y)



