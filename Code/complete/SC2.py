#/usr/bin/env python
#--*--coding:utf-8--*--
import math
import numpy as np
import random
import Domain_Partition
import copy



fir_pro = [0.0043335808056442675, 0.004997436325052889, 0.004482487708706725, 0.00593378702413151, 0.004787080440321329, 0.004732256498500137, 0.005897219331343113, 0.004100288292199329, 0.004033062306351729, 0.004942885702298939, 0.005698611147169671, 0.00433605333033881, 0.005816448561766392, 0.004838966665201451, 0.005820275189591764, 0.005294310232056265, 0.005249544513468861, 0.004345862947450193, 0.005671800860211339, 0.005120696242003786, 0.0047554565290314375, 0.005155194144194754, 0.005474904977842727, 0.005480673908085349, 0.004107186152213814, 0.004414379469111339, 0.005410620608503747, 0.005749264429975706, 0.0058876968535609125, 0.005319418814495649, 0.00513039239789115, 0.005663794699662232, 0.004418842743064668, 0.004064682471739925, 0.004393567182673486, 0.005687958180107989, 0.005134741625955735, 0.00538559841615058, 0.004195299601168858, 0.004192965112425178, 0.005364608682937654, 0.00489045978938392, 0.004166119136847488, 0.00411787993650427, 0.005907607396332376, 0.005015735075727645, 0.004624659705168721, 0.0046477564457318545, 0.004342671213892046, 0.0051187502650797655, 0.004836657837757147, 0.005491612695740479]

def _distance(zi, xh):
    return math.sqrt((zi[0] - xh[0]) ** 2 + (zi[1] - xh[1]) ** 2)
def computer_pro(loc, sens, epsilon,loc_set):
    temp = []
    for temp_loc in loc_set:
        temp.append(math.exp((epsilon * _distance(temp_loc, loc)) / (-2 * sens)))
    sum_ = sum(temp)
    # rel_pro = [i / sum_ for i in temp]
    rel_pro = []
    for te in temp:
        rel_pro.append(te/sum_)
    return rel_pro
def _min_cycle(set):
    if len(set) <= 1:
        return 0
    elif len(set) == 2:
        return _distance(set[0], set[1])
    else:
        cur_max = 0
        for i in range(len(set)):
            for j in range(i + 1, len(set)):
                if _distance(set[i], set[j]) > cur_max:
                    cur_max = _distance(set[i], set[j])
        return cur_max
def get_sens_ep1(set, epsilon, Em,dataset):
    sens_list = [0 for i in range(len(dataset))]
    ep_list = [0 for i in range(len(dataset))]
    for i in range(len(set)):
        # print(set[i])
        for j in range(len(set[i])):
            index = [dataset.index(loc) for loc in set[i][j]]
            pro = []
            for t in index:
                pro.append(fir_pro[t])
            E_fi = compute_E_Phi(set[i][j], pro, dataset)
            if E_fi >= math.exp(epsilon) * Em:
                eps_g = epsilon
            else:
                eps_g = epsilon
            D = _min_cycle(set[i][j])
            for ind in index:
                sens_list[ind] = D
                ep_list[ind] = eps_g
    return sens_list, ep_list
def get_sens_ep2(set,epsilon,Em,fir_pro_child1,area):
    sens_list = [0 for i in range(len(area))]
    ep_list = [0 for i in range(len(area))]
    for i in range(len(set)):
        # print(set[i]
        index = [area.index(loc) for loc in set[i]]
        pro = []
        for t in index:
            if fir_pro_child1[t] not in pro:
                pro.append(fir_pro_child1[t])
        E_fi = compute_E_Phi(set[i], pro, area)
        if E_fi >= math.exp(epsilon) * Em:
            eps_g = epsilon
        else:
            eps_g = math.log(E_fi/Em)
        D = _min_cycle(set[i])
        for ind in index:
            sens_list[ind] = D
            ep_list[ind] = eps_g
    return sens_list, ep_list
def get_sens_ep(set, epsilon,Em,dataset):
    sens_list = [0 for i in range(len(dataset))]
    ep_list = [0 for i in range(len(dataset))]
    for i in range(len(set)):
        # print(set[i])
        for j in range(len(set[i])):
            print("set[i][j]:",set[i][j])
            index=[dataset.index(loc) for loc in set[i][j]]
            pro = []
            print("index:",index)
            print(len(fir_pro))
            for t in index:
                pro.append(fir_pro[t])
            E_fi = compute_E_Phi(set[i][j], pro,dataset)
            if E_fi >= math.exp(epsilon) * Em:
                eps_g = epsilon
            else:
                eps_g = epsilon
            D = _min_cycle(set[i][j])
            # if D>10:
            #     print(D,set[i][j])
            for ind in index:
                sens_list[ind] = D
                ep_list[ind] = eps_g
    return sens_list, ep_list
def get_sens_ep_Single(set, epsilon,Em,dataset):
    sens_list = [0 for i in range(len(dataset))]
    ep_list = [0 for i in range(len(dataset))]
    for i in range(len(set)):
        # print(set[i])
        for j in range(len(set[i])):
            index = [dataset.index(loc) for loc in set[i][j]]

            pro = []
            for t in index:
                pro.append(fir_pro[t])
            E_fi = compute_E_Phi(set[i][j], pro, dataset)
            if E_fi >= math.exp(epsilon) * Em:
                eps_g = epsilon
            else:
                eps_g = epsilon
            D = _min_cycle(set[i][j])
            # if D>10:
            #     print(D,set[i][j])
            for ind in index:
                sens_list[ind] = D
                ep_list[ind] = eps_g
    return sens_list, ep_list
def gen_sed_pro(x, rel_pro_set,loc_set,dataset):
    temp_list = []
    for i in range(len(loc_set)):
        index = dataset.index(loc_set[i])
        # print(rel_pro_set[i][x],rel_pro_set[x][i])
        cur_pro = fir_pro[index] * rel_pro_set[x][i]
        temp_list.append(cur_pro)
    sum_ = sum(temp_list)
    sed_pro = [i / sum_ for i in temp_list]
    return sed_pro
def gen_sed_pro_child(x, rel_pro_set,loc_set,dataset,fir_pro_child1):
    temp_list = []
    for i in range(len(loc_set)):
        # print(rel_pro_set[i][x],rel_pro_set[x][i])
        cur_pro = fir_pro_child1[i] * rel_pro_set[x][i]
        temp_list.append(cur_pro)
    sum_ = sum(temp_list)
    sed_pro = [i / sum_ for i in temp_list]
    return sed_pro
def gen_sed_pro_child_1(x, rel_pro_set,loc_set,dataset,fir_pro_child1):
    temp_list = []
    for i in range(len(loc_set)):
        # print(rel_pro_set[i][x],rel_pro_set[x][i])
        cur_pro = fir_pro_child1[i] * rel_pro_set[i][x]
        temp_list.append(cur_pro)
    sum_ = sum(temp_list)
    sed_pro = [i / sum_ for i in temp_list]
    return sed_pro
def bayesian_gongji(sed_pro,loc_set):
    min_ = _pro_dis(loc_set[0], loc_set, sed_pro)
    min_index = 0
    for i in range(1,len(loc_set)):
        cur_min = _pro_dis(loc_set[i], loc_set, sed_pro)
        if cur_min < min_:
            min_ = cur_min
            min_index = i
    return loc_set[min_index]
def compute_E_Phi(temp_loc_set, temp_fir_pro,dataset):
    temp_min = None
    t_list = []
    for loc in temp_loc_set:
        if loc not in t_list:
            t_list.append(loc)
    guiyihua_pro = _guiyihua(temp_fir_pro)
    for loc in dataset:
        cur_min = 0.
        for i in range(len(t_list)):
            cur_min += guiyihua_pro[i] * _distance(loc, t_list[i])
        if temp_min is None:
            temp_min = cur_min
        elif cur_min < temp_min:
            temp_min = cur_min
    return temp_min
def _guiyihua(temp_fir_pro):
    sum_ = sum(temp_fir_pro)
    return [i/sum_ for i in temp_fir_pro]
def _pro_dis(cur_loc,location_set, sed_pro):
    sum_ = 0.
    for i in range(len(location_set)):
        sum_ += (sed_pro[i] * _distance(cur_loc, location_set[i]))
    return sum_
def global_distace(loc,set):
    sum=0
    for w in set:
        sum += _distance(loc,w)
    return sum/len(set)
def judge_EPI(set,loc_set,epsilon,Em):
    pro=[]
    for loc in set:
        if fir_pro[loc_set.index(loc)] not in pro:
            pro.append(fir_pro[loc_set.index(loc)])
    E_fi=compute_E_Phi(set,pro,loc_set)
    if E_fi>=math.exp(epsilon)*Em:
        return True
    else:
        return False
def compute_Sparsity(set):
    Sparsity=[]
    for i in range(len(set)):
        _sum=0.
        for j in range(len(set)):
            _sum+=_distance(set[i],set[j])
        Sparsity.append(_sum)
    Sparsity_index=np.argsort(Sparsity)
    return Sparsity_index

def get_sens_eg(set, epsilon,Em,fir):
    ep_list = [0 for i in range(len(set))]
    for i in range(len(set)):
        index=[set.index(loc) for loc in set]
        E_fi = compute_E_Phi(set, fir,set)
        if E_fi >= math.exp(epsilon) * Em:
            eps_g = epsilon
        else:
            eps_g = epsilon
        D = _min_cycle(set)
        for ind in index:
            ep_list[ind] = eps_g
    return D, ep_list

def Max_ep(bag_list,Em,e0,dataset):
    eg_list = []
    eg_list.append(e0/(2*_min_cycle(bag_list)))
    list = []
    for loc in bag_list:
        if loc not in list:
            list.append(loc)
    del_list = []
    del list[-1]
    pro = []
    length = len(list)
    for loc in list:
        ind = dataset.index(loc)
        pro.append(fir_pro[ind])
    while length >= 2:
        Ehi = compute_E_Phi(list,pro,dataset)
        e = math.log(Ehi/Em)
        eg_list.append(e/(2*_min_cycle(list)))
        if length > 2:
            del list[-1]
            del pro[-1]
        length -= 1
    eg_list.reverse()
    max_eg = max(eg_list)
    max_ind = eg_list.index(max_eg)
    dcount = len(eg_list) - max_ind  - 1
    i = len(bag_list)-1
    while dcount > 0:
        del_list.append(bag_list[i])
        del bag_list[i]
        i -= 1
        dcount -= 1
    return  bag_list, del_list

def kmeans(loc_set,mid_array,epsilon,Em):
    n=len(loc_set)
    k=len(mid_array)
    count=0
    while True:
        sign = [0 for i in range(k)]
        bag_list = [[] for i in range(k)]
        remain_locs = []
        surplus_locs=[]
        for loc in loc_set:
            remain_locs.append(loc)
        for i in range(k):
            for j in range(2):
                index = np.argmin([_distance(mid_array[i], remain_loc) for remain_loc in remain_locs])
                bag_list[i].append(remain_locs[index])
                del remain_locs[index]
        for i in range(k):
            flag=judge_EPI(bag_list[i],loc_set,epsilon,Em)
            if flag==True:
                sign[i]=1
        Sparsity_index=compute_Sparsity(remain_locs)
        for i in range(len(remain_locs)):
            global_distance1 = []
            for mid in mid_array:
                global_distance1.append(_distance(remain_locs[Sparsity_index[i]], mid))
            x_index1=np.argsort(global_distance1)
            sign_flag=True
            for x in x_index1:
                if sign[x]==0:
                    bag_list[x].append(remain_locs[Sparsity_index[i]])
                    flag = judge_EPI(bag_list[x],loc_set,epsilon,Em)
                    if flag == True:
                        sign[x] = 1
                    sign_flag=False
                    break
            if sign_flag==True:
                surplus_locs.append(remain_locs[Sparsity_index[i]])
        for i in range(len(surplus_locs)):
            global_distance2 = []
            for mid in mid_array:
                global_distance2.append(_distance(surplus_locs[i], mid))
            x_index2=np.argsort(global_distance2)
            for x in x_index2:
                bag_list[x].append(surplus_locs[i])
                Flag=judge_EPI(bag_list[x],loc_set,epsilon,Em)
                if Flag==False:
                        # print(surplus_locs[i])
                    bag_list[x].remove(surplus_locs[i])
                else:
                    break
        mid_error=0

        # print('pre:',mid_array)
        for i in range(k):
            if len(bag_list[i])>0:
                x_bar=0
                y_bar=0
                for loc in bag_list[i]:
                    x_bar+=loc[0]
                    y_bar+=loc[1]
                x_bar/=len(bag_list[i])
                y_bar/=len(bag_list[i])
                mid_error+=_distance(mid_array[i],[x_bar,y_bar])
                mid_array[i]=[x_bar,y_bar]

            # print('later:',mid_array)
            # print(mid_error)
        if(mid_error<1):
            break
        count+=1
        if count>50:
            break
    return bag_list,mid_array,sum(sign)==k
def kmeans1(loc_set,mid_array,epsilon,Em,dataset):
    n=len(loc_set)
    k=len(mid_array)
    count=0
    sign = [0 for i in range(k)]
    bag_list = [[] for i in range(k)]
    remain_locs = []
    surplus_locs=[]
    for loc in loc_set:
        remain_locs.append(loc)
    for i in range(k):
        for j in range(2):
            index = np.argmin([_distance(mid_array[i], remain_loc) for remain_loc in remain_locs])
            bag_list[i].append(remain_locs[index])
            del remain_locs[index]
    for i in range(k):
        flag=judge_EPI(bag_list[i],loc_set,epsilon,Em)
        if flag==True:
            sign[i]=1
    s_flag = True
    for si in sign:
        if si != True:
            s_flag = False
    if s_flag == True:
        Sparsity_index=compute_Sparsity(remain_locs)
        for i in range(len(remain_locs)):
            global_distance1 = []
            for mid in mid_array:
                global_distance1.append(_distance(remain_locs[Sparsity_index[i]], mid))
            x_index1=np.argsort(global_distance1)
            sign_flag=True
            for x in x_index1:
                if sign[x]==0:
                    bag_list[x].append(remain_locs[Sparsity_index[i]])
                    flag = judge_EPI(bag_list[x],loc_set,epsilon,Em)
                    sign_flag = False
                    if flag == True:
                        sign[x] = 1
                    break
            if sign_flag==True:
                surplus_locs.append(remain_locs[Sparsity_index[i]])
        for i in range(len(surplus_locs)):
            global_distance2 = []
            for mid in mid_array:
                global_distance2.append(_distance(surplus_locs[i], mid))
            x_index2=np.argsort(global_distance2)
            for x in x_index2:
                bag_list[x].append(surplus_locs[i])
                Flag=judge_EPI(bag_list[x],loc_set,epsilon,Em)
                if Flag==False:
                    bag_list[x].remove(surplus_locs[i])
                else:
                    break
    else:
        Sparsity_index = compute_Sparsity(remain_locs)
        for i in range(len(remain_locs)):
            # print(i)
            # print(sign)
            r_dis = []
            # print(remain_locs[Sparsity_index[i]])
            for j in range(len(mid_array)):
                if sign[j] == 1:
                    r_dis.append(-999)
                else:
                    r_dis.append(_distance(remain_locs[Sparsity_index[i]], mid_array[j]))
            r_index = np.argsort(r_dis)
            sign_flag = True
            # print(r_dis)
            # print(r_index)
            for x in r_index:
                if sign[x] == 0 and r_dis[x] != -999:
                    if remain_locs[Sparsity_index[i]] not in bag_list[x]:
                        bag_list[x].append(remain_locs[Sparsity_index[i]])
                    flag = judge_EPI(bag_list[x], loc_set, epsilon, Em)
                    if flag == True:
                        sign[x] = 1
                        if sign[x] == 1 and len(bag_list[x]) > 2:
                            bag_list[x],del_list = Max_ep(bag_list[x],Em,epsilon,dataset)
                            for d_loc in del_list:
                                d_dis = []
                                for mi in range(len(mid_array)):
                                    s_flag = True
                                    if mi == x:
                                        d_dis.append(999)
                                    else:
                                        d_dis.append(_distance(d_loc,mid_array[mi]))
                                d_index = np.argsort(d_dis)
                                d_flag = True
                                for dx in d_index:
                                    if sign[dx] == 0 and d_dis[dx] != 999:
                                        if d_loc not in bag_list[dx]:
                                            bag_list[dx].append(d_loc)
                                        d_flag == False
                                        break
                                if d_flag == True:
                                    bag_list[d_index[0]].append(d_loc)
                    sign_flag = False
                    break
            if sign_flag == True:
                surplus_locs.append(remain_locs[Sparsity_index[i]])
        for i in range(len(surplus_locs)):
            global_distance2 = []
            for mid in mid_array:
                global_distance2.append(_distance(surplus_locs[i], mid))
            # x = np.argmin(global_distance2)
            x_index2=np.argsort(global_distance2)
            for x in x_index2:
                bag_list[x].append(surplus_locs[i])
                Flag=judge_EPI(bag_list[x],loc_set,epsilon,Em)
                if Flag==False:
                    # print(surplus_locs[i])
                    bag_list[x].remove(surplus_locs[i])
                else:
                    break

    for i in range(k):
        if len(bag_list[i])>0:
            x_bar=0
            y_bar=0
            for loc in bag_list[i]:
                x_bar+=loc[0]
                y_bar+=loc[1]
            x_bar/=len(bag_list[i])
            y_bar/=len(bag_list[i])
                # mid_error+=self.distance(mid_array[i],[x_bar,y_bar])
            mid_array[i]=[x_bar,y_bar]

    return bag_list,mid_array
def compute_Error(release_pro_set,area,dataset):

    bys_avg = []
    att_bay_loc_list = []
    for i in range(len(area)):

        sed_pro = gen_sed_pro(i, release_pro_set,area,dataset)

        att_bay_loc = bayesian_gongji(sed_pro,area)
        att_bay_loc_list.append(att_bay_loc)
    # print(att_bay_loc_list)
    for loc in area:
        loc_index = area.index(loc)
        release_pro = release_pro_set[loc_index]
        sum_ = 0
        for i in range(len(area)):

            sum_ += release_pro[i] * _distance(loc, att_bay_loc_list[i])
        bys_avg.append(sum_)
    bys_error = 0.
    for i in range(len(area)):
        ind = dataset.index(area[i])
        bys_error += (fir_pro[ind] * bys_avg[i])
    return bys_error
def compute_QLoss(release_pro_set,loc_set,dataset):

    u_list = []
    # print(loc_set)
    for k in range(len(loc_set)):
        ult_loss = 0
        for i in range(len(loc_set)):  # x'
            ult_loss += release_pro_set[k][i] * _distance(loc_set[k], loc_set[i])
        u_list.append(ult_loss)
    utility_loss = 0.
    for i in range(len(loc_set)):
        loc = loc_set[i]
        index = dataset.index(loc)
        utility_loss += (fir_pro[index] * u_list[i])
    return utility_loss
def compute_Error_child(release_pro_set,area,dataset,fir_pro_child1):
    bys_avg = []
    att_bay_loc_list = []
    for i in range(len(area)):

        sed_pro = gen_sed_pro_child(i, release_pro_set,area,dataset,fir_pro_child1)

        att_bay_loc = bayesian_gongji(sed_pro,area)
        att_bay_loc_list.append(att_bay_loc)
    # print(att_bay_loc_list)
    for loc in area:
        loc_index = area.index(loc)
        release_pro = release_pro_set[loc_index]
        sum_ = 0
        for i in range(len(area)):

            sum_ += release_pro[i] * _distance(loc, att_bay_loc_list[i])
        bys_avg.append(sum_)
    bys_error = 0.
    for i in range(len(area)):
        bys_error += (fir_pro_child1[i] * bys_avg[i])
    return bys_error
def compute_Error_child_1(release_pro_set,area,dataset,fir_pro_child1):

    bys_avg = []
    att_bay_loc_list = []
    for i in range(len(area)):

        sed_pro = gen_sed_pro_child_1(i, release_pro_set,area,dataset,fir_pro_child1)

        att_bay_loc = bayesian_gongji(sed_pro,area)
        att_bay_loc_list.append(att_bay_loc)
    # print(att_bay_loc_list)
    for loc in area:
        loc_index = area.index(loc)
        release_pro = release_pro_set[loc_index]
        sum_ = 0
        for i in range(len(area)):

            sum_ += release_pro[i] * _distance(loc, att_bay_loc_list[i])
        bys_avg.append(sum_)
    bys_error = 0.
    for i in range(len(area)):
        bys_error += (fir_pro_child1[i] * bys_avg[i])
    return bys_error
def construct_X(Lists,mid_lists):
    lists = []
    for li in Lists:
        for i in range(len(li)):
            lists.append(li[i])
    print("Lists",Lists)
    print(len(lists),lists)
    area = []
    areas = []
    mid_list =[]
    c = 0
    for i in range(len(mid_lists)):
        for j in range(len(mid_lists[i])):
            mid_list.append([mid_lists[i][j],c])
            c+=1
    print(mid_list)
    print("mid_list",len(mid_list),mid_list)
    for i in range(len(mid_list)):
        Re = []
        Dis = []
        for j in range(len(mid_list)):
            Dis.append(_distance(mid_list[i][0],mid_list[j][0]))
        index_list = np.argsort(Dis)
        le = len(index_list)
        # print(index_list)
        Re.append(lists[mid_list[index_list[0]][1]])
        Re.append(lists[mid_list[index_list[1]][1]])
        # print("Re", Re)
        count = len(lists[mid_list[index_list[0]][1]])+len(lists[mid_list[index_list[1]][1]])
        for index in range(2,le):
            # print(index,count)
            if count >= 50:
                break
            Re.append(lists[mid_list[index_list[index]][1]])
            count += len(lists[mid_list[index_list[index]][1]])
        # if count<50:
        #     print("you")
        RES = []
        areas.append(Re)
        for re in Re:
            for r in re:
                RES.append(r)
        area.append(RES)
        # areas.append(RES)
    # print(len(area))
    return area,areas
def kmeans_once(loc_set, mid_array):
    n = len(loc_set)
    k = len(mid_array)
    bag_list = [[] for i in range(k)]

    remain_locs = []
    for loc in loc_set:
        remain_locs.append(loc)
    for i in range(k):
        for j in range(2):
            index = np.argmin([_distance(mid_array[i], remain_loc) for remain_loc in remain_locs])
            bag_list[i].append(remain_locs[index])
            del remain_locs[index]
    for i in range(len(remain_locs)):
        global_distance = []
        for mid in mid_array:
            global_distance.append(_distance(remain_locs[i], mid))
        x = np.argmin(global_distance)
        bag_list[x].append(remain_locs[i])

    for i in range(k):
        if len(bag_list[i]) > 0:
            x_bar = 0
            y_bar = 0
            for loc in bag_list[i]:
                x_bar += loc[0]
                y_bar += loc[1]
            x_bar /= len(bag_list[i])
            y_bar /= len(bag_list[i])
            mid_array[i] = [x_bar, y_bar]
    return bag_list, mid_array


