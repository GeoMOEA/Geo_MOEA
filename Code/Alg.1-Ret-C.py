def compute_Sparsity(set):
    Sparsity=[]
    for i in range(len(set)):
        _sum=0.
        for j in range(len(set)):
            _sum+=_distance(set[i],set[j])
        Sparsity.append(_sum)
    Sparsity_index=np.argsort(Sparsity)
    return Sparsity_index
def _distance(zi, xh):
    return math.sqrt((zi[0] - xh[0]) ** 2 + (zi[1] - xh[1]) ** 2)
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
def Ret_C(loc_set,mid_array,epsilon,Em,dataset):
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