def PSO(population,epsilon, Em,dataset,N, k,set):
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