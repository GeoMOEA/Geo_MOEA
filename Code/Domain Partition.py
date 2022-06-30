def divison(dataset,max1,min1,max2,min2,length,width,flag,count,Deep):
    L = len(dataset)
    Set_1 = []
    Set_2 = []
    if flag == 0:
        mid = (min1 + max1) / 2
        while True:
            j=0
            Set_1 = []
            Set_2 = []
            while True:
                if dataset[j][0] <= mid:
                    Set_1.append(dataset[j])
                else:
                    Set_2.append(dataset[j])
                j+=1
                if j>=L:
                    break
            l1 = len(Set_1)
            l2 = len(Set_2)
            if abs(l1 - l2) <=5:
                break
            if l1>l2:
                mid -= 0.1
            else:
                mid += 0.1
    if flag == 1:
        mid = (min1 + max1) / 2
        while True:
            Set_1 = []
            Set_2 = []
            j=0
            while True:
                if dataset[j][1] <= mid:
                    Set_1.append(dataset[j])
                else:
                    Set_2.append(dataset[j])
                j+=1
                if j>=L:
                    break
            l1 = len(Set_1)
            l2 = len(Set_2)
            if abs(l1-l2)<=5:
                break
            if l1>l2:
                mid -= 0.1
            else:
                mid += 0.1
    return Set_1,Set_2
def huafen(dataset,count,Deep):
    if count == Deep:
        Set.append(dataset)
        return
    length, width, max_x, max_y, min_x, min_y = compute(dataset)
    if length > width:
        Set_1, Set_2 = divison(dataset, int(max_x), int(min_x), int(max_y), int(min_y),length, width,0,count,Deep)
    else:
        Set_1, Set_2 = divison(dataset,int(max_x), int(min_x), int(max_y), int(min_y),length, width,1,count,Deep)
    count +=1
    huafen(Set_1,count,Deep)
    huafen(Set_2,count,Deep)
    return Set