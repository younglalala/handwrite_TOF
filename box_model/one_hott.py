import numpy as np

def one_hot(data,cls):
    all_data=[]
    for d in data:
        zeor_data = [x * 0 for x in range(cls)]
        zeor_data[int(d)]=1
        all_data.append(zeor_data)
    return np.array(all_data)

# a=[[1,3,4],[0,1,3],[2,5,6]]
def one_hot2(data,cls):
    all_data=[]
    for dat in data:
        one_data=[]
        for d in dat:
            zeor_data = [x * 0 for x in range(cls)]
            zeor_data[int(d)] = 1
            one_data.append(zeor_data)
        all_data.append(one_data)

    return np.array(all_data)





def one_hot3(data,dim1,dim2):
    all_data=[]
    for d in data:
        new_data=[(x * 0+1)*7 for x in range(dim1)]
        for x in d:
            new_data[int(x)]=x
        one_data=[]
        for n in new_data:
            zeor_data = [x * 0 for x in range(dim2)]
            zeor_data[int(n)]=1
            one_data.append(zeor_data)
        all_data.append(one_data)

    return np.array(all_data)


# a=[[1,2,3,4],[1,5,6],[3,5],[7]]
def one_hot1(data,cls):
    all_data=[]
    for d in data:
        new_data = [x * 0 for x in range(cls)]
        if len(d)==1 and d[0]==7:
            all_data.append(new_data)
        else:
            new_data = [x * 0 for x in range(cls)]
            for i in d:
                new_data[i]=1
            all_data.append(new_data)
    return np.array(all_data)
























