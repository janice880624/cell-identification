import numpy as np
import cv2
import math
import os 
import time

from numpy import linalg as la
# from PIL import Image 

start = time.process_time() # 計時開始
#%%
""" ----------載入圖片----------"""
def Loadimg(img_path):
    if img_path.any():
        img_lab = cv2.cvtColor(img_path, cv2.COLOR_BGR2LAB)
        # cv2.namedWindow('input', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('input', img)
        print('img.shape: ', img_lab.shape) # numpy中的shape，是指矩陣中的（行，列），對應圖片就是（高，寬）
        B, G, R = cv2.split(img_lab)   # 分離img之三通道
        L = np.reshape(B, (-1, 1)) # 經由.reshape(列, 行)，化成N列1行
        A = np.reshape(G, (-1, 1)) # 經由.reshape(列, 行)，化成N列1行
        B = np.reshape(R, (-1, 1)) # 經由.reshape(列, 行)，化成N列1行
        return L, A, B
    else:
        print('Error: No Image')
#%% 
""" ----------最小平方法----------"""
def LS(L, A, B):
    L_rows = np.size(L, 0)       # 計算矩陣L的行數
    one = np.ones([L_rows, 1])   # L_rows*1的矩陣
    MatA = np.hstack([L, A, one])  
    Ans = np.linalg.lstsq(MatA, B, rcond = -1) # np.linalg.lstsq --> 最小平方法，也可用以下的SVD直接得值
    Ans = Ans[0]
    Ans = Ans.transpose()
    # -----SVD奇異值分解-----
    # u, s, vh = np.linalg.svd(MatA, full_matrices=False, compute_uv=True, hermitian=False)
    # d = np.diag(s)
    # u = u[:, :d.shape[0]]
    # U, sigma, VT = np.linalg.svd(MatA, full_matrices=False, compute_uv=True, hermitian=False) # 基於降低SVD的重建
    # Ans = np.dot(np.dot(np.dot(vh.T, np.linalg.inv(d)), u.T), B)
    return Ans
""" ----------旋轉矩陣----------"""
# 經過LS計算法向量與 z 軸的夾角，然後求出旋轉矩陣，將旋轉向量轉換成3x3矩陣
# 然後把點旋轉後，就會對齊座標系，再來只要將三維點的z值直接壓成0，這樣就投影到了LS的平面上
def Rotation(VecA):
    Eye = np.eye(3)                           # 宣告單位矩陣
    VecZ = np.array([0, 0, 1])
    W = np.cross(VecZ, VecA)                  # 後面會輸入VecA，W = VecZ 外積 VecA
    # print('W: \n', W)
    W_norm = la.norm(W)                       # 計算W的norm(範數)
    W = W / W_norm                            # W的Nomalized
    W_skew = np.array([(   0    ,  -W[0][2],   W[0][1]), # W_skew為W的反矩陣
                       ( W[0][2],     0    ,  -W[0][0]),
                       (-W[0][1],   W[0][0],        0)])
    VecA_norm = la.norm(VecA)       # 計算VecA的norm(範數)
    VecZ_norm = la.norm(VecZ)       # 計算VecZ的norm(範數)
    angle = VecZ.transpose() * VecA / VecZ_norm / VecA_norm
    tht = math.acos(angle[0][2])        # 問題，0是甚麼??
    W_skew2 = np.dot(W_skew, W_skew)
    R = Eye + (W_skew * (math.sin(tht))) + (W_skew2 * (1 - math.cos(tht)))
    return R   
""" ----------投影到2D平面上----------"""
def Project(L, A, B, R):
    # L_rows = np.size(L, 0)
    Points = np.hstack([L, A, B])
    NewP = np.dot(Points, R)
    P2D = np.vstack([NewP[:,0], NewP[:,1]]) # P2D為NewP的第一行元素加上第二行元素
    P2D = np.transpose(P2D)
    return P2D  
""" ----------在平面上等分切網格----------"""
def Cutgrids(inputt, Num):
    Grids = np.zeros([Num, Num])
    input_rows = np.size(inputt, 0)
    GridIndex = np.zeros([input_rows, 2])
    MaxVal = np.max(inputt, axis = 0)  # 提出每一行最大值
    MinVal = np.min(inputt, axis = 0)  # 提出每一行最小值
    PadX = (MaxVal[0] - MinVal[0]) / Num;
    PadY = (MaxVal[1] - MinVal[1]) / Num;
    
    for i in range(input_rows):
        X = max((int(math.ceil(((inputt[i][0] - MinVal[0]) / PadX))) - 1), 0) # math.ceil返回不小於輸入值的整數
        Y = max((int(math.ceil(((inputt[i][1] - MinVal[1]) / PadY))) - 1), 0)
        Grids[X][Y] += 1
        GridIndex[i][0] = X
        GridIndex[i][1] = Y
    return Grids, GridIndex
""" ----------分群中心----------"""
def ClusterCenter(Grid, Grididx, L, A, B, Num):
    CPoint = np.zeros([Num, 3])
    Gvec = np.reshape(Grid, (-1, 1))
    Gsize = np.size(Grid, 0)
    GSort = sorted(range(len(Gvec)), key = lambda k : Gvec[k], reverse = True)
    Grididx_rows, Grididx_cols = Grididx.shape
    X = int()
    Y = int()
    if (Num <= cv2.countNonZero(Gvec)):
        for i in range(Num):
            k = 0
            P2Didx = np.zeros([int(Gvec[GSort[i]]), 3])
            X = int((GSort[i]) / Gsize)
            Y = int((GSort[i]) - (X * (Gsize)))
            for j in range(Grididx_rows):
                eigXY = np.zeros(shape = Grididx_cols)
                eigXY[0] = Grididx[j][0]
                eigXY[1] = Grididx[j][1]
                if(eigXY[0] == X and eigXY[1] == Y):
                    # P2Didx[k] = np.hstack([L[j], A[j], B[j]])
                    P2Didx[k][0] = L[j]
                    P2Didx[k][1] = A[j]
                    P2Didx[k][2] = B[j]
                    k += 1 
            CPoint[i] = np.mean(P2Didx, axis = 0)
        return CPoint
    else:
        print('Error: Max Cluster Number Error\n')
        exit(-1)     
""" ----------離群中心----------"""
def Belonging(Center, L, A, B, ver):
    L_rows = np.size(L, 0)
    Center_rows = np.size(Center, 0)
    Data = np.hstack([L, A, B])
    DisData = np.zeros([L_rows, Center_rows])
    DisData_rows, DisData_cols = DisData.shape
    vec = np.zeros([L_rows, 3])
    dis = np.zeros([L_rows, 1])
    for i in range(Center_rows):
        if ver == 1:
            vec = (-1)*Data + Center[i]
            vec_rows, vec_cols = vec.shape
            for j in range(vec_rows):
                dis[j] = la.norm(vec[j])
            dis_max = np.max(dis, axis = 0)
            dis_min = np.min(dis, axis = 0)
            dis_diff = dis_max - dis_min
            NorDis = (dis - dis_min) / dis_diff # 0.0981597
            for k in range(vec_rows):
                DisData[k][i] = NorDis[k]
            
        if ver == 2:
            vec = -Data - Center[i]
            NorDis = 1 / dis
            DisData[i] = NorDis
        if ver == 3:
            vec = -Data - Center[i]
            dis = np.linalg.norm(ver[i], ord = 2)
            DisData[i] = dis
            
    if ver == 2:
        SumDis = np.sum(DisData, axis = 1) # axis = 1，每一列
        SumDis = SumDis.reshape(DisData_rows, 1)
        Temp = DisData / SumDis
        DisData = Temp
        
    if ver == 3:
        Ones = np.zeros([DisData_rows, DisData_cols])
        SumDis = np.sum(DisData, axis = 1)
        SumDis = SumDis.reshape(DisData_rows, 1)
        SumDis = np.repeat(SumDis, DisData_cols, axis=1)
        Temp = Ones - (DisData / SumDis)
        DisData = Temp
        
    return DisData      
""" ----------根據群心移動----------"""
def Moving(alpha, Center, belong, L, A, B):
    L_rows = np.size(L, 0)
    MoveDis = np.zeros([L_rows, 3])
    Data = np.hstack([L, A, B])
    Center_rows = np.size(Center, 0)
    for i in range(Center_rows):
        dis = np.zeros([L_rows, 3])
        Belong = np.zeros([L_rows, 3])
        vec = Data - Center[i]
        Tempbelong = belong[:,i]
        Belong = np.vstack([Tempbelong, Tempbelong, Tempbelong])
        Belong = Belong.transpose()
        Move = alpha * vec * Belong
        MoveDis = MoveDis + Move
    out = Data + MoveDis
    return out
#%%
""" ----------輸出----------"""
def OutPut(Data, original, color):
    # Data_rows = np.size(Data, 0)   
    Temp = Data.copy()
    original_rows, original_cols, original_channels = original.shape
    Lab = np.reshape(Temp, (original_rows, original_cols, 3))
    Lab2 = Lab.copy()
    Lab2[Lab2 < 0] = 0
    Lab2[Lab2 > 255] = 255
    Lab2 = np.around(Lab2)
    Lab_char = Lab2.astype('uint8')
    rgb = cv2.cvtColor(Lab_char, cv2.COLOR_LAB2BGR)
    B, G, R = cv2.split(rgb)   # 分離img之三通道
    Lab3ch_B = np.reshape(B, (-1, 1)) # 經由.reshape(列, 行)，化成N列1行
    Lab3ch_G = np.reshape(G, (-1, 1)) # 經由.reshape(列, 行)，化成N列1行
    Lab3ch_R = np.reshape(R, (-1, 1)) # 經由.reshape(列, 行)，化成N列1行
    color = np.hstack([Lab3ch_B, Lab3ch_G, Lab3ch_R])
    return rgb
#%%
""" ----------熵----------"""
def Entropy(Data):
    entropy = float(0)
    Cube = np.zeros([255*255*255, 1]) 
    Data_array = np.reshape(Data, (-1, 1)) # 經由.reshape(列, 行)，化成N列1行
    Data_array = np.reshape(Data_array, (3, -1))
    Data_array = Data_array.transpose()
    Data_array_rows, Data_array_cols = Data_array.shape
    for i in range(Data_array_rows):
        Cube[Data_array[i][0] + 255 * (max((Data_array[i][1] - 1), 0)) + 65025 * (max((Data_array[i][2] - 1), 0))] += 1
    Cube = np.int64(Cube)
    Cube_1 = Cube / 16581375
    Cube_1_rows, Cube_1_cols = Cube_1.shape
    for j in range(Cube_1_rows):
        if(Cube_1[j] > 0):
            entropy = entropy - Cube_1[j] * math.log(Cube_1[j], 2)
    return entropy

#%%
def entropy_cal(input_img):
    result = 0
    resultCH = 0
    img_rows, img_cols, img_channels = input_img.shape
    if(img_channels == 1):
        temp = np.zeros([256, 1])
        for i in range(256):
            temp[i] = 0.0
        for j in range(len(img_rows)):
            
            for k in range(len(img_cols)):
                # i = int(t[k])
                temp[i] = temp[i] + 1
        for l in range(256):
            temp[i] = temp[i] / (img_rows * img_cols)
        for m in range(256):
            if(temp[i] == 0.0):
                result = result
            else:
                result = result - (temp[i] * (math.log(temp[i], 2)))
    else:
        # B, G, R = cv2.split(img)   # 分離img之三通道
        for i in range(3): 
            imgCh = input_img.copy()
            imgCh2 = input_img[:,:,i]
            imgCh_rows, imgCh_cols, imgCh_channels = imgCh.shape
            t = np.zeros([imgCh_rows, imgCh_cols])
            temp = np.zeros([256, 1])
            for j in range(256):
                temp[j] = 0.0            
            for k in range(imgCh_rows): # 400
                for l in range(imgCh_cols): # 600
                    t[k][l] = imgCh2[k][l]
                    num = int(t[k][l])
                    temp[num] += 1
            for m in range(256):
                temp2 = temp.copy()
                temp2 = temp2 / (imgCh_rows*imgCh_cols)
            resultCH = 0
            for n in range(256):
                if temp2[n] == 0:
                    resultCH = resultCH
                else:
                    resultCH = resultCH - (temp2[n] * (math.log(temp2[n], 2)))
            result = result + resultCH  
        result = result / 3 # 5.35394
    return result

#%%
""" ----------FACE----------"""
def FACE(image, alpha, ClusterNum, GridSize, ExplosionWay, opt, Per, Number, iterNum, entropySetSize, 
         entropyNumOfSets):
    L, A, B = Loadimg(image)
    L_rows, L_cols = L.shape
    normal = LS(L, A, B)
    R = Rotation(normal)
    p2d = Project(L, A, B, R)
    Color = np.array([L_rows, 3])
    Grids, Grididx = Cutgrids(p2d, GridSize)
    Centers = ClusterCenter(Grids, Grididx, L, A, B, ClusterNum)
    Belong = Belonging(Centers, L, A, B, ExplosionWay)
    Result = Moving(0, Centers, Belong, L, A, B)
    boundertotal = int(np.count_nonzero(Result == 255) + np.count_nonzero(Result == 0))
    E = list()
    E.clear()
    percent = list()
    entropySetAvgSet = list()
    OutSet = list()
    entropySet = np.zeros([entropySetSize])
    entropyIndex = int(0)
        
    for i in range(iterNum):
        Result = Moving(i/100, Centers, Belong, L, A, B)
        Out = OutPut(Result, image, Color)
        ReturnOut = Out
        E.append(Entropy(Out))
        OP = int(np.count_nonzero(Result > 255))
        ON = int(np.count_nonzero(Result <  0 ))
        # print('i', i,'OP:', OP, 'ON:', ON)
        Result_rows, Result_cols = Result.shape
        percent.append(((OP + ON - boundertotal) / (Result_rows * Result_cols)) * 100)
        print("Iter:", i, " ", "E:", E[len(E) - 1], " ", percent[len(percent) - 1], "%") 
            # Iter : 0 E : 0.294583 -0.21375%
            # Entropy set average = 0.356929
            # Iter : 1 E : 0.287264 2.26847%
            # Iter : 2 E : 0.286135 6.73542%
            # Iter : 3 E : 0.281373 8.97167%
        entropySet[entropyIndex] = entropy_cal(Out)
        entropyIndex += 1
        if(i % entropySetSize == 0):
            entropySetAvg = 0
            for k in range(len(entropySet)):
                entropySetAvg = entropySetAvg + entropySet[k]
            entropySetAvg = entropySetAvg / 15
            print("Entropy set average =", entropySetAvg)
            entropySetAvgSet.append(entropySetAvg)
            OutSet.append(Out)
                    
            if(len(entropySetAvgSet) > 3):          
                entropySetAvgSet.pop(0)
                OutSet.pop(0)
                entropyDecreasing = False
                for j in range(1, 0, -1):
                    if(entropySetAvgSet[j+1] < entropySetAvgSet[j]):
                        entropyDecreasing = True
                    else:
                        entropyDecreasing = False
                        break
                    
                if entropyDecreasing:
                    ReturnOut = OutSet[0]
                    break  
            entropyIndex = 0
                
        if (percent[len(percent) - 1] >= 15):
            break
        
    return ReturnOut

def FACE_alpha(image, alpha, ClusterNum, GridSize, ExplosionWay, opt, Per, Number):
    L, A, B = Loadimg(image)
    L_rows, L_cols = L.shape
    normal = LS(L, A, B)
    R = Rotation(normal)
    p2d = Project(L, A, B, R)
    Color = np.array([L_rows, 3])
    Grids, Grididx = Cutgrids(p2d, GridSize)
    Centers = ClusterCenter(Grids, Grididx, L, A, B, ClusterNum)
    Belong = Belonging(Centers, L, A, B, ExplosionWay)

    E = list()
    E.clear()
    percent = list()
    
    Result = Moving(0, Centers, Belong, L, A, B)
    boundertotal = int(np.count_nonzero(Result == 255) + np.count_nonzero(Result == 0))
    
    Result = Moving(alpha, Centers, Belong, L, A, B)
    Out = OutPut(Result, image, Color)
    
    E.append(Entropy(Out))
    OP = int(np.count_nonzero(Result > 255))
    ON = int(np.count_nonzero(Result <  0 ))
    Result_rows, Result_cols = Result.shape
    percent.append(((OP + ON - boundertotal) / (Result_rows * Result_cols)) * 100)
    print("E: " + E[len(E) - 1] + " " + percent[len(percent) - 1] + "%\n") 
    return Out

#%%
# inputPath = 'C:/Users/yang2/Desktop/Project10/_01.png/' # input資料夾位置
# savePath = 'C:/Users/yang2/Desktop/123/' # output資料夾位置
inputPath = 'img/' # input資料夾位置
savePath = 'img/image/' # output資料夾位置

inputList = os.listdir(inputPath) 
saveList = os.listdir(savePath)

image = cv2.imread(inputPath, cv2.IMREAD_COLOR)
a = 0
Mat_out = FACE(image, a, 3, 3, 1, True, 15, 0, 1000, 15, 3)
cv2.imwrite(savePath + str(1) + "_out.png", Mat_out)
# for file in os.listdir(inputPath):
#     image = cv2.imread(os.path.join(inputPath, file), cv2.IMREAD_COLOR)
#     a = 0
#     Mat_out = FACE(image, a, 3, 3, 1, True, 15, 0, 1000, 15, 3)
#     cv2.imwrite(savePath + str(inputList.index(file)+1) + "_out.jpg", Mat_out)

#%%
end = time.process_time()
diff = end - start
print("執行時間", diff, 'sec')