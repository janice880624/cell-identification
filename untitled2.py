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
        # print('img.shape: ', img_lab.shape) # numpy中的shape，是指矩陣中的（行，列），對應圖片就是（高，寬）
        B, G, R = cv2.split(img_lab)   # 分離img之三通道
        L = np.reshape(B, (-1, 1)) # 經由.reshape(列, 行)，化成N列1行
        # print('L: ', L)
        A = np.reshape(G, (-1, 1)) # 經由.reshape(列, 行)，化成N列1行
        # print('A: ', A)
        B = np.reshape(R, (-1, 1)) # 經由.reshape(列, 行)，化成N列1行
        # print('B: ', B)
        return L, A, B
    else:
        print('Error: No Image')
#%% 
""" ----------最小平方法----------"""
def LS(L, A, B):
    L_rows = np.size(L, 0)     # 計算矩陣L的行數
    one = np.ones([L_rows, 1])   # L_rows*1的矩陣
    MatA = np.hstack([L, A, one]) 
    Ans = np.linalg.lstsq(MatA, B, rcond = -1) # np.linalg.lstsq --> 最小平方法，也可用以下的SVD直接得值
    print('Ans: \n', Ans)        
    Ans = Ans[0]
    print('Ans: \n', Ans)   
    Ans = Ans.transpose()
    print('Ans: \n', Ans)   
    # -----SVD奇異值分解-----
    # u, s, vh = np.linalg.svd(MatA, full_matrices=False, compute_uv=True, hermitian=False)
    # d = np.diag(s)
    # u = u[:, :d.shape[0]]
    # U, sigma, VT = np.linalg.svd(MatA, full_matrices=False, compute_uv=True, hermitian=False) # 基於降低SVD的重建
    # Ans = np.dot(np.dot(np.dot(vh.T, np.linalg.inv(d)), u.T), B)
    return Ans
"""--------------------------------------------------------------"""
# 經過LS計算法向量與 z 軸的夾角，然後求出旋轉矩陣，將旋轉向量轉換成3x3矩陣
# 然後把點旋轉後，就會對齊座標系，再來只要將三維點的z值直接壓成0，這樣就投影到了LS的平面上
def Rotation(VecA):
    Eye = np.eye(3)                           # 宣告單位矩陣
    VecZ = np.array([0, 0, 1])
    W = np.cross(VecZ, VecA)                  # 後面會輸入VecA，W = VecZ 外積 VecA
    # print('W: \n', W)
    W_norm = la.norm(W)                       # 計算W的norm(範數)
    W = W / W_norm    
    print('W: \n', W)                        # W的Nomalized
    W_skew = np.array([(   0    ,  -W[0][2],   W[0][1]), # W_skew為W的反矩陣
                        ( W[0][2],     0    ,  -W[0][0]),
                        (-W[0][1],   W[0][0],        0)])
    VecA_norm = la.norm(VecA)       # 計算VecA的norm(範數)
    VecZ_norm = la.norm(VecZ)       # 計算VecZ的norm(範數)
    angle = VecZ.transpose() * VecA / VecZ_norm / VecA_norm
    print('angle: \n', angle)
    tht = math.acos(angle[0][2])        # 問題，0是甚麼??
    W_skew2 = np.dot(W_skew, W_skew)
    R = Eye + (W_skew * (math.sin(tht))) + (W_skew2 * (1 - math.cos(tht)))
    return R   
"""-----------------------------------------------------------------"""
def Project(L, A, B, R):
    # L_rows = np.size(L, 0)
    Points = np.hstack([L, A, B])
    print('point: \n', Points)
    NewP = np.dot(Points, R)
    P2D = np.vstack([NewP[:,0], NewP[:,1]]) # P2D為NewP的第一行元素加上第二行元素
    P2D = np.transpose(P2D)
    return P2D  
#%%
inputPath = '01.png' # input資料夾位置

# inputList = os.listdir(inputPath) 
image = cv2.imread(inputPath, cv2.IMREAD_COLOR)
L, A, B = Loadimg(image)
Ans = LS(L, A, B) 
normal = LS(L, A, B)
R = Rotation(normal)
p2d = Project(L, A, B, R)
print('ans:',Ans)
print('rot:',R)
a = 0


#%%
end = time.process_time()
diff = end - start
print("執行時間", diff, 'sec')