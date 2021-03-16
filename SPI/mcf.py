import math 
import matplotlib.pyplot as plt 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys
from matplotlib import cm
from PIL import Image
import time
import cmath
import random
import math
import cv2
import fourier_filter

start = time.time()
delta=1*(10**-6)
print("delta="+str(delta*1000*1000)+"µm")

z=1#propagation distance;m
print("伝搬距離z="+str(z)+"m")


# 画像サイズsize+1(pixel)
size=1023

M=size+1#space size
print("M="+str(M))

#####################################
# 計算には使ってないはず
lamda=1550*(10**-9)#wavelength,unit:m
print("λ="+str(lamda)+"m")

print("瞳面(入力面)の1ピクセルの大きさ(＝delta)＝"+str(delta*1000*1000)+"µm")
print("瞳面(入力面)の一辺の大きさ(＝delta*M)＝"+str(delta*M*1000*1000)+"µm")
################################

x = y = np.arange(-size/2, size/2+1, 1)
X, Y = np.meshgrid(x, y)


#print("radius="+str(radius*1000*1000)+"µm")
#print("MFD="+str(radius*2*1000*1000)+"µm")

#フレネル領域orフラウンホーファー領域
print("フレネル領域orフラウンホーファー領域")
if z>((10**-6)**2)/lamda:
    print("フラウンホーファー領域です")
else:
    print("!!!!!!フレネル領域です")
    sys.exit()
    
process_time = time.time() - start
print("process_time:"+str(process_time))

#####################################################


start = time.time()
pi=np.pi

P1=30
P2=50
Z=0
phase=0
# np.random.seed(1)
# rs=np.random.randint(-5,6,(2,14))
rs=np.zeros((2,14))


x = y = np.arange(-size/2, size/2+1, 1)
X, Y = np.meshgrid(x, y)

# ガウス分布の式、中心(X,Y),位相phase
# 光の複素振幅分布：Aexp(jΘ)←A：振幅、Θ:位相
def f(X,Y,phase,fluc):
    return fluc*np.exp(-(2*((X)**2 + (Y)**2)/(2**2)))*np.exp(1j*phase)


#fp1 = open('位相情報_7マルチコア（正六角形+中心）.csv', 'w')
#fp1.write("sita_0"+","+"sita_1"+","+"sita_2"+","+"sita_3"+","+"sita_4"+","+"sita_5"+","+"sita_6"+"\n")
   
#fp2 = open('出力強度分布配列4×4_7マルチコア（正六角形+中心）.csv', 'w')
#writer = csv.writer(fp2, lineterminator='\python')
#fp2.write("sita_0"+","+"sita_1"+","+"sita_2"+","+"sita_3"+","+"sita_4"+","+"sita_5"+","+"sita_6"+"\n")
          

# tがパターン数
for t in range(0,300):

    random.seed(t)
    sita_0=random.uniform(-pi, pi)
    sita_1=random.uniform(-pi, pi)
    sita_2=random.uniform(-pi, pi)
    sita_3=random.uniform(-pi, pi)
    sita_4=random.uniform(-pi, pi)
    sita_5=random.uniform(-pi, pi)
    sita_6=random.uniform(-pi, pi)
    sita_7=random.uniform(-pi, pi)
    sita_8=random.uniform(-pi, pi)
    sita_9=random.uniform(-pi, pi)
    sita_10=random.uniform(-pi, pi)
    sita_11=random.uniform(-pi, pi)
    sita_12=random.uniform(-pi, pi)
    sita_13=random.uniform(-pi, pi)

    #fp1.write(str(sita)+"\n")
    #print(sita)
    sita=[sita_0,sita_1,sita_2,sita_3,sita_4,sita_5,sita_6,sita_7,sita_8,sita_9,
        sita_10,sita_11,sita_12,sita_13]
    
    #fluc=np.random.uniform(0.8,1.2,(14,1))
    fluc=np.ones((14,1))

    Z=0
    
    Z=  f(X                           +rs[0,0],Y+P1                        +rs[1,0],sita[0],fluc[0])
    Z=Z+f(X+P1*math.sin(2/7*pi)       +rs[0,1],Y+P1*math.cos(2/7*pi)       +rs[1,1],sita[1],fluc[1])
    Z=Z+f(X+P1*math.cos(4/7*pi-1/2*pi)+rs[0,2],Y-P1*math.sin(4/7*pi-1/2*pi)+rs[1,2],sita[2],fluc[2])
    Z=Z+f(X+P1*math.sin(1/7*pi)       +rs[0,3],Y-P1*math.cos(1/7*pi)       +rs[1,3],sita[3],fluc[3])
    Z=Z+f(X-P1*math.sin(1/7*pi)       +rs[0,4],Y-P1*math.cos(1/7*pi)       +rs[1,4],sita[4],fluc[4])
    Z=Z+f(X-P1*math.cos(4/7*pi-1/2*pi)+rs[0,5],Y-P1*math.sin(4/7*pi-1/2*pi)+rs[1,5],sita[5],fluc[5])
    Z=Z+f(X-P1*math.sin(2/7*pi)       +rs[0,6],Y+P1*math.cos(2/7*pi)       +rs[1,6],sita[6],fluc[6])
    Z=Z+f(X+P2*math.sin(1/7*pi)       +rs[0,7],Y+P2*math.cos(1/7*pi)       +rs[1,7],sita[7],fluc[7])
    Z=Z+f(X+P2*math.cos(4/7*pi-1/2*pi)+rs[0,8],Y+P2*math.sin(4/7*pi-1/2*pi)+rs[1,8],sita[8],fluc[8])
    Z=Z+f(X+P2*math.sin(2/7*pi)       +rs[0,9],Y-P2*math.cos(2/7*pi)       +rs[1,9],sita[9],fluc[9])
    Z=Z+f(X                           +rs[0,10],Y-P2                       +rs[1,10],sita[10],fluc[10])
    Z=Z+f(X-P2*math.sin(2/7*pi)       +rs[0,11],Y-P2*math.cos(2/7*pi)      +rs[1,11],sita[11],fluc[11])
    Z=Z+f(X-P2*math.cos(4/7*pi-1/2*pi)+rs[0,12],Y+P2*math.sin(4/7*pi-1/2*pi)+rs[1,12],sita[12],fluc[12])
    Z=Z+f(X-P2*math.sin(1/7*pi)       +rs[0,13],Y+P2*math.cos(1/7*pi)      +rs[1,13],sita[13],fluc[13])
    #pattern14_2
    # Z=Z+f(X-50+rs[0,0] ,Y   +rs[1,0],sita[0])
    # Z=Z+f(X-17+rs[0,1] ,Y   +rs[1,1],sita[1])
    # Z=Z+f(X+17+rs[0,2] ,Y   +rs[1,2],sita[2])
    # Z=Z+f(X+50+rs[0,3] ,Y   +rs[1,3],sita[3])
    # Z=Z+f(X-33+rs[0,4] ,Y+25+rs[1,4],sita[4])
    # Z=Z+f(X   +rs[0,5] ,Y+25+rs[1,5],sita[5])
    # Z=Z+f(X+33+rs[0,6] ,Y+25+rs[1,6],sita[6])
    # Z=Z+f(X-17+rs[0,7] ,Y+50+rs[1,7],sita[7])
    # Z=Z+f(X+17+rs[0,8] ,Y+50+rs[1,8],sita[8])
    # Z=Z+f(X-33+rs[0,9] ,Y-25+rs[1,9],sita[9])
    # Z=Z+f(X   +rs[0,10],Y-25+rs[1,10],sita[10])
    # Z=Z+f(X+33+rs[0,11],Y-25+rs[1,11],sita[11])
    # Z=Z+f(X-17+rs[0,12],Y-50+rs[1,12],sita[12])
    # Z=Z+f(X+17+rs[0,13],Y-50+rs[1,13],sita[13])
    # Z=Z/np.max(abs(Z))

    #print(Z.shape)
    #画像をarrayに変換
    core1 = np.asarray(Z)
    #print(core1.shape)

    if t<1:

        # 入力面表示するなら
        # 強度分布
        w=h=150
        x=y=(size+1-w)/2
        x=math.floor(x)#切り上げて整数
        y=math.floor(y)
        
        input_intensity_kakudai = core1[y: y+h, x: x+w]
        plt.figure(30)
        plt.title("input kakudai")
        plt.imshow(abs(input_intensity_kakudai), cmap = 'hot')
        plt.colorbar()
        #plt.savefig('入力面強度分布.png')
        # plt.show()
        center=size/2+1
        center=math.floor(center)#切り下げして整数
        #print(type(center))
        #print("center="+str(center))
        #print("ビーム間距離(＝1ピクセルの大きさ×中心間ピクセル)＝"+str(delta*1000*1000*P)+"µm")#水平方向のみ、ななめとなりは考慮できていない

        # 位相分布
        phase_input=np.angle(core1)
        phase_input_kakudai=phase_input[y: y+h, x: x+w]
        # aaaaa=input_intensity_kakudai.imag/input_intensity_kakudai.real
        # phase_input=math.atan(aaaaa)
        plt.figure(31)
        plt.title("phase input ")
        plt.imshow(phase_input, cmap = 'jet')
        plt.colorbar()
        plt.figure(32)
        plt.title("phase input kakudai")
        plt.imshow(phase_input_kakudai, cmap = 'jet')
        plt.colorbar()
        # plt.show()
        
        core1_phase2=np.zeros((size+1,size+1))
        for i in range (0,size+1):
            for j in range (0,size+1):
                if abs(core1[i,j])>=1/np.e:
                    core1_phase2[i,j]=cmath.phase(core1[i,j])
        plt.figure(33)
        plt.title("phase input(MFD no uchigawadake)")
        plt.imshow(core1_phase2, cmap = 'jet')
        #表示
        plt.colorbar()
        #plt.savefig('位相分布2-5.png')
        # plt.show()

        input_phase_kakudai = core1_phase2[y: y+h, x: x+w]
        plt.figure(34)
        plt.title("phase input kakudai(MFD no uchigawadake)t=%d"%t)
        plt.imshow(input_phase_kakudai, cmap = 'jet', vmin = -pi, vmax = pi, interpolation = 'none')
        plt.colorbar()
        #plt.savefig('入力面位相分布.png')
        plt.show()

        core_rayout=np.ones((size+1,size+1))
        for i in range (0,size+1):
            for j in range (0,size+1):
                if abs(core1[i,j])>=0.135:
                    core_rayout[i,j]=0.6
        w=h=250
        x=y=(size+1-w)/2
        x=math.floor(x)#切り上げて整数
        y=math.floor(y)
        core_rayout2=core_rayout[y:y+w,x:x+h]
        plt.figure(34)
        plt.imshow(core_rayout2, cmap = 'gray',vmin=0,vmax=1)
        plt.show()
    
        
    # フラウンホーファー回折計算(入力面のフーリエ変換)
    #core2=fourier_filter.ZeroPadding(core1)
    Dif=np.fft.fftshift(np.fft.fft2((core1)))

    # 強度は振幅の2乗
    #Intensity=fourier_filter.highpass(Dif,10)
    #Intensity=(abs(Dif)**2)
    #Dif2=fourier_filter.filter_circle(Dif,500,0)
    #img_ifft = np.fft.ifftshift(Dif2)   
    #img_back = np.fft.ifft2(img_ifft)
    #img_back2 = np.fft.ifftshift(img_back)
    Intensity=abs(Dif)**2

    # トリミングして保存、トリミング範囲は考える必要がある
    w=h=128
    x=y=(size+1-w)/2
    x=math.floor(x)#切り上げて整数
    y=math.floor(y)

    tori2 = Intensity[y: y+h, x: x+w]
    
    if t<3:
        #plt.title("パターン16×16（t=%d）"%t)
        plt.figure(50)
        #plt.title("出力面強度分布（4ピクセル×4ピクセル）_入力面256×256(t=%d）"%t)
        plt.imshow(tori2, cmap = 'gray')
        plt.colorbar()
        #plt.savefig('パターン16×16（t=%d）.png'%t)
        #plt.savefig('出力面強度分布（4ピクセル×4ピクセル）_入力面256×256(t=%d）.png'%t)
        plt.show()

        #print("switch="+str(switch))
    
    
    #np.savetxt('C:/Users/ryuta/研究室/プログラム/SPI-MCF/pattern14.random1_big/%d.csv'%t, tori2, delimiter=",")

    if t%20==0:
        print(t)
    
#fp1.close()
#fp2.close()
#process_time = time.time() - start
#print("process_time:"+str(process_time))


