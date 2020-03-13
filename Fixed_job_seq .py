# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:01:30 2019

@author: 將軍
"""
import pandas as pd
import numpy as np

job = ['b1','b2','b3','b4','b5','b6','b7','b8','b9','b10','b11','b12','b13','b14','b15','b16','b17','b18','b19','b20']
#       'b21','b22','b23','b24','b25','b26','b27','b28','b29','b30','b31','b32','b33','b34','b35','b36','b37','b38','b39','b40',
#       'b41','b42','b43','b44','b45','b46','b47','b48','b49','b50','b51','b52','b53','b54','b55','b56','b57','b58','b59','b60',
#       'b61','b62','b63','b64','b65','b66','b67','b68','b69','b70','b71','b72','b73','b74','b75','b76','b77','b78','b79','b80',
#       'b81','b82','b83','b84','b85','b86','b87','b88','b89','b90','b91','b92','b93','b94','b95','b96','b97','b98','b99','b100']

task = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15','a16','a17','a18','a19','a20']
#       'a21','a22','a23','a24','a25','a26','a27','a28','a29','a30','a31','a32','a33','a34','a35','a36','a37','a38','a39','a40',
#       'a41','a42','a43','a44','a45','a46','a47','a48','a49','a50','a51','a52','a53','a54','a55','a56','a57','a58','a59','a60',
#       'a61','a62','a63','a64','a65','a66','a67','a68','a69','a70','a71','a72','a73','a74','a75','a76','a77','a78','a79','a80',
#       'a81','a82','a83','a84','a85','a86','a87','a88','a89','a90','a91','a92','a93','a94','a95','a96','a97','a98','a99','a100']

f=open('Array.txt')
arr=f.read()
arr=arr.split('\n')
del arr[100]
for i in range(100):
    arr[i]=arr[i].split(',')
del arr[20:100]
for i in range(20):
    del arr[i][20:100]
    
arr1=[[row[col] for row in arr] for col in range(len(arr[0]))]  #讓job變成列
df = pd.DataFrame(arr1,index=job,columns=task)

#讀入時間
with open('data.txt','r') as dat:
     data = dat.readlines()

task_time = []
job_time = []

for i in range(1,21):
    task_time.append(float(data[i][:-1]))
task_df = pd.DataFrame(task_time,index=task)

for i in range(102,122):
    job_time.append(float(data[i][:-1]))
job_df = pd.DataFrame(job_time,index=job)   
    
    
#印SEQ 
minn=0  #記錄最小job completion time和
count2=0    #記錄現在第幾次迴圈
for g in range(2432902008176640000):
    print('第',g,'次測試')
    count2+=1
    df1=df.iloc[np.random.permutation(len(df))]
        

    seq = []
    for j in range(len(job)):
        count = 0                               #紀錄此job有幾個前置task
        for t in range(len(task)):
            if (df1.iloc[j][t]=='1'):              
                for r in range(j+1,len(job)):   #刪掉其他job與此job的聯集
                    if (df1.iloc[r][t]=='1'):
                        df1.iloc[r][t]='0'
                count=count+1
                seq.append(df1.columns[t])
        seq.append(df1.index[j])
    
    print(seq)
    
    #算時間 
    time=0      #記錄總completion time
    time2=0     #紀錄job的completion time
    
    for k in range(len(seq)):
        if (seq[k][0]=='a'):
            for j in range(len(task_df)):
                if (task_df.index[j]==seq[k]):
                    time=time+task_df.iloc[j]
        if (seq[k][0]=='b'):
            for j in range(len(job_df)):
                if (job_df.index[j]==seq[k]):
                    time=time+job_df.iloc[j]
                    time2=time2+time
    if (count2==1):     #一開始先帶第一次迴圈的ob completion time和
        minn=time2[0]
    else:
        if (minn>=time2[0]):
            minn=time2[0]

                        

                    
    
    
    print("總completion time",time[0])
    print("總job的completion time和",time2[0])

print('min_job_completion_time',minn)

        
    

