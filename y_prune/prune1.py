import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck, Conv, C2f, SPPF, Detect
import numpy as np

def prune_conv(conv1,conv2,threshold):
    gamma=conv1.bn.weight.data.detach()
    beta=conv1.bn.bias.detach()
    keep_idxs=[]
    local_threshold = threshold
    while(len(keep_idxs)<8):
        keep_idx = torch.where(gamma.abs()>=local_threshold)[0]
        keep_idx = torch.ceil(torch.tensor(len(keep_idx)/8))*8 #为保证最后的通道式是8的倍数
        new_threashold = torch.sort(gamma.abs(),descending=True)[0][int(keep_idx-1)]
        keep_idxs = torch.where(gamma.abs()>=new_threashold)[0] #得到剪枝后通道的index
        local_threshold = new_threashold*0.5
    n = len(keep_idxs)
    print("prune rate for this layer: {%.2f} %",n/len(gamma) * 100)
    conv1.bn.weight.data = gamma[keep_idxs] #对conv1的bn层进行调整
    conv1.bn.bias.data = beta[keep_idxs]
    conv1.bn.running_var.data = conv1.bn.running_var.data[keep_idxs]
    conv1.bn.running_mean.data = conv1.bn.running_mean.data[keep_idxs]
    conv1.bn.num_features = n
    conv1.conv.weight.data = conv1.conv.weight.data[keep_idxs] #对conv1的conv层进行调整
    conv1.conv.out_channels = n
    if conv1.conv.bias is not None:
        conv1.conv.bias.data = conv1.conv.bias.data[keep_idxs]
    if not isinstance(conv2,list): #对conv2层进行处理，转换为list，此处统一为List是因为后续会传入list（1对多）
        conv2 = [conv2] 
    for item in conv2:
        if item is not None:
            if isinstance(item,Conv): #找到conv2中的conv层并对其输入通道进行调整
                conv = item.conv
            else:
                conv = item
            conv.in_channels = n #conv2层的输入通道数量调整到与剪枝后conv1层的输出通道一致
            conv.weight.data = conv.weight.data[:,keep_idxs]

def prune(m1,m2,threshold):
    if isinstance(m1,C2f): 
        m1 = m1.cv2
    if not isinstance(m2,list):
        m2 = [m2]
    for i, item in enumerate(m2):
        if isinstance(item,C2f) or isinstance(item,SPPF):
            m2[i] = item.cv1
    prune_conv(m1,m2,threshold)

if __name__ == "__main__":
    yolo = YOLO("weights/best.pt")  # build a new model from scratch
    model = yolo.model
    ws = []
    bs = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            w = m.weight.abs().detach()
            b = m.bias.abs().detach()
            ws.append(w)
            bs.append(b)
            print(name, w.max().item(), w.min().item(), b.max().item(), b.min().item())
    factor = 0.8 #剪枝率暂时设置为0.8，推荐选用小的剪枝率进行多次剪枝操作
    ws = torch.cat(ws)
    threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]

    seq = model.model
    #针对head部分，low level的先不要减，可能包含有重要的信息
    for i in range(3, 9):
        if i in [6, 4, 9]: 
            continue
        prune(seq[i], seq[i+1],threshold)
    #对15,18,21层处理，15和18不仅和detect层相连，同时下个layer中也有conv层，但是21层是和Detect层直连
    for n,i in enumerate([15,18,21]):
        if(i!=21):
            prune(seq[i],[seq[i+1],seq[-1].cv2[n][0],seq[-1].cv3[n][0]],threshold) #C2f-conv,C2f-Detect
        prune_conv(seq[-1].cv2[n][0],seq[-1].cv2[n][1],threshold)        #Detect.cv2
        prune_conv(seq[-1].cv2[n][1],seq[-1].cv2[n][2],threshold)        #Detect.cv2
        prune_conv(seq[-1].cv3[n][0],seq[-1].cv3[n][1],threshold)        #Detect.cv3
        prune_conv(seq[-1].cv3[n][1],seq[-1].cv3[n][2],threshold)        #Detect.cv3
    #遍历模型并对bottleneck内的剪枝 
    for name, m in model.named_modules():                
        if isinstance(m, Bottleneck):                          
            prune_conv(m.cv1, m.cv2,threshold)
    for name, p in yolo.model.named_parameters():
        p.requires_grad = True
    yolo.train(data="ultralytics/cfg/datasets/VOC.yaml", epochs=300,workers=8,batch=32)
    print("done")
