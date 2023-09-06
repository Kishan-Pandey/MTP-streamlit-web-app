## Importing Libraries
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2

import os
import pickle
import filetype
from math import sqrt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset,DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

## Initializing App
inputImage = st.sidebar.file_uploader("Upload image:")
laneLength = st.sidebar.number_input('Insert Lane Length', value=8)
laneWidth  = st.sidebar.number_input('Insert Lane Width',  value=3.5)
points = []

st.header("📈 Demo Application")
st.write("1. Upload Pavement Image or Video")
st.write("2. Select the Referance and input the dimensions")
st.write("3. Get the Results")

## Defining the UNet model and helper functions and classes 
def Conv3(in_c,out_c):
    Conv=nn.Sequential(nn.Conv2d(in_c,out_c,kernel_size=3,stride=1,padding=1),
                        nn.LeakyReLU(inplace=True),
                        nn.BatchNorm2d(out_c),
                        nn.Conv2d(out_c,out_c,kernel_size=3,stride=1,padding=1),
                        nn.LeakyReLU(inplace=True),
                        nn.BatchNorm2d(out_c)
                        )
    return Conv

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.MaxPoll1=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.down_conv1=Conv3(3,32)
        self.down_conv2=Conv3(32,64)
        self.down_conv3=Conv3(64,128)
        self.down_conv4=Conv3(128,256)
        self.down_conv5=Conv3(256,512)
        self.ConvTraspose1=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2,padding=0)
        self.up_sample1=Conv3(512,256)
        self.ConvTraspose2=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2,padding=0)
        self.up_sample2=Conv3(256,128)
        self.ConvTraspose3=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2,padding=0)
        self.up_sample3=Conv3(128,64)
        self.ConvTraspose4=nn.ConvTranspose2d(64,32,kernel_size=2,stride=2,padding=0)
        self.up_sample4=Conv3(64,32)
        self.out_con2d=nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)
    def forward(self,inputs):
        layer1=self.down_conv1(inputs)
        layer2=self.MaxPoll1(layer1)
        layer3=self.down_conv2(layer2)
        layer4=self.MaxPoll1(layer3)
        layer5=self.down_conv3(layer4)
        layer6=self.MaxPoll1(layer5)
        layer7=self.down_conv4(layer6)
        layer8=self.MaxPoll1(layer7)
        layer9=self.down_conv5(layer8)
        layer10=self.ConvTraspose1(layer9)
        layer=torch.cat([layer7,layer10],1)
        layer11=self.up_sample1(layer)
        layer12=self.ConvTraspose2(layer11)
        layer13=self.up_sample2(torch.cat([layer12,layer5],1))
        layer14=self.ConvTraspose3(layer13)
        layer15=self.up_sample3(torch.cat([layer14,layer3],1))
        layer16=self.ConvTraspose4(layer15)
        layer17=self.up_sample4(torch.cat([layer16,layer1],1))
        outputs=torch.sigmoid(self.out_con2d(layer17))
        return outputs

class PreproseConcreteData(Dataset):
    def __init__(self,train_data,transforms=None,IsTest=False):
        self.train=train_data
        self.transforms=transforms
        self.IsTest=IsTest
    def __len__(self):
        return self.train.shape[0]  
    def __getitem__(self,idx):
        Image=cv2.imread(self.train.iloc[idx]['Images'])
        Image=cv2.resize(Image,(256,256))
        if not self.IsTest:
            Mask=cv2.imread(self.train.iloc[idx]['Masks'],0)
            Mask=cv2.resize(Mask,(256,256))
            ret,Mask = cv2.threshold(Mask,127,1,cv2.THRESH_BINARY)
            Mask=np.expand_dims(Mask, axis=0)
        if self.transforms:
            Image = self.transforms(**{"image": np.array(Image)})["image"]
        if self.IsTest:
            return Image
        Mask=torch.tensor(Mask,dtype=torch.float32)
        return Image,Mask

## Implementing BFS to bound a crack and pothole in a rectangle box, and to get all the co-ordinates
def doBFS(image, x, y):
  n = len(image)
  m = len(image[0])
  xFactor = [0, 0, 1, -1]
  yFactor = [1, -1, 0, 0]
  
  leftNode  = [255, 255]
  rightNode = [0, 0]
  topNode   = [255, 255]
  downNode  = [0, 0]
  pixels = 0
  q = []
  image[x][y] = 0
  q.append([x, y])

  while q :
    pixels += 1
    s = q.pop(0)

    if s[0] <= leftNode[0] :
      leftNode = s
    if s[0] >= rightNode[0] :
      rightNode = s
    if s[1] >= downNode[1] :
      downNode = s
    if s[1] <= topNode[1] :
      topNode = s

    for i in range(4):
      newX = s[0] + xFactor[i]
      newY = s[1] + yFactor[i]
      if newX>=0 and newX<n and newY>=0 and newY<m and image[newX][newY]>=1 :
        image[newX][newY] = 0
        q.append([newX, newY])
  return [pixels, [leftNode[0], topNode[1]], [rightNode[0], downNode[1]], [[leftNode[1], leftNode[0]], [downNode[1], downNode[0]], [rightNode[1], rightNode[0]], [topNode[1], topNode[0]]]]

## Function to find the area of patch relative to its bounding box
def findAreaRatio(mask, points):
    n = len(mask)
    m = len(mask[0])

    topNode, rightNode, downNode, leftNode = points
    patchArea       = 0
    boundingBoxArea = 0
    for j in range(m):
        for i in range(n):
            if(i>=leftNode[0] and i<=rightNode[0] and j>=topNode[1] and j<=downNode[1]):
                boundingBoxArea += 1
                if(mask[j][i]>=1):
                    patchArea += 1
    return round(patchArea/boundingBoxArea, 3)

## Function to find area of bounding box, given its co-ordinates
def findArea(x1, y1, x2, y2, x3, y3, x4, y4):
  return abs( ((x1*y2)-(x2*y1))+((x2*y3)-(x3*y2))+((x3*y4)-(x4*y3))+((x4*y1)-(x1*y4)) )/2

## Implementing Helper functions to find actual area and length given the output from UNet
def helper(img, lane, outputImg, krenelSize, color):
    print(">>> ", laneLength)
    widthPixelDensity  = (int)(laneWidth)/255
    heightPixelDensity = (int)(laneLength)/255
    ksize = krenelSize #(7, 7)
    image = cv2.blur(img, ksize) 
    n = len(image)
    m = len(image[0])

    data = []
    for i in range(255):
        for j in range(255):
            if image[i][j] >= 1 :
                pixels = doBFS(image, i, j)
                data.append(pixels)
    
    # Inverse Perspective Mapping
    print(">>>> ", points)
    # impTL = [100, 50]
    # impBL = [0, 250]
    # impTR = [175, 50]
    # impBR = [250, 250]
    impBL, impBR, impTL, impTR = points
    pts1  = np.float32([impTL, impBL, impTR, impBR])
    pts2  = np.float32([[0, 0], [0, 255], [255, 0], [255, 255]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    tranformes_frame = cv2.warpPerspective(outputImg, matrix, (255, 255))
    outputImg = cv2.circle(outputImg, impTL, 5, (255, 0, 0), -1)
    outputImg = cv2.circle(outputImg, impBL, 5, (255, 0, 0), -1)
    outputImg = cv2.circle(outputImg, impTR, 5, (255, 0, 0), -1)
    outputImg = cv2.circle(outputImg, impBR, 5, (255, 0, 0), -1)

    for i in data:
        if i[0] >= 100:
            topNode, rightNode, downNode, leftNode = i[3]
            bbTL = [leftNode[0], topNode[1]]
            bbBL = [leftNode[0], downNode[1]]
            bbBR = [rightNode[0], downNode[1]]
            bbTR = [rightNode[0], topNode[1]]
            transformed_pts  = cv2.perspectiveTransform(np.float32([[bbTL, bbBL, bbBR, bbTR]]), matrix)[0].astype(int)
            tl, bl, br, tr = transformed_pts
            if color == (0, 0, 255):
                # cv2.circle(tranformes_frame, transformed_pts[0], 2, (0, 0, 0), -1)
                # cv2.circle(tranformes_frame, transformed_pts[1], 2, (0, 0, 0), -1)
                # cv2.circle(tranformes_frame, transformed_pts[2], 2, (0, 0, 0), -1)
                # cv2.circle(tranformes_frame, transformed_pts[3], 2, (0, 0, 0), -1)
                # st.image(tranformes_frame)
                area = findArea(tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1])*widthPixelDensity*heightPixelDensity
                areaRatio   = findAreaRatio(img, [topNode, rightNode, downNode, leftNode])

                outputImg = cv2.rectangle(outputImg, [leftNode[0], topNode[1]], [rightNode[0], downNode[1]], color, 1)
                outputImg = cv2.putText(outputImg, str(round(area*areaRatio, 2)), [rightNode[0], topNode[1]], cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)
            else: 
                lengthTLBR = sqrt(((br[0]-tl[0])*widthPixelDensity)**2 + ((br[1]-tl[1])*heightPixelDensity)**2)
                lengthTRBL = sqrt(((bl[0]-tr[0])*widthPixelDensity)**2 + ((bl[1]-tr[1])*heightPixelDensity)**2)
                outputImg = cv2.rectangle(outputImg, [leftNode[0], topNode[1]], [rightNode[0], downNode[1]], color, 1)
                outputImg = cv2.putText(outputImg, str(round(max(lengthTLBR, lengthTRBL), 2)), [rightNode[0], topNode[1]], cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)
    return outputImg

def helper1(img, unetImg, color):
    for i in range(255):
        for j in range(255):
            if img[i][j] >= 1 :
                unetImg[i][j] = color
    return unetImg

## Implementing UNet given an image, giving 3outputs (cracks, potholes, lane)
def unetOnImage(image_file):
    # #checking fot 1038.png
    imgPath = "uplodedData/" + image_file
    print(imgPath)
    te_data=[(imgPath,imgPath)]
    te_df=pd.DataFrame(te_data, columns=["Images","Masks"])

    imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    train_tfms = A.Compose([ ToTensor(normalize=imagenet_stats) ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    te_ds=PreproseConcreteData(te_df, transforms=train_tfms,IsTest=True)
    te_dl = DataLoader(te_ds, batch_size=16, shuffle=True)
    # print(te_dl)

    for inputs in te_dl:
        inputs = inputs.to(device)
        outputs = tryModel(inputs)
        output2 = unetForPotholesModel(inputs)
        output3 = unetForLanesModel(inputs)
    outputs=outputs.squeeze()
    output2=output2.squeeze()
    output3=output3.squeeze()
    pred=np.array(outputs.to('cpu').detach()*255.0,dtype=np.uint8)
    pred2=np.array(output2.to('cpu').detach()*255.0,dtype=np.uint8)
    pred3=np.array(output3.to('cpu').detach()*255.0,dtype=np.uint8)
    org = cv2.imread(imgPath)
    org = cv2.resize(org,(256,256))
    outputImg = np.copy(org)

    outputImg = helper1(pred3, outputImg, (0, 255, 0))
    outputImg = helper1(pred, outputImg, (255, 0, 0))
    outputImg = helper1(pred2, outputImg, (0, 0, 255))

    outputImg = helper(pred, pred3, outputImg, (7, 7), (255, 0, 0))
    outputImg = helper(pred2, pred3, outputImg, (5, 5), (0, 0, 255))
    return [org, outputImg]

## Importing saved UNet models for crack, potholes, lane
tryModel = Unet()
with open("unetForCrackModelCPU.pickle", "rb") as fp:
    tryModel.load_state_dict(pickle.load(fp))

unetForPotholesModel = Unet()
with open("unetForPotholesModelCPU.pickle", "rb") as fp:
    unetForPotholesModel.load_state_dict(pickle.load(fp))

unetForLanesModel = Unet()
with open("unetForLaneModelCPU.pickle", "rb") as fp:
    unetForLanesModel.load_state_dict(pickle.load(fp))

def save_uploaded_file(uploadedFile):
    with open(os.path.join("uplodedData", uploadedFile.name), "wb") as f:
        f.write(uploadedFile.getbuffer())

## APP
if inputImage is not None:
    if filetype.is_image(inputImage):
        if inputImage is not None:
            image = Image.open(inputImage)
            image = image.resize((256, 256))
            col1, col2 = st.columns(2)
            with col1:
                st.header("Original Image")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  
                    stroke_width=3,
                    stroke_color="#000000",
                    background_color="#000000",
                    background_image=image if image else None,
                    update_streamlit=True,
                    height=256,
                    width=256,
                    drawing_mode='point',
                    point_display_radius=3,
                    key="canvas",
                    display_toolbar=False,
                )
                
                if canvas_result.json_data is not None:
                    data = canvas_result.json_data["objects"]
                    for i in range(len(data)):
                        points.append([(int)(data[i]["left"]), (int)(data[i]["top"])])
            with col2:
                if len(points)==4:
                    st.header("Prediction Image")
                    output = unetOnImage(inputImage.name)
                    st.image(output[1])
    elif filetype.is_video(inputImage):
        vidObj = cv2.VideoCapture("uplodedData/" + inputImage.name)
        count = 0
        while 1:
            success, image = vidObj.read()
            if success :
                cv2.imwrite("uplodedData/videoOutput/%d.jpg" % count, image)
                count += 1
            else: 
                break
        
        image_folder = 'uplodedData/videoOutput'
        video_name = 'mygeneratedvideo.mp4'
        
        images = [img for img in os.listdir(image_folder)
                if img.endswith(".jpg") or
                    img.endswith(".jpeg") or
                    img.endswith("png")]
        images = [ str(i)+".jpg" for i in sorted([ int(num.split('.')[0]) for num in images])]
        # frame = cv2.imread(os.path.join(image_folder, images[0]))
        frame = Image.open(os.path.join(image_folder, images[0]))
        height, width = (256, 256)
        video = cv2.VideoWriter(video_name, 0, 60, (width, height)) 

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  
            stroke_width=3,
            stroke_color="#000000",
            background_color="#000000",
            background_image=frame if frame else None,
            update_streamlit=True,
            height=256,
            width=256,
            drawing_mode='point',
            point_display_radius=3,
            key="canvas",
            display_toolbar=False,
        )
        if canvas_result.json_data is not None:
            data = canvas_result.json_data["objects"]
            for i in range(len(data)):
                points.append([(int)(data[i]["left"]), (int)(data[i]["top"])])

        if len(points)==4:
            for image in images: 
                output = unetOnImage("videoOutput/"+image)
                video.write(output[1]) 
            cv2.destroyAllWindows() 
            video.release()
            