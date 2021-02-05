
#***********公用信息***********
QITA = "qita"
NEIBIANXING = "neibianxing"
LIANJIAO = "lianjiao"
PIFENG = "pifeng"
SHAKONG = "shakong"
GUOSHI = "guoshi"
GUOSHIDUANJIN ="guoshiduanjin"
HUASHANG ="huashang"
WUZI = "wuzi"
YANGHUA = "yanghua"
YIWU = "yiwu"
HEIDIAN = "heidian"
TONGHEIDIAN = "tongheidian"
FEILINJIAO = "feilinjiao"
CLASS_LIST = ["qita","neibianxing","lianjiao","pifeng","shakong","guoshi","guoshiduanjin","huashang","wuzi","yanghua","yiwu","heidian","tongheidian","feilinjiao"]
CLASS_ORDER_LIST = [0,1,2,3,10,11,12,20,21,22,23,24,25,26]
IOU_THRE = 0.001   #AI与ai框的iou阈值
DETECTION_CLASS_LIST = ["lianjiao", "shakong","guoshi", "yanghua", "wuzi", "huashang", "yiwu", "heidian", "tongheidian", "feilinjiao"]
ClASSIFY_CLASS_LIST = ["lianjiao","yiwu"]
NB_ClASSIFY_CLASS_LIST =["neibianxing","other"]
SEGMENTATION_CLASS_LIST = ["shakong","wuzi","yanghua","huashang","heidian","tongheidian","yiwu","feilinjiao"]
#***********检测相关****************************
#-----------模型相关------------------------
DETECTION_RESIZE = 640       #输入模型的大小
DETECTION_BATCH_SIZE = 1                #batch_size
DETECTION_PADDING_GRAY = 114  #预处理要补充的灰度值
DETECTION_STRIDE = [8,16,32] #yolov5的stride
DETECTION_CONFTHRE = 0.3#类别判断的阈值
DETECTION_IOUTHRE = 0.5# iou阈值
DETECTION_ANCHORNUM = 3#每个点预测的anchor数目
DETECTION_CLASSNUM = 10 #class数量
DETECTION_SCOREBBOX = 5 #得分和框坐标的总数 5
DETECTION_SCOREINDEX = 4  #得分的index
DETECTION_ANCHOR_WIDTH =  [[10, 16, 33], [30, 62, 59], [116, 156, 373]]
DETECTION_ANCHOR_HEIGHT = [[13, 30, 23], [61, 45, 119], [90, 198, 326]]
#-----------TensorRT相关-------------------------
DETECTION_ONNXFILENAME = "add_flj_no_pifeng_10cls_mix.onnx"   #onnx模型名
DETECTION_TRTFILENAME = "add_flj_no_pifeng_10cls_mix.bin"   #bin模型名
DETECTION_CUDAID = 0               #GPUid
DETECTION_CUDAID_FM = 0               #GPUid
DETECTION_DATADIRS = "E:/mvi_project/new_ai_debug/online_debug/model/"         #模型保存位置
DETECTION_SHAPE = [1,3,640,640]
#***********分类相关****************************
#-----------模型相关------------------------
CLASSIFY_RESIZE = 128                 #分类图像放resize大小
CLASSIFY_BATCH_SIZE = 4                #分类图像的batch_size
CLASSIFT_CLASS_NUM = 2
CLASSIFY_ENLAGE_SIZE = 10
CLASSIFY_MIN_LENGTH  = 16
CLASSIFY_MEAN_R  = 0.4914
CLASSIFY_MEAN_G  = 0.4822
CLASSIFY_MEAN_B  = 0.4465
CLASSIFY_STD_R   = 0.2023
CLASSIFY_STD_G   = 0.1994
CLASSIFY_STD_B   = 0.2010
#------------tensorRT相关---------------------
CLASSIFY_ONNXFILENAME = "lunkuo_lj_yw_4_batch_128.onnx"   #onnx模型名
CLASSIFY_TRTFILENAME = "lunkuo_lj_yw_4_batch_128.bin"    #bin模型名
CLASSIFY_CUDAID = 0                #GPUid
CLASSIFY_CUDAID_FM = 0                #GPUid
CLASSIFY_DATADIRS = "E:/mvi_project/new_ai_debug/online_debug/model/"         #模型保存位置
CLASSIFY_SHAPE = [4, 3, 128, 128]                #输入维度
#**********************************************
#***********内变形分类相关****************************
#-----------模型相关------------------------
NB_CLASSIFY_RESIZE = 512                 #分类图像放resize大小
NB_CLASSIFY_BATCH_SIZE = 1                #分类图像的batch_size
NB_CLASSIFT_CLASS_NUM = 2
NB_CLASSIFY_ENLAGE_SIZE = 10
NB_CLASSIFY_MIN_LENGTH  = 16
NB_CLASSIFY_MEAN_R  = 0.4914
NB_CLASSIFY_MEAN_G  = 0.4822
NB_CLASSIFY_MEAN_B  = 0.4465
NB_CLASSIFY_STD_R   = 0.2023
NB_CLASSIFY_STD_G   = 0.1994
NB_CLASSIFY_STD_B   = 0.2010
#------------tensorRT相关---------------------
NB_CLASSIFY_ONNXFILENAME = "nbx_other_1_batch_512.onnx"   #onnx模型名
NB_CLASSIFY_TRTFILENAME = "nbx_other_1_batch_512.bin"    #bin模型名
NB_CLASSIFY_CUDAID = 0                #GPUid
NB_CLASSIFY_CUDAID_FM = 0               #GPUid
NB_CLASSIFY_DATADIRS = "E:/mvi_project/new_ai_debug/online_debug/model/"         #模型保存位置
NB_CLASSIFY_SHAPE= [1, 3, 512, 512]               #输入维度

#**********************************************

#***********分割相关****************************
#-----------模型相关------------------------
SEGMENTATION_RESIZE = 512
SEGMENTATION_BATCH_SIZE = 4
SEGMENTATION_CLASS_NUM = 3
SEGMENTATION_SCORE_THRE = 0.3
SEGMENTATION_ENLARGE_SIZE = 1.5            #分割图像放大倍数
SEGMENTATION_MIN_LENGTH = 16            #分割图像最小size
SEGMENTATION_PADDING_GRAY = 0   #预处理要补充的灰度值
SEGMENTATION_MEAN_R  = 0.33148703
SEGMENTATION_MEAN_G  = 0.22545604
SEGMENTATION_MEAN_B  = 0.17101818
SEGMENTATION_STD_R   = 0.22985311
SEGMENTATION_STD_G   = 0.16206841
SEGMENTATION_STD_B   = 0.1269387
#------------tensorRT相关---------------------
SEGMENTATION_ONNXFILENAME = "unet_b4_512.onnx"   #onnx模型名
SEGMENTATION_TRTFILENAME = "unet_b4_512.bin"    #bin模型名
SEGMENTATION_CUDAID = 0                #GPUid
SEGMENTATION_CUDAID_FM = 0 
SEGMENTATION_DATADIRS = "E:/mvi_project/new_ai_debug/online_debug/model/"          #模型保存位置
SEGMENTATION_SHAPE = [4, 3, 512, 512]                #输入维度
#-----------service相关------------------------
#**********************************************