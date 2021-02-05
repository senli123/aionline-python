from data_def import Service_data,TensorRT_data,Classification_data,Segmentation_data,Detection_data
from AI_data_def import *
class ParamsInit:
    def __init__(self):
        self.CLASS_MAP = {"qita": 0, "neibianxing": 1, "lianjiao": 2, "pifeng": 3,"shakong": 10,
                            "guoshi": 11, "guoshiduanjin": 12, "huashang": 20, "wuzi": 21, 
                            "yanghua": 22, "yiwu": 23, "heidian": 24, "tongheidian": 25, "feilinjiao": 26}
        self.CLASSID_MAP = {0: "qita", 1: "neibianxing", 2: "lianjiao", 3: "pifeng", 10: "shakong",
                            11: "guoshi", 12: "guoshiduanjin", 20: "huashang", 21: "wuzi",
                            22: "yanghua", 23: "yiwu", 24: "heidian", 25: "tongheidian", 26: "feilinjiao"}
        self.Serviceparams = Service_data(
                                        IOU_THRE,          #vi和ai融合时的阈值
										DETECTION_CLASS_LIST, #检测的类别list
										ClASSIFY_CLASS_LIST,
										NB_ClASSIFY_CLASS_LIST
        )
        #------------正面--------------
        #tenosrRT初始化参数
        #分类
        self.CTparams = TensorRT_data(
                            CLASSIFY_ONNXFILENAME,   #onnx模型名
							CLASSIFY_TRTFILENAME,    #bin模型名
							CLASSIFY_CUDAID,                #GPUid
							CLASSIFY_DATADIRS,         #模型保存位置
							CLASSIFY_SHAPE,                  #输入图片的维度
                            
        )
        #检测
        self.DTparams = TensorRT_data(
                            DETECTION_ONNXFILENAME,   #onnx模型名
                            DETECTION_TRTFILENAME,    #bin模型名
                            DETECTION_CUDAID,                #GPUid
                            DETECTION_DATADIRS,         #模型保存位置
                            DETECTION_SHAPE,             #输入图片的维度    
        )
        #分割
        self.STparams = TensorRT_data(
                            SEGMENTATION_ONNXFILENAME,   #onnx模型名
                            SEGMENTATION_TRTFILENAME,    #bin模型名
                            SEGMENTATION_CUDAID,                #GPUid
                            SEGMENTATION_DATADIRS,         #模型保存位置
                            SEGMENTATION_SHAPE,                  #输入图片的维度
                            
                            
        )
        #内变形
        self.NBCTparams = TensorRT_data(
                            NB_CLASSIFY_ONNXFILENAME,   #onnx模型名
							NB_CLASSIFY_TRTFILENAME,    #bin模型名
							NB_CLASSIFY_CUDAID,                #GPUid
							NB_CLASSIFY_DATADIRS,                 #模型保存位置
							NB_CLASSIFY_SHAPE,                   #输入图片的维度
        )
        #构造模型所需参数
	    #分类
        self.Cdata = Classification_data(
                                CLASSIFY_RESIZE,      #输入模型的大小
								CLASSIFY_BATCH_SIZE,   #bs
								CLASSIFT_CLASS_NUM,  #类别个数
								CLASSIFY_ENLAGE_SIZE,  #放大倍数
								CLASSIFY_MIN_LENGTH,    #放大后的最小size
								CLASSIFY_MEAN_R,
								CLASSIFY_MEAN_G,
								CLASSIFY_MEAN_B,
								CLASSIFY_STD_R,
								CLASSIFY_STD_G,
								CLASSIFY_STD_B,
        ) 
        #内变形
        self.NBCdata = Classification_data (
                                NB_CLASSIFY_RESIZE,      #输入模型的大小
								NB_CLASSIFY_BATCH_SIZE,   #bs
								NB_CLASSIFT_CLASS_NUM,  #类别个数
								NB_CLASSIFY_ENLAGE_SIZE,
								NB_CLASSIFY_MIN_LENGTH,
								NB_CLASSIFY_MEAN_R,
								NB_CLASSIFY_MEAN_G,
								NB_CLASSIFY_MEAN_B,
								NB_CLASSIFY_STD_R,
								NB_CLASSIFY_STD_G,
								NB_CLASSIFY_STD_B,
        ) 
        #检测
        self.Ddata = Detection_data(
                            DETECTION_RESIZE,      #输入模型的大小
							DETECTION_PADDING_GRAY,  #预处理要补充的灰度值
							DETECTION_STRIDE, #yolov5的stride
							DETECTION_CONFTHRE, #类别判断的阈值
							DETECTION_IOUTHRE, # iou阈值
							DETECTION_ANCHORNUM,   #每个点预测的anchor数目
							DETECTION_CLASSNUM,   #class数量
							DETECTION_SCOREBBOX,   #得分和框坐标的总数 5
							DETECTION_SCOREINDEX,  #得分的index
                            DETECTION_ANCHOR_WIDTH,  #anchor配置
                            DETECTION_ANCHOR_HEIGHT  #anchor配置
        )
        #分割
        self.Sdata = Segmentation_data(
                            SEGMENTATION_RESIZE,      #输入模型的大小
							SEGMENTATION_BATCH_SIZE,   #bs
							SEGMENTATION_CLASS_NUM,
							SEGMENTATION_SCORE_THRE,
							SEGMENTATION_ENLARGE_SIZE,  #放大倍数
							SEGMENTATION_MIN_LENGTH,     #放大后的最小size
							SEGMENTATION_PADDING_GRAY,
							SEGMENTATION_MEAN_R,
							SEGMENTATION_MEAN_G,
							SEGMENTATION_MEAN_B,
							SEGMENTATION_STD_R,
							SEGMENTATION_STD_G,
							SEGMENTATION_STD_B,
        )
        #----------------------------反面-----------------------------------------------
        #分类
        self.CTparams_FM = TensorRT_data(
                            CLASSIFY_ONNXFILENAME,   #onnx模型名
                            CLASSIFY_TRTFILENAME,    #bin模型名
                            CLASSIFY_CUDAID_FM,                #GPUid
							CLASSIFY_DATADIRS,                 #模型保存位置
							CLASSIFY_SHAPE,                   #输入图片的维度
        ) 
        #检测
        self.DTparams_FM = TensorRT_data(
                            DETECTION_ONNXFILENAME,   #onnx模型名
                            DETECTION_TRTFILENAME,    #bin模型名
                            DETECTION_CUDAID_FM,                #GPUid
                            DETECTION_DATADIRS,         #模型保存位置
                            DETECTION_SHAPE,                 #输入图片维度

        )
        #分割
        self.STparams_FM = TensorRT_data(
                            SEGMENTATION_ONNXFILENAME,   #onnx模型名
                            SEGMENTATION_TRTFILENAME,    #bin模型名
                            SEGMENTATION_CUDAID_FM,                #GPUid
                            SEGMENTATION_DATADIRS,         #模型保存位置
                            SEGMENTATION_SHAPE,                 #输入图片维度
                            
        )
        #构造模型所需参数
		#分类
        self.Cdata_FM = Classification_data(
                            CLASSIFY_RESIZE,      #输入模型的大小
                            CLASSIFY_BATCH_SIZE,   #bs
                            CLASSIFT_CLASS_NUM,  #类别个数
                            CLASSIFY_ENLAGE_SIZE,  #放大倍数
                            CLASSIFY_MIN_LENGTH,    #放大后的最小size
                            CLASSIFY_MEAN_R,
                            CLASSIFY_MEAN_G,
                            CLASSIFY_MEAN_B,
                            CLASSIFY_STD_R,
                            CLASSIFY_STD_G,
                            CLASSIFY_STD_B,
        )
        #检测
        self.Ddata_FM = Detection_data(
                            DETECTION_RESIZE,      #输入模型的大小
							DETECTION_PADDING_GRAY,  #预处理要补充的灰度值
							DETECTION_STRIDE, #yolov5的stride
							DETECTION_CONFTHRE, #类别判断的阈值
							DETECTION_IOUTHRE, # iou阈值
							DETECTION_ANCHORNUM,   #每个点预测的anchor数目
							DETECTION_CLASSNUM,   #class数量
							DETECTION_SCOREBBOX,   #得分和框坐标的总数 5
							DETECTION_SCOREINDEX,  #得分的index
                            DETECTION_ANCHOR_WIDTH,
                            DETECTION_ANCHOR_HEIGHT
        )
        #分割
        self.Sdata_FM = Segmentation_data(
                            SEGMENTATION_RESIZE,      #输入模型的大小
							SEGMENTATION_BATCH_SIZE,   #bs
							SEGMENTATION_CLASS_NUM,
							SEGMENTATION_SCORE_THRE,
							SEGMENTATION_ENLARGE_SIZE,  #放大倍数
							SEGMENTATION_MIN_LENGTH,     #放大后的最小size
							SEGMENTATION_PADDING_GRAY,
							SEGMENTATION_MEAN_R,
							SEGMENTATION_MEAN_G,
							SEGMENTATION_MEAN_B,
							SEGMENTATION_STD_R,
							SEGMENTATION_STD_G,
							SEGMENTATION_STD_B,
        )
							


