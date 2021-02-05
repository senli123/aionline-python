#Service所需数据类型定义
class Service_data:
    def __init__(self, thre, Detection_list, Classify_list, NB_Classify_list):  
            self.Iou_thre = thre # vi和ai融合所需的iou阈值
            self.Detection_class_list = Detection_list
            self.Classify_class_list = Classify_list
            self.NB_Classify_class_list = NB_Classify_list

#tensorRT所需数据类型定义
class TensorRT_data:
    def __init__(self, OnnxName, trtName, GPUID, Dirs, shape):
        self.OnnxFileName = OnnxName   #onnx模型名
        self.trtFileName = trtName    #bin模型名
        self.CudaID = GPUID                #GPUid
        self.dataDirs = Dirs         #模型保存位置
        self.shape = shape

#分类model所需数据类型定义
class Classification_data: 
    def __init__(self, shape, bs, classnum, enlarge, minsize, Rmean, Gmean, Bmean, Rstd, Gstd, Bstd):
        self.Resize = shape     #输入模型的大小
        self.BatchSize = bs   #bs
        self.ClassNum = classnum    #类别个数
        self.EnlageSize = enlarge  #放大倍数
        self.MinSize = minsize     #放大后的最小size    
        self.MEAN_R = Rmean 
        self.MEAN_G = Gmean   
        self.MEAN_B = Bmean 
        self.STD_R = Rstd 
        self.STD_G = Gstd 
        self.STD_B = Bstd 

#分割model所需数据类型定义
class Segmentation_data:   
    def __init__(self, shape, bs, classnum, thre, enlarge, minsize, Padding, Rmean, Gmean, Bmean, Rstd, Gstd, Bstd):
        self.Resize = shape       #输入模型的大小
        self.BatchSize = bs   #bs
        self.Class_num = classnum 
        self.Score_thre = thre  #阈值
        self.EnlageSize = enlarge  #放大倍数
        self.MinSize = minsize    #放大后的最小size
        self.PaddingGary = Padding  #预处理要补充的灰度值
        self.MEAN_R = Rmean 
        self.MEAN_G = Gmean 
        self.MEAN_B = Bmean 
        self.STD_R = Rstd 
        self.STD_G = Gstd 
        self.STD_B = Bstd 

#检测model所需数据类型定义
class Detection_data:  
    def __init__(self, shape, padding, stride, thre, iou, anchor_num, classnum, scorebbox, 
                    scoreindex, anchor_width_list, anchor_height_list):
        self.Resize = shape       #输入模型的大小
        self.PaddingGary = padding #预处理要补充的灰度值
        self.Stride = stride #yolov5的stride
        self.ConfThre = thre #类别判断的阈值
        self.IouThre = iou # iou阈值
        self.AnchorNum = anchor_num  #每个点预测的anchor数目
        self.ClassNum = classnum    #class数量
        self.ScoreBbox = scorebbox  #得分和框坐标的总数 5
        self.ScoreIndex = scoreindex  #得分的index
        self.base_anchor_width = anchor_width_list
        self.base_anchor_height = anchor_height_list

#保存检测结果的数据类型
class InstanceInfo:
    def __init__(self, xmin, ymin, xmax, ymax, score, classID):
        self.x1 = xmin           # 预测框的左上角坐标和宽高
        self.y1 = ymin 
        self.x2 = xmax 
        self.y2 = ymax 
        self.score = score             # 某一类别的概率（最大概率）
        self.class_id = classID   # 概率最大的类别t
