from TensorRT.Infer_Interface_TensorRT import Infer_TensorRT as Infer_Interface
from model.detection.yolov3.yolov3_preprocess import PreprocessYOLO as Preprocess
from model.detection.yolov3.yolov3_postprocess import PostprocessYOLO as Postprocess
class Yolov3:
    def __init__(self):
        input_resolution_yolov3_HW = (608, 608)
        self.output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]
        postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
                          "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  # A list of 9 two-dimensional tuples for the YOLO anchors
                                           (59, 119), (116, 90), (156, 198), (373, 326)],
                          "obj_threshold": 0.6,                                               # Threshold for object coverage, float value between 0 and 1
                          "nms_threshold": 0.5,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                          "yolo_input_resolution": input_resolution_yolov3_HW,
                          "CATEGORY_NUM":80}
        self.pre = Preprocess(input_resolution_yolov3_HW)
        self.Infer = Infer_Interface("yolov3.trt", 1)
        self.post = Postprocess(**postprocessor_args)
    def model_infer(self,img_path):
        image_raw,image = self.pre.process(img_path)
        shape_orig_WH = image_raw.size
        trt_outputs = self.Infer.infer(image)
        self.Infer.destory()
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.output_shapes)]
        boxes, classes, scores = self.post.process(trt_outputs, (shape_orig_WH))
        return image_raw, boxes, classes, scores