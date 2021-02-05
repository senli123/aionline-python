import os, sys
os.chdir(sys.path[0])
import wget
from detection.yolov3.yolov3  import Yolov3
from common import retry_call
def load_label_categories(label_file_path):
    categories = [line.rstrip('\n') for line in open(label_file_path)]
    return categories

LABEL_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'coco_labels.txt')
ALL_CATEGORIES = load_label_categories(LABEL_FILE_PATH)
def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color='blue'):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw
def generate_md5_checksum(local_path):
    """Returns the MD5 checksum of a local file.

    Keyword argument:
    local_path -- path of the file whose checksum shall be generated
    """
    with open(local_path, 'rb') as local_file:
        data = local_file.read()
        return hashlib.md5(data).hexdigest()
def download_file(local_path, link, checksum_reference=None):
    """Checks if a local file is present and downloads it from the specified path otherwise.
    If checksum_reference is specified, the file's md5 checksum is compared against the
    expected value.

    Keyword arguments:
    local_path -- path of the file whose checksum shall be generated
    link -- link where the file shall be downloaded from if it is not found locally
    checksum_reference -- expected MD5 checksum of the file
    """
    if not os.path.exists(local_path):
        print('Downloading from %s, this may take a while...' % link)
        ## Try 3 times
        retry_call(wget.download, args=[link, local_path], n_retries=3)
        print()
    if checksum_reference is not None:
        checksum = generate_md5_checksum(local_path)
        if checksum != checksum_reference:
            raise ValueError(
                'The MD5 checksum of local file %s differs from %s, please manually remove \
                 the file and try again.' %
                (local_path, checksum_reference))
    return local_path

def main():
    onnx_file_path = 'yolov3.onnx'
    engine_file_path = "yolov3.trt"
    # Download a dog image and save it to the following file path:
    input_image_path = download_file('dog.jpg',
        'https://github.com/pjreddie/darknet/raw/f86901f6177dfc6116360a13cc06ab680e0c86b0/data/dog.jpg', checksum_reference=None)

    yolov3_model = Yolov3()
    image_raw,boxes, classes, scores = yolov3_model.model_infer(input_image_path)
    obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
    output_image_path = 'dog_bboxes.png'
    obj_detected_img.save(output_image_path, 'PNG')
    print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))
if __name__ == '__main__':
    main()