import tornado.ioloop
import tornado.web
import cv2
from PIL import Image
from io import BytesIO
import base64

from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import classify

# Path to edgetpu compatible model
modelPath = './model_edgetpu.tflite'

# The path to labels.txt that was downloaded with your model
labelPath = './labels.txt'

print('Initializing opencv Video Stream')
cap = cv2.VideoCapture(0)

data = {'image': None, 'label': None}
# Load your model onto the TF Lite Interpreter
interpreter = make_interpreter(modelPath)
interpreter.allocate_tensors()
labels = read_label_file(labelPath)

# This function takes in a TFLite Interptere and Image, and returns classifications
def classifyImage(interpreter, image):
    size = common.input_size(interpreter)
    size = (int(size[0]), int(size[1]))
    common.set_input(interpreter, cv2.resize(image, size,
                                             interpolation=cv2.INTER_LINEAR))
    interpreter.invoke()
    return classify.get_classes(interpreter)


def update_data():
    if cap.isOpened():
        ret,frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            heigth, width, _ = img.shape
            margin = round((width-heigth)/2)
            img = img[0:heigth, margin:margin+heigth]
            img_inference = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)
            img = Image.fromarray(img_inference)
            output = BytesIO()
            img.save(output, format="png")
            data['image'] = base64.b64encode(output.getvalue()).decode()
            classification_result = classifyImage(interpreter, img_inference)
            data['label'] = f'{labels[classification_result[0].id]}, Score: {classification_result[0].score}'
            print(f'Label: {labels[classification_result[0].id]}, Score: {classification_result[0].score}')
            print(classification_result)
            #if classification_result [0][0] == 0 and  classification_result[0][1] > 0.95:
            #    pass
        else:
            data['image'] = None
            data['label'] = None
    else:
        data['image'] = None
        data['label'] = None

class ImgHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'GET')
        
    def initialize(self, data):
        self.data = data

    def get(self):
        self.write(data['image'])
        self.set_header("Server","Vision-Sorting/alpha/0.1")
        self.set_header("Content-type",  "image/png")

class LabelHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'GET')
        
    def initialize(self, data):
        self.data = data

    def get(self):
        response = {'label': None}
        response['label'] = data['label']
        self.write(response)
        self.set_header('Server','Vision-Sorting/alpha/0.1')
        self.set_header('Content-Type', 'application/json')

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [("/data=img", ImgHandler, {'data' : data}),
                    ("/data=label", LabelHandler, {'data' : data})]
        tornado.web.Application.__init__(self, handlers)

def main():
    interpreter = make_interpreter(modelPath)
    interpreter.allocate_tensors()
    labels = read_label_file(labelPath)
    application = Application()
    print('Starting Server on Port 8888')
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8888)

    # Callback function to update configs
    tornado.ioloop.PeriodicCallback(update_data, 50).start()
    tornado.ioloop.IOLoop.instance().start()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
