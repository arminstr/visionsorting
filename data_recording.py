import tornado.ioloop
import tornado.web
import tornado.websocket
import cv2
from PIL import Image
from io import BytesIO
import base64
import uuid


print('Initializing opencv Video Stream')
cap = cv2.VideoCapture(0)

data = {'image': None}

def update_data():
    if cap.isOpened():
        print("capturing image... ")
        ret,frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            heigth, width, _ = img.shape
            margin = round((width-heigth)/2)
            img = img[0:heigth, margin:margin+heigth]
            img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)
            img = Image.fromarray(img)
            output = BytesIO()
            img.save(output, format="png")
            data['image'] = base64.b64encode(output.getvalue()).decode()            
        else:
            data['image'] = None
    else:
        data['image'] = None

class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, data):
        self.data = data
    
    def open(self):
        self.write_message({"type":"init", "data":"vision__sorting/alpha/0.1"})

    def on_message(self, message):
        if (message == "req?frame"):
            print("req?frame")
            self.write_message({"type":"image", "uuid":str(uuid.uuid1()),"data":data["image"]})

    def on_close(self):
        pass

    def check_origin(self, origin):
        return True

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [(r"/websocket", WebSocketHandler, {'data' : data})]
        tornado.web.Application.__init__(self, handlers)

def main():
    application = Application()
    print('Starting Server on Port 8888')
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8888)

    # Callback function to update configs
    tornado.ioloop.PeriodicCallback(update_data, 100).start()
    tornado.ioloop.IOLoop.instance().start()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
