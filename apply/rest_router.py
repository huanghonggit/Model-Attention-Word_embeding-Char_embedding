import sys

sys.path.append(["../../", "../", "./"])
from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource, fields, marshal_with
import socket
from apply.predict import predict, load_vocab
import driver.HyperConfig as Config

config = Config.Config('config/hyper_param.cfg')
vocab = load_vocab(config.load_vocab_path)

# 数据格式化
resource_fields = {
    'code': fields.Integer(default=200),
    'msg': fields.String(default='OK'),
    'data_text': fields.String(default=''),
    'data_val': fields.String(default='')
    # 'corpus': fields.String(default=''),
    # 'data_text': fields.String(attribute='corpus'),  # 重命名属性，属性data以result名字返回
    # 'data_col': fields.String(default='')
}


# 响应实体
class RespEntity(object):
    def __init__(self, code=200, msg='OK', data_text=None, data_val=None): # , data_col=None
        self.code = code
        self.msg = msg
        self.data_text = data_text
        self.data_val = data_val
        # self.data_col = data_col


# Flask-RESTful提供了一个用于参数解析的RequestParser类，可以很方便的解析请求中的-d参数，并进行类型转换
class Toparse(Resource):
    def __init__(self):
        self._req_param = 'text'
        self._req_parse = reqparse.RequestParser()
        # 要求一个值传递的参数，只需要添加 required=True
        # 如果指定了help参数的值，在解析的时候当类型错误被触发的时候，它将会被作为错误信息给呈现出来
        # 接受一个键有多个值的话，你可以传入 action='append'
        self._req_parse.add_argument(self._req_param, type=str, required=True, help='invalid params!')

    @marshal_with(resource_fields)
    def get(self):
        print("visiting....")
        return RespEntity(code=200, msg='你访问成功了！')

    @marshal_with(resource_fields)
    def post(self):
        print(request.headers)
        print(request.method)
        print(request.url)
        if not request.json:
            abort(400, message='no json format!')
            return RespEntity(code=400, msg='no json format!')

        print(request.json)

        # print(request.json.get('name'))
        # return {'result': 'wlz'}  #Flask-RESTful会自动地处理转换成JSON数据格式，可以省去jsonify

        args = self._req_parse.parse_args()  # 返回Python字典

        req_data = args[self._req_param]
        if req_data is None:
            return RespEntity(code=400, msg='bad request!')

        req_lst = req_data.split('|||')
        req_lst = [s.strip() for s in req_lst if s.strip() != '']
        print('req_lst:',req_lst)

        res_lst, emo_values = predict(req_lst, vocab, config)
        values = []
        for each in emo_values:
            each = each.item()
            values.append(round(each, 4))  # 取前四位

        result = ''.join(res_lst)

        return RespEntity(data_text=result, data_val=values) #, data_tag=crow_max, data_col=col


# 获取内网IP
def get_ip():
    hostname = socket.gethostname()  # 主机名
    return socket.gethostbyname(hostname)  # 内网IP


app = Flask(__name__)
api = Api(app)
# 设置路由urls:
api.add_resource(Toparse, '/parse', '/parse/')  # 一个资源挂载在多个路由上


if __name__ == '__main__':
    ip = get_ip()
    print("ip"+ip)
    app.run(host=ip, port=9300)



# print("type:",type(emo_values))
        # pos = []
        # gen = []
        # neg = []
        # for line in emo_values:
        #     pos.append(line[0])
        #     gen.append(line[1])
        #     neg.append(line[2])
        # pos_total = sum(pos)
        # gen_total = sum(gen)
        # neg_total = sum(neg)
        # val_tol = []
        # val_tol.append(pos_total)
        # val_tol.append(gen_total)
        # val_tol.append(neg_total)