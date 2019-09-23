import numpy as np


def parse_model_cfg(path):
    """
    输入: 配置文件路径
    返回值: 列表对象,其中每一个元素为一个字典类型对应于一个要建立的神经网络模块（层）
    返回值示例: 
        [{"type":"net", "batch":"16", ...},
         {"type":"convolutional", "batch_normalize":"1", ...},
         {"type":"shortcut", "from":"-3", ...},
         {"type":"yolo", "mask":"6,7,8", ...},
         {"type":"route", "layers":"-4", ...},
         {"type":"upsample", "stride":"2", ...},
         ...]
    注: 配置文件定义了6种不同type：{'net', 'convolutional', 'route', 'shortcut', 'upsample', 'yolo'}
        其中，'net'相当于超参数,网络全局配置的相关参数
    """
    # Parses the yolo-v3 layer configuration file and returns module definitions
    file = open(path, 'r')
    lines = file.read().split('\n')                           # store the lines in a list等价于readlines
    lines = [x for x in lines if x and not x.startswith('#')] # 去掉以#开头的注释行
    lines = [x.rstrip().lstrip() for x in lines]              # 去掉左右两边的空格(rstricp是去掉右边的空格，lstrip是去掉左边的空格)
    mdefs = []  # module definitions
    for line in lines:
        if line.startswith('['):                              # 这是cfg文件中一个层(块)的开始  
            mdefs.append({})                                  # 新建一个空白字典寸储描述下一个块的信息
            mdefs[-1]['type'] = line[1:-1].rstrip()           # 把cfg的[]中的块名作为键type的值
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0              # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=")                        # 按等号分割
            key = key.rstrip()

            if 'anchors' in key:
                # 如果key值为"anchors"，则将value值转换为numpy数组的形式进行存储
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors
            else:
                mdefs[-1][key] = val.strip()                  # 左边是key(去掉右空格)，右边是value(去掉左空格)，形成一个block字典的键值对

    return mdefs


def parse_data_cfg(path):
    # Parses the data configuration file
    options = dict()
    with open(path, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options
