from sklearn.feature_extraction import DictVectorizer#字典特征
from sklearn.feature_extraction.text import CountVectorizer   # 文字特征提取(主要是频率统计
from  sklearn.feature_extraction.text import TfidfVectorizer#tfidf
import jieba

def dict_demo():
    """
    对字典类型的数据进行特征抽取
    :return: None
    """
    data = [{'city': '北京','temperature':100}, {'city': '上海','temperature':60}, {'city': '深圳','temperature':30}]
    #数据，用字典提取相应结果，用列表装载多个对象
    # 1、实例化转换器类
    transfer = DictVectorizer(sparse=False)
    """
     [[   0.    1.    0.  100.]
     [   1.    0.    0.   60.]
     [   0.    0.    1.   30.]]
    实际是一个one hot输出，前三列通过不同的1和0即可得到不同的数据
    
    """
    transfer = DictVectorizer()
    #稀疏矩阵，减少存储压力，但不太好观察性质
    #对比sparese值不同的效果
    # 2、调用fit_transform转换数据
    data = transfer.fit_transform(data)
    print("返回的结果:\n", data)

    # 打印特征名字
    print("特征名字：\n", transfer.get_feature_names_out())

def eng_word_demo():
    """
    对文本进行特征抽取，countvetorizer
    :return: None
    """
    data = ["life is short,i like like python", "life is too long,i dislike python"]
    # 1、实例化一个转换器类
    # transfer = CountVectorizer(sparse=False) # 注意,tests.CountVectorizer（）没有sparse这个参数

    transfer = CountVectorizer(stop_words='english')
    # transfer = CountVectorizer()
    """
    stop_words可以不统计某个单词
    单个字母和标点符号也不会被统计
    """
    # 2、调用fit_transform
    data = transfer.fit_transform(data)
    print("返回特征名字：\n", transfer.get_feature_names_out())
    print("文本特征抽取的结果：\n", data.toarray())#没有非离散化选项，用.toarray()展示
    #返回的结果为词语出现的频率,顺序为transfer.get_feature_names_out()的元素顺序
    #对应的特征名

def chinese_word_demo1():
    data = ["人生苦短，我喜欢Python","生活太长久，我不喜欢Python"]
    # 1、实例化一个转换器类
    transfer = CountVectorizer()
    """
    CountVectorizer不太能实现中文词语的分割
    """
    # 2、调用fit_transform
    data = transfer.fit_transform(data)
    print("返回特征名字：\n", transfer.get_feature_names_out())
    print("文本特征抽取的结果：\n", data.toarray())  # 没有非离散化选项，用.toarray()展示
    # 返回的结果为词语出现的频率,顺序为transfer.get_feature_names_out()的元素顺序
    # 对应的特征名

def cut_word(text): #拆词
    """
    对中文进行分词
    "我爱北京天安门"————>"我 爱 北京 天安门"
    :param text:
    :return: text
    """
    # 用结巴对中文字符串进行分词
    text = " ".join(list(jieba.cut(text)))
    return text##c
def chinese_word_demo2():
    data = [
    '今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。' ,
'    我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。',
'    如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。'
    ]

    test_list=[]
    for i in data:
        test_list.append((cut_word(i))) #对list的每一个句子进行拆词
    # print(test_list)

    transfer = CountVectorizer()
    """
    CountVectorizer不太能实现中文词语(没有空格 )的分割,同样中文也可以用stop_words,
    根据自己工作中的词汇来维护这个词汇表
    """
    # 2、调用fit_transform
    data = transfer.fit_transform(test_list)
    print("返回特征名字：\n", transfer.get_feature_names_out())
    print("文本特征抽取的结果：\n", data.toarray())  # 没有非离散化选项，用.toarray()展示
    # 返回的结果为词语出现的频率,顺序为transfer.get_feature_names_out()的元素顺序
    # 对应的特征名

def tfidf_demo():
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # 将原始数据转换成分好词的形式
    text_list = []
    for sent in data:
        text_list.append(cut_word(sent))
    print(text_list)

    # 1、换成tfidf的类
    # transfer = CountVectorizer(sparse=False)
    transfer = TfidfVectorizer(stop_words=['一种', '不会', '不要'])
    # 2、调用fit_transform
    data = transfer.fit_transform(text_list)
    print("文本特征抽取的结果：\n", data.toarray())
    print("返回特征名字：\n", transfer.get_feature_names_out())

if __name__ == '__main__':
    # dict_demo()
    # eng_word_demo()
    # chinese_word_demo1()#错误示范
    # print(cut_word(text= "人生苦短，我喜欢Python"))
    # chinese_word_demo2()
    tfidf_demo()