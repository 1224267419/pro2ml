import pandas as pd
from sklearn.decomposition import PCA

from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.cluster import KMeans
if __name__ == '__main__':

    # 数据读取
    order_product = pd.read_csv('./Instacart_dataset/order_products__prior.csv')
    products = pd.read_csv('./Instacart_dataset/products.csv')
    orders = pd.read_csv('./Instacart_dataset/orders.csv')
    aisles = pd.read_csv('./Instacart_dataset/aisles.csv')
    '''
    order._products._prior..csv:订单与商品信息
    字段：order_.id,product_id,add_to_cart_order,,reordered
    products.csv:商品信息
    字段：product_.id,product_name,aisle,_id,department_.id
    orders..csv:用户的订单信息
    字段：order_id,user_id,eval_set,order_.number,.
    aisles.csv:商品所属具体物品类别
    字段：aisle_id,aisle
    '''
    print("数据读取完成")
    # 2.1 合并表格
    table1 = pd.merge(order_product, products, on=["product_id", "product_id"])
    table2 = pd.merge(table1, orders, on=["order_id", "order_id"])
    table = pd.merge(table2, aisles, on=["aisle_id", "aisle_id"])
    # on=[]即以那一列为不变量进行表格合并
    # 2.2 交叉表合并
    print(table.shape)
    data = pd.crosstab(table["user_id"], table["aisle"])
    print(table.shape)

    print("表格合并完成")

    # data = data[:1000]  # 减少数据量,后面想尝试完整数据时可以注释掉

    print("pac前特征数为",data.shape)  # pac前特征数:134


    transfer = PCA(n_components=0.9)  #
    trans_data = transfer.fit_transform(data)

    print("pac保留90%信息后特征数为:",trans_data.shape)  # 仅使用22个特征即可保留90%的信息


    estimator = KMeans(n_clusters=5)
    y_pre = estimator.fit_predict(trans_data)
    silhouette_score(trans_data, y_pre)  #轮廓系数,取值范围[-1,1],值越大越好
    calinski_harabasz_score(trans_data, y_pre) #CH指标,值越大越好
