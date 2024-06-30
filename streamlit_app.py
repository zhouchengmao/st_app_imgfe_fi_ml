# coding: utf-8


import streamlit as st
from PIL import Image
from torchvision import transforms
# from torchvision.models import resnet34, ResNet34_Weights
# from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg16, VGG16_Weights
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.io import imread
from skimage import data, color, feature, transform
from skimage import filters, measure

from LegacyMLUtils import *

# 机器学习
##from sklearn.linear_model import LogisticRegression
##from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
##from sklearn.ensemble import GradientBoostingClassifier
# 机器学习 - lgbm, xgb, catboost
from lightgbm import LGBMClassifier

##from xgboost.sklearn import XGBClassifier
##from catboost import CatBoostClassifier


# pytorch配置设备
DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'  # * nvidia cuda or mac m1
DEVICE = torch.device(DEVICE_NAME)
# DEVICE_NAME = 'cpu'  # * cpu only
# DEVICE = torch.device("cpu")  # * cpu only
print(f'torch {torch.__version__}', f'; device: {DEVICE}')

# 初始化机器学习模型
col_y = 'target'
vgg16_forest = RandomForestClassifier(n_estimators=10, min_samples_leaf=15, criterion='entropy', n_jobs=-1,
                                      random_state=1)  # 随机森林 forest.fit(X_train,y_train)
vgg16_gbm = LGBMClassifier(learning_rate=0.11, n_estimators=10, lambda_l1=0.01, lambda_l2=10, max_depth=2,
                           bagging_fraction=0.8, feature_fraction=0.5)  # lgb
vgg16_forest_result = None
vgg16_gbm_result = None
vgg16_new_cols = None
vgg16_imp = None
vgg16_ss = None
skimage_hog_forest = RandomForestClassifier(n_estimators=10, min_samples_leaf=15, criterion='entropy', n_jobs=-1,
                                            random_state=1)  # 随机森林 forest.fit(X_train,y_train)
skimage_hog_gbm = LGBMClassifier(learning_rate=0.11, n_estimators=10, lambda_l1=0.01, lambda_l2=10, max_depth=2,
                                 bagging_fraction=0.8, feature_fraction=0.5)  # lgb
skimage_hog_forest_result = None
skimage_hog_gbm_result = None
skimage_hog_new_cols = None
skimage_hog_imp = None
skimage_hog_ss = None


# 配置机器学习模型（目前有：随机森林，LightGBM）
def setup_ml_models():
    global vgg16_forest, vgg16_gbm, vgg16_new_cols, skimage_hog_forest, skimage_hog_gbm, skimage_hog_new_cols, vgg16_imp, vgg16_ss, skimage_hog_imp, skimage_hog_ss, vgg16_forest_result, vgg16_gbm_result, skimage_hog_forest_result, skimage_hog_gbm_result

    with st.spinner("Training for `vgg16`, please wait..."):
        df_train_vgg16 = read_csv('6.2-tiff_vgg16_train.csv')
        df_train_vgg16_lgbm, vgg16_new_cols = to_lgbm_fi(df_train_vgg16)
        vgg16_new_cols.remove(col_y)

        df_train_vgg16_lgbm, vgg16_imp, vgg16_ss = preprocessing_data_train(df_train_vgg16_lgbm)
        vgg16_X, vgg16_y = split_x_y(df_train_vgg16_lgbm)
        vgg16_y = vgg16_y.astype(int)
        model_fit(vgg16_forest, vgg16_X, vgg16_y)
        vgg16_forest_result = model_score(vgg16_forest, vgg16_X, vgg16_y)
        model_fit(vgg16_gbm, vgg16_X, vgg16_y)
        vgg16_gbm_result = model_score(vgg16_gbm, vgg16_X, vgg16_y)

    with st.spinner("Training for `skimage-hog`, please wait..."):
        df_train_skimage_hog = read_csv('6.2-tiff_ski-hog_train.csv')
        df_train_skimage_hog_lgbm, skimage_hog_new_cols = to_lgbm_fi(df_train_skimage_hog)
        skimage_hog_new_cols.remove(col_y)

        df_train_skimage_hog_lgbm, skimage_hog_imp, skimage_hog_ss = preprocessing_data_train(df_train_skimage_hog_lgbm)
        skimage_hog_X, skimage_hog_y = split_x_y(df_train_skimage_hog_lgbm)
        skimage_hog_y = skimage_hog_y.astype(int)
        model_fit(skimage_hog_forest, skimage_hog_X, skimage_hog_y)
        skimage_hog_forest_result = model_score(skimage_hog_forest, skimage_hog_X, skimage_hog_y)
        model_fit(skimage_hog_gbm, skimage_hog_X, skimage_hog_y)
        skimage_hog_gbm_result = model_score(skimage_hog_gbm, skimage_hog_X, skimage_hog_y)


# 特征提取处理函数保存引用
single_vgg16_feature_extractor = None
single_hog_feature_extractor = None


# 配置VGG16模型
# torch-VGG16
def setup_vgg16_model():
    # 定义图片变换器
    trf1_vgg16 = transforms.Compose([
        transforms.Resize(256),
        ##transforms.Scale(256),  # Deprecated: Scale -> Change to: Resize
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    # 初始化模型
    model = m_vgg16 = vgg16(weights=VGG16_Weights.DEFAULT)  # Deprecated: pretrained=True
    ###model.fc = nn.Linear(2048, 2048)  # 重新定义最后一层输出
    ###nn.init.eye_(model.fc.weight)
    ##nn.init.eye(model.fc.weight)  # Deprecated: eye -> Change to: eye_
    model = m_vgg16 = model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    # 封装处理过程：单独一次处理
    def single_vgg16_feature_extractor(ffn):
        trf1 = trf1_vgg16
        model = m_vgg16

        # 打开图片
        img = Image.open(ffn)

        # 图片变换
        img1 = trf1(img)

        # 转换输入数据，进行预测操作（提取出特征数据）
        X = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False).to(DEVICE)
        y = model(X)

        ny = y.data.to('cpu').numpy()[0]
        return ny

    return single_vgg16_feature_extractor


# 配置skimage-HOG特征提取器
# skimage-方向梯度直方圖（Histogram of Oriented Gradient, HOG）
def setup_skimage_hog():
    def single_hog_feature_extractor(ffn, visualize=False):
        # 打开图片
        image = imread(ffn)

        # 图片变换
        image = transform.resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True)
        image = color.rgb2gray(image)

        # 提取特征
        return feature.hog(image, visualize=visualize)

    return single_hog_feature_extractor


# 通用的处理单个图片的函数
def fe(fe_func, ffn):
    with st.spinner("Feature extracting, please wait..."):
        # 提取特征
        vec = fe_func(ffn)
        # 转换为DataFrame
        dlist = [vec]
        dlist = np.array(dlist)
        df = pd.DataFrame(dlist, columns=[f'ft{i + 1}' for i in range(len(dlist[0]))])
        return df


# 渲染UI界面
def render_ui():
    global vgg16_forest, vgg16_gbm, vgg16_new_cols, skimage_hog_forest, skimage_hog_gbm, skimage_hog_new_cols, vgg16_imp, vgg16_ss, skimage_hog_imp, skimage_hog_ss, skimage_hog_ss, vgg16_forest_result, vgg16_gbm_result, skimage_hog_forest_result, skimage_hog_gbm_result

    st.title("Dr. Z.C.M.")
    st.header("Online Pathological Diagnosis", divider='rainbow')
    st.subheader("""Select a file to upload, and get predicted result.""")

    uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", ],
                                     accept_multiple_files=False)
    if uploaded_file is not None:
        # 显示上传的图像
        st.text("Your uploaded image file:")
        # st.write(uploaded_file)
        st.image(Image.open(uploaded_file))

        # 初始化特征提取函数（可能会走模型下载、缓存的过程）
        single_vgg16_feature_extractor = setup_vgg16_model()
        single_hog_feature_extractor = setup_skimage_hog()

        # 通过预训练模型或skimage方法进行特征提取，得到二维表数据后用于预测
        # 目前暂时实现了：VGG16模型提取方法，skimage-HOG处理方法
        with st.spinner("Processing, please wait..."):
            pocd_vgg16 = fe(single_vgg16_feature_extractor, uploaded_file)
            pocd_skimage_hog = fe(single_hog_feature_extractor, uploaded_file)
        st.text("File proceeded successfully.")

        # 初始化机器学习模型，执行训练
        setup_ml_models()

        # 将上面得到的数据分别送入不同模型，执行预测
        # VGG16
        st.subheader("""Feature extraction with `vgg16`, predict with model `RandomForestClassifier` and `LGBMClassifier`""")
        with st.spinner("Predicting with `vgg16`, please wait..."):
            pocd_vgg16_lgbm = pocd_vgg16[vgg16_new_cols]
            pocd_vgg16_lgbm = preprocessing_data_predict(pocd_vgg16_lgbm, vgg16_imp, vgg16_ss)

            st.text("Preview predict data of `vgg16`:")
            st.write(pocd_vgg16_lgbm)

            pocd_vgg16_forest_pd = model_predict(vgg16_forest, pocd_vgg16_lgbm)
            pocd_vgg16_gbm_pd = model_predict(vgg16_gbm, pocd_vgg16_lgbm)

            st.text(f"Model `RandomForestClassifier` predict result is: {pocd_vgg16_forest_pd[0]}")
            st.text(f"(Model Score: accuracy {'%.3f' %vgg16_forest_result['accuracy_score']}, precision {'%.3f' %vgg16_forest_result['preci_score']}, recall {'%.3f' %vgg16_forest_result['recall_score']}, f1 {'%.3f' %vgg16_forest_result['f1_score']})")

            st.text(f"Model `LGBMClassifier` predict result is: {pocd_vgg16_gbm_pd[0]}")
            st.text(f"(Model Score: accuracy {'%.3f' %vgg16_gbm_result['accuracy_score']}, precision {'%.3f' %vgg16_gbm_result['preci_score']}, recall {'%.3f' %vgg16_gbm_result['recall_score']}, f1 {'%.3f' %vgg16_gbm_result['f1_score']})")
        # skimage-hog
        st.subheader("""Feature extraction with `skimage-hog`, predict with model `RandomForestClassifier` and `LGBMClassifier`""")
        with st.spinner("Predicting with `skimage-hog`, please wait..."):
            pocd_skimage_hog_lgbm = pocd_skimage_hog[skimage_hog_new_cols]
            pocd_skimage_hog_lgbm = preprocessing_data_predict(pocd_skimage_hog_lgbm, skimage_hog_imp, skimage_hog_ss)

            st.text("Preview predict data of `skimage-hog`:")
            st.write(pocd_skimage_hog_lgbm)

            pocd_skimage_hog_forest_pd = model_predict(skimage_hog_forest, pocd_skimage_hog_lgbm)
            pocd_skimage_hog_gbm_pd = model_predict(skimage_hog_gbm, pocd_skimage_hog_lgbm)

            st.text(f"    Model `RandomForestClassifier` predict result is: {pocd_skimage_hog_forest_pd[0]}")
            st.text(f"(Model Score: accuracy {'%.3f' %skimage_hog_forest_result['accuracy_score']}, precision {'%.3f' %skimage_hog_forest_result['preci_score']}, recall {'%.3f' %skimage_hog_forest_result['recall_score']}, f1 {'%.3f' %skimage_hog_forest_result['f1_score']})")

            st.text(f"    Model `LGBMClassifier` predict result is: {pocd_skimage_hog_gbm_pd[0]}")
            st.text(f"(Model Score: accuracy {'%.3f' %skimage_hog_gbm_result['accuracy_score']}, precision {'%.3f' %skimage_hog_gbm_result['preci_score']}, recall {'%.3f' %skimage_hog_gbm_result['recall_score']}, f1 {'%.3f' %skimage_hog_gbm_result['f1_score']})")


# 主程序入口
if __name__ == "__main__":
    render_ui()
