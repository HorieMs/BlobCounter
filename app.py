import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import datetime
import plotly.graph_objs as go

st.set_page_config(
     page_title="Blob counter",
     #page_icon="⛓️",
     page_icon="⛆",
     layout="wide",
     initial_sidebar_state="auto",
     menu_items={
         'Get Help': 'https://koumyou.org/',
         'Report a bug': "https://koumyou.org/",
         'About': "# This is a Blob counter app!"
     }
 )

# Blobパラメータの初期化
params = cv2.SimpleBlobDetector_Params() 

color_enhancement_menu=('None','Hue rotation','Hue extraction')
color_enhancement_mode=color_enhancement_menu[0]
h_offset=0     # 色相回転角[deg], 0のときは、色相抽出
hsv_gain=np.array([1.0,1.0,1.0])
h_ext_center=90     # 色相抽出角[deg],0-360
h_ext_hrange=90     # 色相抽出角[deg. ] h_ext_center-h_ext_hrange ...　h_ext_center+h_ext_hrange の範囲のみを抽出
input_type=('File','Camera')


@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

@st.cache
def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def simple_blob(image):

    # 検知器作成
    detector = cv2.SimpleBlobDetector_create(params) 

    # ブロブ検知 
    kp = detector.detect(image) 

    # ブロブを赤丸で囲む
    blank = np.zeros((1, 1))  
    blobs = cv2.drawKeypoints(image, kp, blank, (0, 0, 255), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS) 


    # ブロブの個数  
    count = len(kp) 
    #print(f'丸の個数: {count}')
    text=f"{count:d} Blob"
    org=(0,int(blobs.shape[0]*0.98))
    blobs=cv2.putText(blobs, text, org, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=4, color=(255,0,0))


    
    # 画像を表示
    st.image(blobs,caption='blobs',use_column_width=True,channels="BGR")

    st.write(text)

    df = pd.DataFrame(columns=['x', 'y', 'size', 'angle']) 
    df_size=np.empty(count,dtype=float) 
    for i in range(len(kp)):
        df_size[i]=kp[i].size
        df.loc[str(i+1).zfill(6)] = [kp[i].pt[0],kp[i].pt[1],kp[i].size,kp[i].angle]
    
    #df.to_csv("key point.csv")
    #print(df)
    st.write(df)

    csv = convert_df(df)

    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    # YYYYMMDDhhmmss形式に書式化
    d = now.strftime('%Y%m%d%H%M%S')
    fname='data_'+d+'.csv'
    st.subheader('Download blob data')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=fname,
        mime='text/csv',
    )

    # ヒストグラム
    st.subheader('Histogram of blob size')
    #data = go.Histogram(x=df_size,xbins=dict(start=0, end=101, size=10)) # 区間の指定。sizeが区間幅
    data = go.Histogram(x=df_size) 
    # レイアウトの指定
    layout = go.Layout( xaxis = dict(title="size"), yaxis = dict(title="count"), bargap = 0.1) # 棒の間隔
    fig = dict(data=[data], layout=layout)
    st.plotly_chart(fig, use_container_width=True)


def color_enhancement(img):
    hsv_img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL) # H最大値255で変換した結果

    enh_hsv_img=np.empty(hsv_img.shape,dtype='uint8')

    if color_enhancement_mode==color_enhancement_menu[1]:
        # 色相回転の場合
        enh_h=np.uint16((np.round((hsv_img[:,:,0]+h_offset)*hsv_gain[0])))%256
        enh_s=np.clip(np.round(hsv_gain[1]*hsv_img[:,:,1]),0,255)
        enh_v=np.clip(np.round(hsv_gain[2]*hsv_img[:,:,2]),0,255)
    else:
        # 色相抽出の場合
        h_start=np.int16(np.round(255*(h_ext_center-h_ext_hrange)/360))
        h_end=np.int16(np.round(255*(h_ext_center+h_ext_hrange)/360))
        enh_h=hsv_img[:,:,0]
        enh_s=hsv_img[:,:,1]
        enh_v=hsv_img[:,:,2]
        if (h_start>=0) and (h_end<=255):
            bin_img=(hsv_img[:,:,0]<h_start) | (hsv_img[:,:,0]>h_end)
            enh_v[bin_img]=0
        elif (h_start<0) and (h_end<=255):
            h_start=255+h_start
            bin_img=(hsv_img[:,:,0]>h_end) & (hsv_img[:,:,0]<h_start)
            enh_v[bin_img]=0
        elif (h_start>=0) and (h_end>255):
            h_end=(h_end % 256)
            bin_img=(hsv_img[:,:,0]>h_end) & (hsv_img[:,:,0]<h_start)
            enh_v[bin_img]=0
        else:
            pass

        enh_s=np.clip(np.round(hsv_gain[1]*enh_s),0,255)
        enh_v=np.clip(np.round(hsv_gain[2]*enh_v),0,255)

    enh_hsv_img[:,:,0]=np.uint8(enh_h)
    enh_hsv_img[:,:,1]=np.uint8(enh_s)
    enh_hsv_img[:,:,2]=np.uint8(enh_v)

    enh_bgr_img=cv2.cvtColor(enh_hsv_img, cv2.COLOR_HSV2BGR_FULL) # H最大値255で変換した結果
    return enh_bgr_img



if __name__ == "__main__":
    st.header("Blob counter")

    st.subheader('Demo software for OpenCV and streamlit')

    
    
    st.sidebar.header('Parameters')

    st.sidebar.subheader('Input')
    InputType = st.sidebar.radio( "Input type",input_type, key='InputType')

    st.sidebar.subheader('Color enhancement')

    if color_enhancement_mode==color_enhancement_menu[0]:
        index=0
    elif color_enhancement_mode==color_enhancement_menu[1]:
        index=1
    else:
        index=2
 
    color_enhancement_mode=st.sidebar.selectbox('Mode', index=index,options=color_enhancement_menu)
    if color_enhancement_mode==color_enhancement_menu[1]:
        h_offset=st.sidebar.slider('Hue rotation [deg]',min_value=-180,max_value=180,value=h_offset,step=5,key='h_offset')
    elif color_enhancement_mode==color_enhancement_menu[2]:
        # 色相抽出角[deg],0-360
        h_ext_center=st.sidebar.slider('Hue [deg]',min_value=0,max_value=360,value=h_ext_center,step=5,key='h_ext_center')
        # 色相抽出角[deg. ] h_ext_center-h_ext_hrange ...　h_ext_center+h_ext_hrange の範囲のみを抽出
        h_ext_hrange=st.sidebar.slider('Range of Hue [deg]',min_value=5,max_value=180,value=h_ext_hrange,step=5,key='h_ext_hrange')
    if color_enhancement_mode!=color_enhancement_menu[0]:
        hsv_gain[1]=st.sidebar.slider('Staturation(Chorama)',min_value=0.0,max_value=2.0,value=float(hsv_gain[1]),step=0.1,key='saturation')
        hsv_gain[2]=st.sidebar.slider('Value(Brightness)',min_value=0.1,max_value=2.0,value=float(hsv_gain[2]),step=0.1,key='value')
 


    st.sidebar.header('Blob')

    chk_fbc=st.sidebar.checkbox('Filter by Color',value=params.filterByColor,key='chk_fbc')
    if chk_fbc:
        params.filterByColor=True
        BlobType = st.sidebar.radio( "Blob type",     ('Dark', 'Light'),key='BlobType')
        if BlobType=='Dark':
            params.blobColor=0
        else:
            params.blobColor=255
    else:
        params.filterByColor=False
    
    # ブロブ領域（minArea <= blob < maxArea）
    chk_fba=st.sidebar.checkbox('Filter by Area',value=params.filterByArea,key='chk_fba')
    if chk_fba:
        params.filterByArea = True
        params.minArea=st.sidebar.number_input('MinArea',min_value=0,max_value=1000,value=int(params.minArea),step=1,format='%d',key='fba_min')
        params.maxArea=st.sidebar.number_input('MaxArea',min_value=0,max_value=1000000,value=int(params.maxArea),step=1,format='%d',key='fba_max')
    else:
        params.filterByArea = False

    # 真円度（ 4∗π∗Area / perimeter∗perimeter によって定義される）
    #（minCircularity <= blob < maxCircularity）
    chk_fbci=st.sidebar.checkbox('Filter by Circularity',value=params.filterByCircularity,key='chk_fbci')
    if chk_fbci:
        params.filterByCircularity = True
        params.minCircularity=max(0.0,params.minCircularity)
        params.maxCircularity=min(1.0,params.maxCircularity)
        values = st.sidebar.slider('Select a range of Circularity', 0.0, 1.0, (params.minCircularity, params.maxCircularity),step=0.01,format='%.2f',key='fbci_val')
        params.minCircularity=values[0]
        params.maxCircularity=values[1]
    else:
        params.filterByCircularity = False

    
    # 凸面の情報（minConvexity <= blob < maxConvexity）
    chk_fbco=st.sidebar.checkbox('Filter by Convexity',value=params.filterByConvexity,key='chk_fbco')
    if chk_fbco:
        params.filterByConvexity = True
        params.minConvexity=max(0.0,params.minConvexity)
        params.maxConvexity=min(1.0,params.maxConvexity)
        values = st.sidebar.slider('Select a range of Convexity', 0.0, 1.0, (params.minConvexity, params.maxConvexity),step=0.01,format='%.2f',key='fbco_val')
        params.minConvexity=values[0]
        params.maxConvexity=values[1]
    else:
        params.filterByConvexity = False

    # 楕円形を表す（minInertiaRatio <= blob < maxInertiaRatio）
    chk_fbi=st.sidebar.checkbox('Filter by InertiaRatio',value=params.filterByInertia,key='chk_fbi')
    if chk_fbi:
        params.filterByInertia = True
        params.minInertiaRatio=max(0.0,params.minInertiaRatio)
        params.maxInertiaRatio=min(1.0,params.maxInertiaRatio)
        values = st.sidebar.slider('Select a range of Inertia Ratio', 0.0, 1.0, (params.minInertiaRatio, params.maxInertiaRatio),step=0.01,format='%.2f',key='fbi_val')
        params.minInertiaRatio=values[0]
        params.maxInertiaRatio=values[1]
    else:
        params.filterByInertia = False




















    st.write("Choose any image and count Blobs:")

    uploaded_img=None
    if InputType==input_type[0]:
        uploaded_img_file = st.file_uploader("Choose an image...",type=['jpg', 'jpeg', 'png'])
        if uploaded_img_file is not None:
            uploaded_img=uploaded_img_file
            
    if InputType==input_type[1]:
        uploaded_img_cam = st.camera_input("Take a picture")
        if uploaded_img_cam is not None:
            uploaded_img=uploaded_img_cam

    if uploaded_img is not None:

        st.image(uploaded_img, caption='Input Image', use_column_width=True)

        img=pil2cv(Image.open(uploaded_img))
        #print(img.shape)
        #print(img.dtype)
        if color_enhancement_mode!=color_enhancement_menu[0]:
            img=color_enhancement(img)

        simple_blob(img)




