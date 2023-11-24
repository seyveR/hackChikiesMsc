import streamlit as st
from PIL import Image
from urllib.parse import urlparse
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_drawable_canvas import st_canvas
import requests
from io import BytesIO
from ultralytics import YOLO
import cv2
from moviepy.editor import VideoFileClip
import pandas as pd
import plotly.express as px

from model import image_to_base64, image_to_base64_2, paint_boxes, load_model


session_state = st.session_state

if not hasattr(session_state, 'selected_label'):
    session_state.selected_label = None

if not hasattr(session_state, 'predictions_dict'):
    session_state.predictions_dict = {}


st.set_page_config(
        page_title="Chikies",
        page_icon=":camera:",
        layout="wide",
    )


query_params = st.experimental_get_query_params()
route = query_params.get('route', ['home'])[0]

def redirect_to_page(page):
    st.experimental_set_query_params(route=page)

home_button = st.sidebar.button("Главная", key="home", on_click=lambda: redirect_to_page("home"))
history_button = st.sidebar.button("История", key="history", on_click=lambda: redirect_to_page("history"))
stats_button = st.sidebar.button("Статистика", key="stats", on_click=lambda: redirect_to_page("stats"))

data = {
    "Название видео": [],
    "В кузове машины": [],
    "Уверенность": [],
    "Ожидаемое": [],
}


if route == "home":
    background_image_url = "https://sun9-13.userapi.com/zFBSFRBBZX4x0KbGOPE0C-7-t4mz7u_UGL2YpQ/EZZFFQ4Ly-M.jpg"  
    background_image = Image.open(BytesIO(requests.get(background_image_url).content))
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url('{background_image_url}');
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
        </style>
        """,
        unsafe_allow_html=True
        )

    model = YOLO('yolovS.pt', task='detect')

    

    logo_url = "https://psv4.userapi.com/c235031/u133344394/docs/d22/bbb84a88a2cf/chikies_logo2_white22.png?extra=JK35nb1F-goarQwzVaQF6d6tXaScLkLvUaSdzlGJ8JuIgkoyUDmM9pJRiO2JFfbSJlrA-9pkqSTNkwPTUCSljkhBA6T_giVzhFCOYLagkajQYCbeaj89MSSbzo55vliWFhAjXydWPNkC3JaT2Qxz7CvT"

    response = requests.get(logo_url)
    logo_image = Image.open(BytesIO(response.content))

    second_image_url = "https://psv4.userapi.com/c909618/u133775271/docs/d56/9f1e8ef82690/logomsc.png?extra=KxLzE6XjCGLG5bEpxHcs5GkCbeh__ivjPxBC_T6UP_OqeJ7uyd5X2MQK6qluE61IteCUFFUZCD8-GBJOvpgTv7BEoojDJQKfmZ1u7v7AFk1jB7g2dnuPW8lHY2AyTUuUdJwJwcVpA18QU-3IsMNItFfpgQ"

    response_second = requests.get(second_image_url)
    second_image = Image.open(BytesIO(response_second.content))

    col1, col2 = st.columns(2)

    col1.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: flex-start; height: 200px;">
            <img src="{logo_url}" alt="Первое изображение" width="220">
        </div>
        """,
        unsafe_allow_html=True
    )

    col2.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: flex-end; height: 200px;">
            <img src="{second_image_url}" alt="Второе изображение" width="150">
        </div>
        """,
        unsafe_allow_html=True
    )

    boxes_checkbox = st.checkbox('Отрисовка мусора', value=True)
    selected_label = st.selectbox("Выберите элемент для детекции:", ['бетон', 'грунт', 'дерево', 'кирпичи'])
    spinner_style = f"max-width: 200px;"
    spinner = st.spinner("Подождите, идет обработка...")
    uploaded_files: list[UploadedFile] = st.file_uploader("Загрузите файлы", type=["mp4"], accept_multiple_files=True)

    for uploaded_file in uploaded_files:
        st.empty() 

        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as video_file:
            video_file.write(uploaded_file.read())

        clip = VideoFileClip(video_path)
        frame = clip.get_frame(130)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(frame_rgb)
        img.save('center_frame.jpg')

        model = YOLO('yolovNano.pt')
        result = model('center_frame.jpg')

        boxes = result[0].boxes
        cls = boxes.cls

        labels = ['бетон', 'кирпичи', 'грунт', 'дерево']
        video_filename = uploaded_file.name

        if video_filename not in session_state.predictions_dict:
           session_state.predictions_dict[video_filename] = {}

        if session_state.selected_label != selected_label:
           session_state.selected_label = selected_label

        real_target = session_state.selected_label


        for cls_index in cls:
            label = labels[int(cls_index.item())]
            confidence = boxes.conf[0]

            session_state.selected_label = selected_label

            session_state.predictions_dict[video_filename][label] = {
                'confidence': confidence.item(),
                'real_target': real_target
            }

            st.markdown(
                f"<p style='text-align:center; font-size:36px; font-weight: bold; margin-left:auto; margin-right:auto;'>В кузове {label}, уверенность: {confidence.item()}</p>",
                unsafe_allow_html=True
            )

            if boxes_checkbox:
                img_with_boxes = paint_boxes(frame_rgb, [result[0]])
                st.markdown(
                    f"""
                    <div style="align-items: center; height: 700px; margin-bottom:120px;">
                        <img src="data:image/png;base64,{image_to_base64_2(img_with_boxes)}" width="750">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                break
            else:
                st.markdown(
                    f"""
                    <div style="align-items: center; height: 700px; margin-bottom:120px;">
                        <img src="data:image/png;base64,{image_to_base64(img)}" width="750">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                break

elif route == 'history':
    background_image_url = "https://psv4.userapi.com/c909228/u133344394/docs/d15/e380bc28a28e/imgonline-com-ua-Replace-color-KwC7aO4m5XDHpNGg.png?extra=cvIIEvpf2PFi7bKOK2xUCq6BbA8emNxZlfluItzlcJQXkSxuOkuXXUv5mVIsa8vmwa-E-dqTxqqHYoMeb506rHOz2SVAd4wv18K3FDxGPoUxyRtNazzpmwNs-pZ5urpnOgqZ-B18JxWec3BXb7Sg6znt"  
    background_image = Image.open(BytesIO(requests.get(background_image_url).content))
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url('{background_image_url}');
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
        </style>
        """,
        unsafe_allow_html=True
        )
    
    st.markdown(
                f"<p style='text-align:center; font-size:45px; font-weight: bold; margin-left:auto; margin-right:auto;'>История</p>",
                unsafe_allow_html=True
            )
    table_data = []

    for video_filename, predictions in session_state.predictions_dict.items():
        for label, info in predictions.items():
            confidence = info['confidence']
            real_target = info['real_target']

            confidence_percent = f"{confidence * 100:.1f}%"

            color = "#37EC31" if label == real_target else "#EC1603"

            table_data.append({
                "Название видео": video_filename,
                "В кузове машины": label,
                "Уверенность": confidence_percent,
                "Ожидаемое": real_target,
                "Цвет": color
            })

    if table_data:
        keys_without_color = [key for key in table_data[0].keys() if key != "Цвет"]

        header_row = "|".join([f"<font style='font-size: 30px;'>{key}</font>" for key in keys_without_color]) + "\n"

        separator_row = "|".join(["---" for _ in keys_without_color]) + "\n"

        data_rows = ""
        for row in table_data:
            row_str = "|".join([f"<font color='{row['Цвет']}' style='font-size: 30px;'>{value}</font>" for key, value in row.items() if key != "Цвет"])
            data_rows += row_str + "\n"

        table_str = header_row + separator_row + data_rows
        st.markdown(table_str, unsafe_allow_html=True)
    else:
        st.text("История пуста")

elif route == 'stats':
    background_image_url = "https://psv4.userapi.com/c909228/u133344394/docs/d24/771e611709b3/imgonline-com-ua-Replace-color-9pK5sUX42lxq.png?extra=j1lz09qHyjHzahrcgCOYP1p2YzgXVrokw-PB6QTjivvkO-y_EBh9VCxru84xazlIjeaI2s7d_zxVb7F-oxfIgnouj9yLMFe1j5e3oL8JJIV_42es1g3yFZ7tjvD0puzmOMLAI8qPXEUMM6-_y9Dt3mCj"  
    background_image = Image.open(BytesIO(requests.get(background_image_url).content))
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url('{background_image_url}');
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"<p style='text-align:center; font-size:45px; font-weight: bold; margin-left:auto; margin-right:auto;'>Статистика</p>",
        unsafe_allow_html=True
    )

    stats_data = {
        "Название видео": [],
        "В кузове машины": [],
        "Уверенность": [],
        "Ожидаемое": [],
    }

    for video_filename, predictions in session_state.predictions_dict.items():
        for label, info in predictions.items():
            confidence = info['confidence']
            real_target = info['real_target']

            stats_data["Название видео"].append(video_filename)
            stats_data["В кузове машины"].append(label)
            stats_data["Уверенность"].append(confidence)
            stats_data["Ожидаемое"].append(real_target)

    df = pd.DataFrame(stats_data)

    df['Совпадение'] = df['В кузове машины'] == df['Ожидаемое']
    correct_percent = df['Совпадение'].sum() / len(df) * 100
    
    fig = px.pie(df, names='В кузове машины', title=f'Процент совпадений: {correct_percent:.2f}%',
                 labels={'В кузове машины': 'Категория'},
                 hover_data=['В кузове машины', 'Совпадение'],
                 hole=0.4)
    
    # Установка размера шрифта заголовка
    fig.update_layout(title_font=dict(size=40))
    
    # Увеличение размера диаграммы
    fig.update_layout(height=600, width=800)
    
    fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.1, 0.1, 0.1]) 
    fig.update_layout(legend=dict(title=''), showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)') 
    
    st.plotly_chart(fig)



