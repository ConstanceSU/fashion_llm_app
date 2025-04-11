import streamlit as st
import cohere
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)
from collections import Counter
from sklearn.cluster import KMeans

#############################
# Add Custom CSS for IBM Plex Serif
#############################

def add_custom_css():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@100;200;300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
    body {
      font-family: 'IBM Plex Serif', serif;
    }
    .ibm-plex-serif-thin {
      font-family: "IBM Plex Serif", serif;
      font-weight: 100;
      font-style: normal;
    }
    .ibm-plex-serif-extralight {
      font-family: "IBM Plex Serif", serif;
      font-weight: 200;
      font-style: normal;
    }
    .ibm-plex-serif-light {
      font-family: "IBM Plex Serif", serif;
      font-weight: 300;
      font-style: normal;
    }
    .ibm-plex-serif-regular {
      font-family: "IBM Plex Serif", serif;
      font-weight: 400;
      font-style: normal;
    }
    .ibm-plex-serif-medium {
      font-family: "IBM Plex Serif", serif;
      font-weight: 500;
      font-style: normal;
    }
    .ibm-plex-serif-semibold {
      font-family: "IBM Plex Serif", serif;
      font-weight: 600;
      font-style: normal;
    }
    .ibm-plex-serif-bold {
      font-family: "IBM Plex Serif", serif;
      font-weight: 700;
      font-style: normal;
    }
    .ibm-plex-serif-thin-italic {
      font-family: "IBM Plex Serif", serif;
      font-weight: 100;
      font-style: italic;
    }
    .ibm-plex-serif-extralight-italic {
      font-family: "IBM Plex Serif", serif;
      font-weight: 200;
      font-style: italic;
    }
    .ibm-plex-serif-light-italic {
      font-family: "IBM Plex Serif", serif;
      font-weight: 300;
      font-style: italic;
    }
    .ibm-plex-serif-regular-italic {
      font-family: "IBM Plex Serif", serif;
      font-weight: 400;
      font-style: italic;
    }
    .ibm-plex-serif-medium-italic {
      font-family: "IBM Plex Serif", serif;
      font-weight: 500;
      font-style: italic;
    }
    .ibm-plex-serif-semibold-italic {
      font-family: "IBM Plex Serif", serif;
      font-weight: 600;
      font-style: italic;
    }
    .ibm-plex-serif-bold-italic {
      font-family: "IBM Plex Serif", serif;
      font-weight: 700;
      font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

#############################
# Global Dictionaries & Setup
#############################

# CSS3 colors dictionary
CSS3_COLORS = {
    "aliceblue": (240, 248, 255),
    "antiquewhite": (250, 235, 215),
    "aqua": (0, 255, 255),
    "aquamarine": (127, 255, 212),
    "azure": (240, 255, 255),
    "beige": (245, 245, 220),
    "bisque": (255, 228, 196),
    "black": (0, 0, 0),
    "blanchedalmond": (255, 235, 205),
    "blue": (0, 0, 255),
    "blueviolet": (138, 43, 226),
    "brown": (165, 42, 42),
    "burlywood": (222, 184, 135),
    "cadetblue": (95, 158, 160),
    "chartreuse": (127, 255, 0),
    "chocolate": (210, 105, 30),
    "coral": (255, 127, 80),
    "cornflowerblue": (100, 149, 237),
    "cornsilk": (255, 248, 220),
    "crimson": (220, 20, 60),
    "cyan": (0, 255, 255),
    "darkblue": (0, 0, 139),
    "darkcyan": (0, 139, 139),
    "darkgoldenrod": (184, 134, 11),
    "darkgray": (169, 169, 169),
    "darkgreen": (0, 100, 0),
    "darkkhaki": (189, 183, 107),
    "darkmagenta": (139, 0, 139),
    "darkolivegreen": (85, 107, 47),
    "darkorange": (255, 140, 0),
    "darkorchid": (153, 50, 204),
    "darkred": (139, 0, 0),
    "darksalmon": (233, 150, 122),
    "darkseagreen": (143, 188, 143),
    "darkslateblue": (72, 61, 139),
    "darkslategray": (47, 79, 79),
    "darkturquoise": (0, 206, 209),
    "darkviolet": (148, 0, 211),
    "deeppink": (255, 20, 147),
    "deepskyblue": (0, 191, 255),
    "dimgray": (105, 105, 105),
    "dodgerblue": (30, 144, 255),
    "firebrick": (178, 34, 34),
    "floralwhite": (255, 250, 240),
    "forestgreen": (34, 139, 34),
    "fuchsia": (255, 0, 255),
    "gainsboro": (220, 220, 220),
    "ghostwhite": (248, 248, 255),
    "gold": (255, 215, 0),
    "goldenrod": (218, 165, 32),
    "gray": (128, 128, 128),
    "green": (0, 128, 0),
    "greenyellow": (173, 255, 47),
    "honeydew": (240, 255, 240),
    "hotpink": (255, 105, 180),
    "indianred": (205, 92, 92),
    "indigo": (75, 0, 130),
    "ivory": (255, 255, 240),
    "khaki": (240, 230, 140),
    "lavender": (230, 230, 250),
    "lavenderblush": (255, 240, 245),
    "lawngreen": (124, 252, 0),
    "lemonchiffon": (255, 250, 205),
    "lightblue": (173, 216, 230),
    "lightcoral": (240, 128, 128),
    "lightcyan": (224, 255, 255),
    "lightgoldenrodyellow": (250, 250, 210),
    "lightgray": (211, 211, 211),
    "lightgreen": (144, 238, 144),
    "lightpink": (255, 182, 193),
    "lightsalmon": (255, 160, 122),
    "lightseagreen": (32, 178, 170),
    "lightskyblue": (135, 206, 250),
    "lightslategray": (119, 136, 153),
    "lightsteelblue": (176, 196, 222),
    "lightyellow": (255, 255, 224),
    "lime": (0, 255, 0),
    "limegreen": (50, 205, 50),
    "linen": (250, 240, 230),
    "magenta": (255, 0, 255),
    "maroon": (128, 0, 0),
    "mediumaquamarine": (102, 205, 170),
    "mediumblue": (0, 0, 205),
    "mediumorchid": (186, 85, 211),
    "mediumpurple": (147, 112, 219),
    "mediumseagreen": (60, 179, 113),
    "mediumslateblue": (123, 104, 238),
    "mediumspringgreen": (0, 250, 154),
    "mediumturquoise": (72, 209, 204),
    "mediumvioletred": (199, 21, 133),
    "midnightblue": (25, 25, 112),
    "mintcream": (245, 255, 250),
    "mistyrose": (255, 228, 225),
    "moccasin": (255, 228, 181),
    "navajowhite": (255, 222, 173),
    "navy": (0, 0, 128),
    "oldlace": (253, 245, 230),
    "olive": (128, 128, 0),
    "olivedrab": (107, 142, 35),
    "orange": (255, 165, 0),
    "orangered": (255, 69, 0),
    "orchid": (218, 112, 214),
    "palegoldenrod": (238, 232, 170),
    "palegreen": (152, 251, 152),
    "paleturquoise": (175, 238, 238),
    "palevioletred": (219, 112, 147),
    "papayawhip": (255, 239, 213),
    "peachpuff": (255, 218, 185),
    "peru": (205, 133, 63),
    "pink": (255, 192, 203),
    "plum": (221, 160, 221),
    "powderblue": (176, 224, 230),
    "purple": (128, 0, 128),
    "rebeccapurple": (102, 51, 153),
    "red": (255, 0, 0),
    "rosybrown": (188, 143, 143),
    "royalblue": (65, 105, 225),
    "saddlebrown": (139, 69, 19),
    "salmon": (250, 128, 114),
    "sandybrown": (244, 164, 96),
    "seagreen": (46, 139, 87),
    "seashell": (255, 245, 238),
    "sienna": (160, 82, 45),
    "silver": (192, 192, 192),
    "skyblue": (135, 206, 235),
    "slateblue": (106, 90, 205),
    "slategray": (112, 128, 144),
    "snow": (255, 250, 250),
    "springgreen": (0, 255, 127),
    "steelblue": (70, 130, 180),
    "tan": (210, 180, 140),
    "teal": (0, 128, 128),
    "thistle": (216, 191, 216),
    "tomato": (255, 99, 71),
    "turquoise": (64, 224, 208),
    "violet": (238, 130, 238),
    "wheat": (245, 222, 179),
    "white": (255, 255, 255),
    "whitesmoke": (245, 245, 245),
    "yellow": (255, 255, 0),
    "yellowgreen": (154, 205, 50),
    "Crimson": (220, 20, 60),
    "DarkOrange": (255, 140, 0),
    "Gold": (255, 215, 0),
    "ForestGreen": (34, 139, 34),
    "DodgerBlue": (30, 144, 255),
    "MediumOrchid": (186, 85, 211),
    "DarkSeaGreen": (143, 188, 143),
    "LemonChiffon": (255, 250, 205),
    "PaleTurquoise": (175, 238, 238),
    "LightSteelBlue": (176, 196, 222),
    "Lavender": (230, 230, 250)
}

# Relevant clothing labels
RELEVANT_LABELS = {
    "T-shirt", "shirt", "jersey", "sweater", "hoodie", "cardigan",
    "jacket", "coat", "blazer", "trench_coat", "lab_coat", "fur_coat",
    "bulletproof_vest", "poncho", "parka", "jean", "trouser", "shorts",
    "miniskirt", "dress", "gown", "Windsor_tie", "bow_tie", "mortarboard",
    "baseball_cap", "cowboy_hat", "sombrero", "hat_with_a_wide_brim", "hood",
    "scarf", "belt", "glove", "purse", "handbag", "backpack", "sandal",
    "clog", "running_shoe", "Loafer", "sneaker", "boot", "hiking_boot",
    "cowboy_boot", "bikini", "swimming_trunks", "pajama", "sunglass",
    "sunglasses", "mask", "shower_cap"
}

#############################
# Utility Functions
#############################
def closest_color(requested_rgb):
    min_dist = float('inf')
    closest_name = None
    for color_name, rgb in CSS3_COLORS.items():
        dist = sum((comp1 - comp2) ** 2 for comp1, comp2 in zip(requested_rgb, rgb))
        if dist < min_dist:
            min_dist = dist
            closest_name = color_name
    return closest_name

def detect_dominant_color(image, top_n=1):
    image = image.convert("RGBA")
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    width, height = image.size
    border = int(min(width, height) * 0.1)
    image = image.crop((border, border, width - border, height - border))
    image = image.resize((50, 50))
    pixels = list(image.getdata())
    filtered_pixels = []
    for (r, g, b) in pixels:
        brightness = (r + g + b) / 3
        if 20 < brightness < 240:
            filtered_pixels.append((r, g, b))
    if len(filtered_pixels) == 0:
        filtered_pixels = pixels
    most_common_pixels = Counter(filtered_pixels).most_common(top_n)
    color_names = []
    for (r, g, b), _ in most_common_pixels:
        color_names.append(closest_color((r, g, b)))
    return color_names

def dominant_color_kmeans(image, k=3):
    image = image.convert("RGB")
    arr = np.array(image)
    h, w, _ = arr.shape
    arr = arr.reshape((h * w, 3))
    kmeans = KMeans(n_clusters=k, n_init="auto")
    kmeans.fit(arr)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = labels[np.argmax(counts)]
    r, g, b = kmeans.cluster_centers_[dominant_cluster]
    return closest_color((int(r), int(g), int(b)))

@st.cache_resource
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    x = np.array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def predict_image(model, preprocessed_img):
    preds = model.predict(preprocessed_img)
    decoded = decode_predictions(preds, top=3)[0]
    return decoded

def display_wardrobe_items():
    st.markdown("<h2 style='text-align: center;'>My Virtual Wardrobe</h2>", unsafe_allow_html=True)
    wardrobe = st.session_state.wardrobe
    for row_start in range(0, len(wardrobe), 3):
        row_items = wardrobe[row_start : row_start + 3]
        cols = st.columns(len(row_items))
        for col, item in zip(cols, row_items):
            with col:
                st.subheader(f"{item['label']}")
                st.write(f"Color: {item['color']}")
                st.image(item["image"], caption=item["label"], width=300)
                new_name = st.text_input("Rename item:", value=item["label"], key=f"rename_input_{id(item)}")
                if st.button("Save name", key=f"save_name_{id(item)}"):
                    item["label"] = new_name
                    st.experimental_rerun()
                if st.button(f"Remove '{item['label']}'", key=f"remove_{id(item)}"):
                    wardrobe.remove(item)
                    st.experimental_rerun()

def build_fashion_prompt(user_style_text, wardrobe_list):
    prompt = "You are a helpful AI fashion assistant. The user has these items in their virtual wardrobe:\n"
    for idx, item in enumerate(wardrobe_list):
        prompt += f"{idx+1}. {item['label']}\n"
    prompt += f"\nThe user says: '{user_style_text}'.\n"
    prompt += ("Suggest an outfit combination from these items that fits their style/event. "
               "You can also include general style tips. Be concise and clear.\n\nAnswer:\n")
    return prompt

def load_cohere_key():
    with open("cohere.key", "r") as f:
        return f.read().strip()

#############################
# Cover & Layout CSS
#############################
def set_fullscreen_layout():
    st.set_page_config(layout="wide")
    st.markdown("""
    <style>
    /* Make anchor jumps smooth */
    html {
        scroll-behavior: smooth;
    }
    .block-container {
        padding: 0 !important;
    }
    header[data-testid="stHeader"] {
        display: none;
    }
    .stToolbar {
        display: none;
    }
    html, body, [data-testid="stAppViewContainer"] {
        margin: 0;
        padding: 0;
        height: 100%;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

def add_cover_css():
    st.markdown("""
    <style>
    .cover-container {
        position: relative;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        text-align: center;
        color: white;
        margin: 0;
        padding: 0;
    }
    .cover-img {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center;
        z-index: 1;
    }
    .cover-title {
        position: absolute;
        top: 35%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 5rem;
        font-weight: 600;
        z-index: 2;
        margin: 0;
        padding: 0;
    }
    .button-container {
        position: absolute;
        top: 55%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 2;
    }
    .build-btn {
        background-color: #ffffffcc;
        color: #000;
        font-size: 1.8rem;
        padding: 1rem 1.5rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .build-btn:hover {
        background-color: #fff;
    }
    </style>
    """, unsafe_allow_html=True)

#############################
# Cover Section (Top of the Page)
#############################
def show_cover_section():
    # Load and encode the background image
    with open("FSH-1738674621600-desktop-005.jpg", "rb") as f:
        img_data = f.read()
    encoded_img = base64.b64encode(img_data).decode()

    st.markdown(f"""
    <div class="cover-container">
        <img class="cover-img" src="data:image/jpeg;base64,{encoded_img}" />
        <div class="button-container">
            <!-- Jump to the anchor #main-app below -->
            <a href="#main-app">
                <button class="build-btn">Build your virtual closet</button>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

#############################
# Main Section (Below the Cover)
#############################
def show_main_section():
    # Anchor for smooth scrolling
    st.markdown('<div id="main-app"></div>', unsafe_allow_html=True)

    # A custom heading with left padding
    st.markdown("<h2 style='text-align: center'>Upload your clothing images</h2>", unsafe_allow_html=True)

    # ~~~~~~~~~~~~~~~~~~~~~~
    # 1) Display 4 sample items with extra padding + plus sign
    # ~~~~~~~~~~~~~~~~~~~~~~
    col_left, col1, col2, col3, col4, col_plus, col_right = st.columns([0.4,1,1,1,1,0.4,0.4])

    with col1:
        st.image("sample_1.png", width=200)
    with col2:
        st.image("sample_2.png", width=200)
    with col3:
        st.image("sample_3.png", width=200)
    with col4:
        st.image("sample_4.png", width=200)
    with col_plus:
        st.markdown("<div style='font-size: 60px; line-height: 150px; text-align: center;'>+</div>", 
                    unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

    # ~~~~~~~~~~~~~~~~~~~~~~
    # 2) File Uploader
    # ~~~~~~~~~~~~~~~~~~~~~~
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    st.write("Drag and drop or browse files below to add your own items:")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    # ~~~~~~~~~~~~~~~~~~~~~~
    # 3) Process uploaded image
    # ~~~~~~~~~~~~~~~~~~~~~~
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Create two columns for layout
        colA, colB = st.columns([1, 2])
        
        # Left column: Display the uploaded image
        with colA:
            st.image(image, caption="Uploaded Clothing Item", width=300)

        # Right column: Display recognition results & radio button
        with colB:
            st.subheader("Image Recognition Results")
            
            # Preprocess & predict with the loaded model
            model = load_model()
            preprocessed = preprocess_image(image)
            predictions = predict_image(model, preprocessed)
            
            # Filter predictions based on relevant clothing labels
            filtered_predictions = [p for p in predictions if p[1] in RELEVANT_LABELS]
            if not filtered_predictions:
                filtered_predictions = predictions
            
            for i, (imagenet_id, label, prob) in enumerate(filtered_predictions):
                st.write(f"{i+1}. **{label}** with confidence {prob:.2f}")
            
            # Provide a radio button for user selection
            prediction_options = [
                f"{label} (confidence: {prob:.2f})"
                for imagenet_id, label, prob in filtered_predictions
            ]
            selected_prediction = st.radio("Select the best identified object:", options=prediction_options)
            
            # Button to add the item to the virtual wardrobe
            if st.button("Add the object to my virtual wardrobe"):
                try:
                    label_part, conf_part = selected_prediction.split("(confidence:")
                    label_part = label_part.strip()
                    conf_part = conf_part.replace(")", "").strip()
                    confidence_val = float(conf_part)
                except Exception:
                    label_part = selected_prediction
                    confidence_val = 0.0
                
                # Detect dominant color
                color_names = detect_dominant_color(image, top_n=1)
                color_label = color_names[0] if color_names else "unknown"
                
                # Store in session state
                if "wardrobe" not in st.session_state:
                    st.session_state.wardrobe = []
                wardrobe_item = {
                    "image": image,
                    "label": label_part,
                    "color": color_label,
                    "confidence": confidence_val
                }
                st.session_state.wardrobe.append(wardrobe_item)
                st.success("Item added to your virtual wardrobe!")

    # ~~~~~~~~~~~~~~~~~~~~~~
    # 4) Display Virtual Wardrobe (if any)
    # ~~~~~~~~~~~~~~~~~~~~~~
    if "wardrobe" in st.session_state and st.session_state.wardrobe:
        display_wardrobe_items()

    # ~~~~~~~~~~~~~~~~~~~~~~
    # 5) LLM-based Outfit Recommendation (hidden until wardrobe has items)
    # ~~~~~~~~~~~~~~~~~~~~~~
    if "wardrobe" in st.session_state and len(st.session_state.wardrobe) > 0:
        # Create a three-column layout to center the elements
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.markdown("<h2 style='text-align: center;'>Describe Your Desired Style</h2>", unsafe_allow_html=True)
            
            # Create a flex container to center the text input and button in one row
            st.markdown("""
                <div style="display: flex; justify-content: center; align-items: center; gap: 0.75rem;">
            """, unsafe_allow_html=True)
            
            # Place the text input and button in the same row; hide the label for the text input
            style_prompt = st.text_input(
                label="", 
                placeholder="Enter the style you are looking for (e.g. 'I have an in-person interview tmr')",
                label_visibility="collapsed"
            )
            generate_button = st.button("Generate Outfit Recommendation")
            
            # Close the flex container
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add margin under the input/button only if the button has not been clicked
            if not generate_button:
                st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)
            
            if generate_button:
                prompt = build_fashion_prompt(style_prompt, st.session_state.wardrobe)
                co = cohere.Client(load_cohere_key())
                response = co.generate(
                    model="command-xlarge",
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.7,
                    k=0,
                    p=0.75
                )
                llm_output = response.generations[0].text.strip()
                st.write(llm_output)
                
                # Add margin under the generated output
                st.markdown("<div style='margin-bottom: 80px;'></div>", unsafe_allow_html=True)

    else:
        st.info("Your wardrobe is empty. Please add items to unlock outfit recommendations.")
        st.markdown("<div style='margin-bottom: 160px;'></div>", unsafe_allow_html=True)

#############################
# Main Control Flow
#############################
def main():
    set_fullscreen_layout()
    add_cover_css()
    show_cover_section()
    show_main_section()

if __name__ == "__main__":
    main()