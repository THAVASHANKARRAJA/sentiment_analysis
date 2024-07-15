from tensorflow.keras.models import load_model
from preprocessing import preprocess_reviews, image_path
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_elements import elements, mui, html, sync
import pickle
import os
import cv2
import pandas as pd
import emoji
import numpy as np
import streamlit.components.v1 as components

model = load_model('models/LSTM.h5')
token = pickle.load((open('token.pkl', 'rb')))
max_len = 130
st.set_page_config (layout="wide")
image_paths_individual = pd.read_csv("image_path.csv")


def find_star(result):
    if result >= 0.8:
        st.markdown("<span style='color:green;font-size: 80px'>* * * * *</span>", unsafe_allow_html=True)
    elif result>0.65:
        st.markdown("<span style='color:green;font-size: 80px'>* * * *</span>", unsafe_allow_html=True)
    elif result>0.5:
        st.markdown("<span style='color:white;font-size: 80px'>* * *</span>", unsafe_allow_html=True)
    elif result>0.3:
        st.markdown("<span style='color:red;font-size: 80px'>* *</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:red;font-size: 80px'>*</span>", unsafe_allow_html=True)
        
def no_star(result):
    st.success("Thank you for your review!")
    if result >= 0.8:
        return 5
    elif result >0.65:
        #st.markdown("<span style='color:grreen;font-size: 80px'>* * * *</span>", unsafe_allow_html=True)
        return 4
    elif result>0.5:
        #st.markdown("<span style='color:white;font-size: 80px'>* * *</span>", unsafe_allow_html=True)
        return 3
    elif result>0.3:
        #st.markdown("<span style='color:red;font-size: 80px'>* *</span>", unsafe_allow_html=True)
        return 2
    else:
        #st.markdown("<span style='color:red;font-size: 80px'>*</span>", unsafe_allow_html=True)
        return 1

def display_images_from_folder(folder_path):
    image_extensions = ["png", "jpg", "jpeg", "gif"]

    image_files = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if any(filename.lower().endswith(ext) for ext in image_extensions)
    ]
    num_images = len(image_files)
    num_columns = 2

    for i in range(0, num_images, num_columns):
        images_row = image_files[i:i + num_columns]
        col1,col2 = st.columns([100,100])
        for img_path, col_elem in zip(images_row, [col1,col2]):
            imagee = cv2.imread(img_path)
            imagee = cv2.resize(imagee,[640,480])
            cv2.imshow('Image', imagee)
            col_elem.image(imagee, use_column_width=True)
            caption = os.path.basename(img_path).split('.')[0]
            with col_elem:

                st.markdown(f"<span style='text-align: center; color: yellow;font-size: 30px'>{caption}</span>", unsafe_allow_html=True)
                
def sentiment(result):
    if result<0.35:
        return 'Negative'
    elif result>=0.35 and result<0.65:
        return 'Neutral'
    else:
        return 'Positive'
Review_customer = {"Product_Name":"","Review":"","Sentiment":"","Star":""}
review_data = pd.read_csv("Review_customer.csv")


def write_review(selected_product,review_data):

    st.markdown("""
    <style>
    .image {
    background-image: url("https://www.wallpapertip.com/wmimgs/83-838362_web-developer.jpg");
    background-size: cover;
    }
    .image img {
    border-radius:5px;
    height:10px;
    width:10px;
    }
    </style>
    """, unsafe_allow_html=True)

    

    img_path = image_path(selected_product)
    print(img_path)
    imagee = cv2.imread(img_path)

    imagee = cv2.resize(imagee,[640,480])
    cv2.imshow('Image', imagee)
    st.image(imagee, use_column_width=True)

    st.button("Buy now")
    
    product_description(selected_product)
    overall = np.round(review_data[review_data['Product_Name']==selected_product]['Star'].apply(overall_star).mean(),1)
    overall = str(overall) + emoji.emojize( ':star:')
    st.markdown(f"<span style='color:yellow;font-size: 30px'>Overall Rating : {overall}</span>", unsafe_allow_html=True)

    st.markdown("<span style='color:white;font-size: 20px'>Enter the review</span>", unsafe_allow_html=True)

    review = st.text_area("Enter your review",label_visibility = 'collapsed')
    review = str(review)
    if st.button("Find Rating "):
        review_encoded = preprocess_reviews(review, max_len,token)
        result = model.predict(review_encoded)
        find_star(result)
        Review_customer["Sentiment"] = (str(sentiment(result)))
        Review_customer["Product_Name"] = (str(selected_product))
        Review_customer["Review"] = (str(review))
        Review_customer["Star"] = str(no_star(result)) + emoji.emojize(" :star:")
        review_data.loc[review_data.shape[0]] = Review_customer
        review_data = review_data[["Product_Name","Review","Sentiment","Star"]]
        review_data.to_csv("Review_customer.csv")

def product_description(selected_product):
    if selected_product == "American Tourist Bag":

        st.markdown("<h2 style='text-align: center; color: white;font-size: 30px'>Red Graphic American Tourist Backpack</h2>", unsafe_allow_html=True)
        st.markdown("<span style='text-align: center; color: white;font-size: 30px'>Price: Rs. 900</span>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div style="color: white;">

            **PRODUCT DETAILS**: 
            Red graphic backpack  
            Non-Padded haul loop  
            2 main compartments with zip closure  
            Padded Mesh back  
            Padded shoulder strap: Padded  
            Water-resistance: No  

            ---  
            **Size & Fit**  
            Height: 48 cm  
            Width: 33 cm  
            Depth: 29 cm  

            ---  
            **Material & Care**  
            Polyester  
            Wipe with a clean, dry cloth to remove dust  

            ---  
            **Specifications**  
            Back: Padded Mesh  
            Compartment Closure: Zip  
            Haul Loop Type: Non-Padded  
            Material: Polyester  
            Number of Main Compartments: 2  
            Number of Zips: 1  
            Occasion: Casual  
            Padded Shoulder Strap: Padded  

            ---  
            [See More](https://example.com)  <!-- Replace the link with your desired URL -->

            </div>
            """,
            unsafe_allow_html=True
        )
        return True
    if selected_product == "Iphone 15 Pro":
        #st.title("")
        st.markdown("<h2 style='text-align: center; color: white;font-size: 30px'>IPhone 15 Pro</h2>", unsafe_allow_html=True)
        st.markdown("<span style='text-align: center; color: white;font-size: 30px'>Price: Rs. 125900</span>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div style="color: white;">

            **PRODUCT DETAILS**
            iPhone 15 Pro  
            Super Retina XDR display  
            A15 Bionic chip  
            Triple-camera system with Ultra Wide, Wide, and Telephoto  
            Face ID  
            Water and dust resistance  

            ---  
            **Size & Weight**  
            Height: 146.7 mm  
            Width: 71.5 mm  
            Depth: 7.7 mm  
            Weight: 187 grams  

            ---  
            **Features**  
            Super Retina XDR display  
            Ceramic Shield front cover  
            Pro camera system  
            LiDAR Scanner  
            Night mode  
            5G capable  

            ---  
            **Battery Life**  
            Up to 75 hours audio playback  
            Up to 19 hours talk time (5G)  
            Up to 75 hours audio playback  

            ---  
            [See More](https://example.com)  <!-- Replace the link with your desired URL -->

            </div>
            """,
            unsafe_allow_html=True
        )
        return True
    if selected_product == "Refrigerator (Samsung)":
        #st.title("Samsung Refrigerator")
        st.markdown("<h2 style='text-align: center; color: white;font-size: 30px'>Samsung Refrigerator</h2>", unsafe_allow_html=True)
        st.markdown("<span style='text-align: center; color: white;font-size: 30px'>Price: Rs. 22590</span>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div style="color: white;">

            **PRODUCT DETAILS**
            Samsung Refrigerator  
            Model: XYZ1234  
            Capacity: 600 liters  
            Energy Efficiency: A++  
            Frost-free  

            ---  
            **Key Features**  
            Twin Cooling Plus System  
            All-around Cooling  
            Digital Inverter Technology  
            Power Cool and Power Freeze  
            LED Lighting  

            ---  
            **Dimensions**  
            Height: 1780 mm  
            Width: 910 mm  
            Depth: 716 mm  

            ---  
            **Additional Information**  
            External Water Dispenser  
            Multi-ventilation  
            Smart Conversion  

            ---  
            [See More](https://example.com)  <!-- Replace the link with your desired URL -->

            </div>
            """,
            unsafe_allow_html=True
        )
        return True
    if selected_product == "Amazon Echo":
        st.markdown("<h2 style='text-align: center; color: white;font-size: 30px'>Amazon Echo</h2>", unsafe_allow_html=True)
        st.markdown("<span style='text-align: center; color: white;font-size: 30px'>Price: Rs. 27090</span>", unsafe_allow_html=True)
        
        st.markdown(
            """
            
            <div style="color: white;">

            **PRODUCT DETAILS**
            Amazon Echo Smart Speaker  
            Model: Echo (4th generation)  
            Voice Assistant: Alexa  
            Smart Home Integration  
            Streaming Music and Podcasts  

            ---  
            **Key Features**  
            Powerful speakers with Dolby processing  
            Voice control your smart home  
            Stream music from popular services  
            Make hands-free calls  

            ---  
            **Design**  
            Modern spherical design  
            Fabric finish  
            Available in different colors  

            ---  
            **Connectivity**  
            Wi-Fi and Bluetooth compatible  

            </div>
            """,
            unsafe_allow_html=True
        )
        return True
    if selected_product == "Dell XPS Laptop":
        st.markdown("<h2 style='text-align: center; color: white;font-size: 30px'>Dell XPS Laptop</h2>", unsafe_allow_html=True)
        st.markdown("<span style='text-align: center; color: white;font-size: 30px'>Price: Rs. 67500</span>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div style="color: white;">

            **PRODUCT DETAILS**
            Dell XPS Laptop  
            Model: XPS 13  
            Display: 13.4-inch FHD+ InfinityEdge  
            Processor: Intel Core i7  
            RAM: 16 GB  
            Storage: 512 GB SSD  
            Operating System: Windows 10  

            ---  
            **Key Features**  
            Stunning 4-sided InfinityEdge display  
            Powerful performance with Intel Core i7  
            Ultra-slim design  
            Backlit keyboard  
            Dell Cinema for immersive entertainment  

            ---  
            **Connectivity**  
            Thunderbolt 3  
            USB-C  
            MicroSD card reader  
            Headphone jack  

            ---  
            **Battery Life**  
            Up to 14 hours   

            </div>
            
            """,
            unsafe_allow_html=True
        )
        
def overall_star(star):
    return int(star.split()[0])
                   

def main(selection):
    #st.sidebar.title("Sidebar Menu")
    #selection = st.sidebar.radio("Go to", ["Home", "Product List", "Contact", "Settings"])

    if selection == "Home":
        home()
    elif selection == "Products":
        product_list()
    elif selection == "Write Review":
        st.markdown("<h1 style='text-align: center; color: white;'>Write your review</h1>", unsafe_allow_html=True)
        # st.markdown("<span style='color:white;font-size: 20px'>Select a product to give your review</span>", unsafe_allow_html=True)
        selected_product = st.selectbox("",image_paths_individual['Product_Name'].unique())
        write_review(selected_product,review_data)
        st.dataframe(review_data[review_data['Product_Name']==selected_product][["Product_Name","Review","Star"]],use_container_width= True)
        
    elif selection == "Settings":
        settings()

def product_list():
    # st.markdown("<h3>Search here....</h3>", unsafe_allow_html=True)
    Product_List_Search()
    page_element="""
    <style>
    [data-testid="stAppViewContainer"]{
    background-image: url("https://ibb.co/GpXNWmB");
    background-size: cover;
    }
    </style>
    """
    st.markdown(page_element, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white;'>Products</h1>", unsafe_allow_html=True)
    # folder_path = "products/products"
    # display_images_from_folder(folder_path)
    Products()
def Products():
    components.html(
        """<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
.body{
background-color:#0e1117;
 height:70%;
 width:20%;}
.container {
margin-top:6%;
margin-bottom:2.5%;
  position: relative;
  width: 50%;
  margin-left:20%;
}

.image {
  display: block;
  width: 100%;
  height: auto%;
  background-color:#0e1117;
  border-radius:10px;
}
 .gallery {
    display: grid;
    grid-template-columns: repeat(2, 1fr); 
    gap: 20px; 
    background-color:#0e1117;
  
   
  }

.overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background-color: #008CBA;
  overflow: hidden;
  width: 100%;
  height: 100%;
  -webkit-transform: scale(0);
  -ms-transform: scale(0);
  transform: scale(0);
  -webkit-transition: .3s ease;
  transition: .3s ease;
}

.container:hover .overlay {
  -webkit-transform: scale(1);
  -ms-transform: scale(1);
  transform: scale(1);
}

.text {
  color: white;
  font-size: 20px;
  position: absolute;
  top: 50%;
  left: 50%;
  -webkit-transform: translate(-50%, -50%);
  -ms-transform: translate(-50%, -50%);
  transform: translate(-50%, -50%);
  text-align: center;
}
</style>
</head>
<body>
<div  class = "gallery">
<div class="container">
  <img src="https://img.freepik.com/free-photo/black-water-bottle-mockup-off-white-background_53876-105917.jpg?t=st=1709891721~exp=1709895321~hmac=471b5bbec96a37c4f80160f058431e945a269694171021b3521fb7bc325f6794&w=360" alt="Avatar" class="image">
  <div class="overlay">
    <div class="text">Smart bottle</div>
  </div>
</div>
<div class="container">
  <img src="https://img.freepik.com/free-photo/elegant-smartphone-composition_23-2149437106.jpg?t=st=1709880852~exp=1709884452~hmac=dc433974773e5a850c1ce9afad389b50cc6135df692d308626a9999d62948a5c&w=360" alt="Avatar" class="image">
  <div class="overlay">
    <div class="text">Iphone 12</div>
  </div>
</div>
<div class="container">
  <img src="https://img.freepik.com/free-photo/makeup-lipstick_23-2148109477.jpg?t=st=1709891883~exp=1709895483~hmac=2798f12e24204b00b579f96cc6b0c356f9d5729a35ab9afbfc95abc1375477a4&w=360" alt="Avatar" class="image">
  <div class="overlay">
    <div class="text">Matte Lip shade</div>
  </div>
</div>
<div class="container">
  <img src="https://img.freepik.com/free-photo/headphones-with-minimalist-monochrome-background_23-2150763315.jpg?t=st=1709892161~exp=1709895761~hmac=dcbfa7b66b9b10a6f095536fe518b114fb9d6e0668997c413ed995733cc83e7e&w=360" alt="Avatar" class="image">
  <div class="overlay">
    <div class="text">Airpods Max</div>
  </div>
</div>
<div class="container">
  <img src="https://img.freepik.com/free-photo/earphones-with-minimalist-monochrome-background_23-2150763361.jpg?t=st=1709892321~exp=1709895921~hmac=f16b293de26e152ea7ef1b75860c5f4535dbef9cbeb7b5a66171e419cdb805bf&w=360" alt="Avatar" class="image">
  <div class="overlay">
    <div class="text">Airpods pro</div>
  </div>
  

</body>
</html>
""",height=1890,width=1500
    )

    
    
def Product_List_Search():
    components.html(
        """ <!DOCTYPE html>
   <html>
   <head>
   <meta name="viewport" content="width=device-width, initial-scale=1">
   <style>
   body {
    padding: 0;
    margin: 0;
    height: 100vh;
    width: 100%;
    background-color: #07051a;
}

form{
    position: relative;
    top: 50%;
    left: 50%;
    transform: translate(-50%,-50%);
    transition: all 1s;
    width: 50px;
    height: 50px;
    background: white;
    box-sizing: border-box;
    border-radius: 25px;
    border: 4px solid white;
    padding: 5px;
}

input{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;;
    height: 42.5px;
    line-height: 30px;
    outline: 0;
    border: 0;
    display: none;
    font-size: 1em;
    border-radius: 20px;
    padding: 0 20px;
}

.fa{
    box-sizing: border-box;
    padding: 10px;
    width: 42.5px;
    height: 42.5px;
    position: absolute;
    top: 0;
    right: 0;
    border-radius: 50%;
    color: #07051a;
    text-align: center;
    font-size: 1.2em;
    transition: all 1s;
}

form:hover,
form:valid{
    width: 500px;
    cursor: pointer;
}

form:hover input,
form:valid input{
    display: block;
}

form:hover .fa,
form:valid .fa{
    background: #07051a;
    color: white;
}


a {
  display: none;
  position: absolute;
  top: 70px;
  bottom:0;
  left: 0;
  right: 0;
  font-size: 20px;
  color: white;
  text-align: center; 
  width: 100%;
}

form:valid a {
  display: block;
}
.text_search{
color:white;
font-size: 30px;
font-weight: 300;
}
   </style>
   </head>
   <body>

   <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.3.1/css/all.css" integrity="sha384-mzrmE5qonljUremFsqc01SB46JvROS7bZs3IO2EmfFsd15uHvIt+Y8vEf7N7fWAU" crossorigin="anonymous">

<form action="">
  <input type="search" required placeholder="Search Here...">
  <i class="fa fa-search"></i>
  <a href="javascript:void(0)" id="clear-btn" style="text-decoration:none;font-weight:bold;color:red">CLEAR</a>
</form>

   <script>
   
const clearInput = () => {
  const input = document.getElementsByTagName("input")[0];
  input.value = "";
}

const clearBtn = document.getElementById("clear-btn");
clearBtn.addEventListener("click", clearInput);


   </script>

   </body>
   </html> 
   """
        ,
        height=200,
    )
def home():

    st.markdown("<h1 style='text-align: center; color: white;'>Product Review Analysis</h1>", unsafe_allow_html=True)


    
    page_element="""
    <style>
    [data-testid="stAppViewContainer"]{
    background-image: url("https://www.wallpapertip.com/wmimgs/83-838362_web-developer.jpg");
    background-size: cover;
    }
    </style>
    """

    st.markdown(page_element, unsafe_allow_html=True)
    slideshow_swipeable()
    footer()


def slideshow_swipeable():
    components.html(
     """ <!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {box-sizing: border-box;}
body {font-family: Verdana, sans-serif;
background-color:#494a4a;
border-radius:10px;
}
.mySlides
 {display: none;
height:20%;
width:20%;
border-radius:40px;}
img {vertical-align: middle;}

/* Slideshow container */
.slideshow-container {
  max-width: 1000px;
  position: relative;
  margin: auto;
  border-radius:10px;
}

/* Caption text */
.text {
  color: #f2f2f2;
  font-size: 30px;
  padding: 8px 12px;
  position: absolute;
  bottom: 8px;
  width: 100%;
  text-align: center;
}

/* Number text (1/3 etc) */
.numbertext {
  color: #f2f2f2;
  font-size: 12px;
  padding: 8px 12px;
  position: absolute;
  top: 0;
}

/* The dots/bullets/indicators */
.dot {
  height: 15px;
  width: 15px;
  margin: 0 2px;
  background-color: #bbb;
  border-radius: 50%;
  display: inline-block;
  transition: background-color 0.6s ease;
}

.active {
  background-color: #717171;
}
.heading{
color: black;
font-weight:800;
  font-size: 45px;
  padding: 8px 12px;
  bottom: 8px;
  width: 100%;
  text-align: center;

}

/* Fading animation */
.fade {
  animation-name: fade;
  animation-duration: 2s;
}

@keyframes fade {
  from {opacity: .4} 
  to {opacity: 1}
}

/* On smaller screens, decrease text size */
@media only screen and (max-width: 300px) {
  .text {font-size: 11px}
}
</style>
</head>
<body>

<div class="heading">WildBean's Smart Choice</div>

<div class="slideshow-container">

<div class="mySlides fade">
  <div class="numbertext">1 / 4</div>
  <img src="https://img.freepik.com/free-photo/still-life-tech-device_23-2150722602.jpg?w=360&t=st=1709880774~exp=1709881374~hmac=3e56e8b4e41d5d57ad1296e29ffdcbc888d7550d03ef199a0e1f4aa941510163" style="width:100%">
  <div class="text">Alexa Echo Dot</div>
</div>

<div class="mySlides fade">
  <div class="numbertext">2 / 4</div>
  <img src="https://img.freepik.com/free-photo/elegant-smartphone-composition_23-2149437106.jpg?t=st=1709880852~exp=1709884452~hmac=dc433974773e5a850c1ce9afad389b50cc6135df692d308626a9999d62948a5c&w=360" style="width:100%">
  <div class="text">Iphone 12</div>
</div>

<div class="mySlides fade">
  <div class="numbertext">3 / 4</div>
  <img src="https://img.freepik.com/free-photo/assortment-different-colored-tumblers_23-2149029267.jpg?t=st=1709880885~exp=1709884485~hmac=75639d14c371764cbf4dc211199e2300588f86b143c71e8cacb509c44ab4d6a0&w=360" style="width:100%">
  <div class="text">Smart bottles</div>
</div>
<div class="mySlides fade">
  <div class="numbertext">4 / 4</div>
  <img src="https://img.freepik.com/free-photo/lamp_1203-7283.jpg?t=st=1709888443~exp=1709892043~hmac=747bc40670a7f3f2ba51eebe16c178ed79ccb8c3c06cf0c7207591bf80f01437&w=360" style="width:100%">
  <div class="text">Study light</div>
</div>

</div>
<br>

<div style="text-align:center">
  <span class="dot"></span> 
  <span class="dot"></span> 
  <span class="dot"></span> 
  <span class="dot"></span> 
</div>

<script>
let slideIndex = 0;
showSlides();

function showSlides() {
  let i;
  let slides = document.getElementsByClassName("mySlides");
  let dots = document.getElementsByClassName("dot");
  for (i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";  
  }
  slideIndex++;
  if (slideIndex > slides.length) {slideIndex = 1}    
  for (i = 0; i < dots.length; i++) {
    dots[i].className = dots[i].className.replace(" active", "");
  }
  slides[slideIndex-1].style.display = "block";  
  dots[slideIndex-1].className += " active";
  setTimeout(showSlides, 3500); // Change image every 2 seconds
}
</script>

</body>
</html> 
"""
   ,
        height=450,
    )
def footer():
    components.html(
        """ <!DOCTYPE html>
   <html>
   <head>
   <meta name="viewport" content="width=device-width, initial-scale=1">
   <style>
   @import url("https://fonts.googleapis.com/css2?family=Noto+Serif+Display:wght@300&family=Rubik:wght@400;500;700&display=swap");

:root {
  --light: #ffffff;
  --dark: #000;
}
body {
  font-family: "Rubik", sans-serif;
  margin: 0;
  padding: 0;
  box-sizing: border-box;
   color:white;
}
h3 {
  font-family: "Noto Serif Display", serif;
  font-size: 2.3rem;
  font-weight: 300;
  text-align: center;
   color:white;
}
a {
  text-decoration: none;
  color: var(--dark);
  font-weight: 400;
  transition: 0.3s ease-in;
  border-bottom: 1px solid transparent;
  margin-bottom: 0.5rem;
  display: inline-flex;
   color:white;
}
a:hover {
  border-bottom: 1px solid var(--dark);
}
ul {
  list-style-type: none;
  padding: 0;
   color:white;
}
button {
  appearance: none;
  border: 0;
  background: transparent;
}
.flex {
  display: flex;
}
.footer_video {
  position: absolute;
  top: 0;
  left: 0;
  object-fit: cover;
  width: 100%;
  height: 100%;
  z-index: -1;
  overflow: hidden;
  border: none;
}
.footer_inner {
  background: var(--light);
  backdrop-filter: blur(50px);
  border: 0.1px solid rgba(233, 232, 232, 0.208);
  border-radius: 5px;
  padding: 2rem;
  margin: 1rem 0;
  background-color:grey;
}
.footer {
  position: relative;
  display: flex;
  align-items: center;
  min-height: 100vh;
}
.container {
  max-width: 1200px;
  width: 100%;
  margin: 0 auto;
  padding: 0 1rem;
  background-color:grey;
  border-radius:10px;
  color:white;
}

form {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: min(100%, 400px);
  border-bottom: 1px solid var(--dark);
   color:white;
}

input {
  padding: 0.75rem 0;
  border: none;
  background: none;
  font-weight: 500;
  transition: border 0.3s cubic-bezier(0.215, 0.61, 0.355, 1);
  border-radius: 0;
  width: 100%;
  font-size: 1.05rem;
  font-weight: bolder;
   color:white;
}
input:focus {
  outline: none;
}
input::placeholder {
  color: var(--dark);
}
@media (min-width: 675px) {
  .layout {
    display: flex;
    flex-wrap: nowrap;
    column-gap: 2rem;
  }
  .w-50 {
    width: 50%;
  }
  .w-25 {
    width: 25%;
  }
}
form {
  position: relative;
   color:white;
}
svg {
  margin: 0.5rem;
}
.c-2 {
  margin-top: 3.5rem;
}
.footer_copyright {
  color: var(--light);
}

   </style>
   </head>
   <body>

  <footer class="footer">
  <video class="footer_video" muted="" loop="" autoplay src="//cdn.shopify.com/s/files/1/0526/6905/5172/t/5/assets/footer.mp4?v=29581141968431347981633714450" type="video/mp4">
  </video>

  <div class="container">
    <div class="footer_inner">
      <div class="c-footer">
        <div class="layout">
          <div class="layout_item w-50">
            <div class="newsletter">
              <h3 class="newsletter_title">Get updates on fun stuff you probably want to know about in your inbox.</h3>
              <form action="">
                <input type="text" placeholder="Email Address">
                <button>
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">
                    <path fill="none" d="M0 0h24v24H0z" />
                    <path d="M16.172 11l-5.364-5.364 1.414-1.414L20 12l-7.778 7.778-1.414-1.414L16.172 13H4v-2z" />
                  </svg>
                </button>
              </form>
            </div>
          </div>

          <div class="layout_item w-25">
            <nav class="c-nav-tool">
              <h4 class="c-nav-tool_title">Menu</h4>
              <ul class="c-nav-tool_list">
                <li>
                  <a href="/collections/all" class="c-link">Home</a>
                </li>

                <li>
                  <a href="/pages/about-us" class="c-link">Write Review </a>
                </li>

                <li>
                  <a href="/blogs/community" class="c-link">Products</a>
                </li>
              </ul>
            </nav>

          </div>

          <div class="layout_item w-25">
            <nav class="c-nav-tool">
              <h4 class="c-nav-tool_title">Support</h4>
              <ul class="c-nav-tool_list">

                <li class="c-nav-tool_item">
                  <a href="/pages/shipping-returns" class="c-link">Shipping &amp; Returns</a>
                </li>

                <li class="c-nav-tool_item">
                  <a href="/pages/help" class="c-link">Help &amp; FAQ</a>
                </li>

                <li class="c-nav-tool_item">
                  <a href="/pages/terms-conditions" class="c-link">Terms &amp; Conditions</a>
                </li>

                <li class="c-nav-tool_item">
                  <a href="/pages/privacy-policy" class="c-link">Privacy Policy</a>
                </li>

                <li class="c-nav-tool_item">
                  <a href="/pages/contact" class="c-link">Contact</a>
                </li>

                <li class="c-nav-tool_item">
                  <a href="/account/login" class="c-link">
                    Login
                  </a>
                </li>
              </ul>
            </nav>

          </div>
        </div>
        <div class="layout c-2">
          <div class="layout_item w-50">
            <ul class="flex">
              <li>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="32" height="32">
                  <path fill="none" d="M0 0h24v24H0z" />
                  <path d="M12 6.654a6.786 6.786 0 0 1 2.596 5.344A6.786 6.786 0 0 1 12 17.34a6.786 6.786 0 0 1-2.596-5.343A6.786 6.786 0 0 1 12 6.654zm-.87-.582A7.783 7.783 0 0 0 8.4 12a7.783 7.783 0 0 0 2.728 5.926 6.798 6.798 0 1 1 .003-11.854zm1.742 11.854A7.783 7.783 0 0 0 15.6 12a7.783 7.783 0 0 0-2.73-5.928 6.798 6.798 0 1 1 .003 11.854z" />
                </svg>
              </li>
              <li>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="32" height="32">
                  <path fill="none" d="M0 0h24v24H0z" />
                  <path d="M1 4h22v2H1V4zm0 14h22v2H1v-2zm18.622-3.086l-.174-.87h-1.949l-.31.863-1.562.003c1.005-2.406 1.75-4.19 2.236-5.348.127-.303.353-.457.685-.455.254.002.669.002 1.245 0L21 14.912l-1.378.003zm-1.684-2.062h1.256l-.47-2.18-.786 2.18zM7.872 9.106l1.57.002-2.427 5.806-1.59-.001c-.537-2.07-.932-3.606-1.184-4.605-.077-.307-.23-.521-.526-.622-.263-.09-.701-.23-1.315-.419v-.16h2.509c.434 0 .687.21.769.64l.62 3.289 1.574-3.93zm3.727.002l-1.24 5.805-1.495-.002 1.24-5.805 1.495.002zM14.631 9c.446 0 1.01.138 1.334.267l-.262 1.204c-.293-.118-.775-.277-1.18-.27-.59.009-.954.256-.954.493 0 .384.632.578 1.284.999.743.48.84.91.831 1.378-.01.971-.831 1.929-2.564 1.929-.791-.012-1.076-.078-1.72-.306l.272-1.256c.656.274.935.361 1.495.361.515 0 .956-.207.96-.568.002-.257-.155-.384-.732-.702-.577-.317-1.385-.756-1.375-1.64C12.033 9.759 13.107 9 14.63 9z" />
                </svg>
              </li>
              <li>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="32" height="32">
                  <path fill="none" d="M0 0h24v24H0z" />
                  <path d="M15 17a7.5 7.5 0 1 1 0-15 7.5 7.5 0 0 1 0 15zM2 2h4v20H2V2z" />
                </svg>
              </li>
              <li>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="32" height="32">
                  <path fill="none" d="M0 0h24v24H0z" />
                  <path d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm-1-6v2h2v-2h1a2.5 2.5 0 0 0 2-4 2.5 2.5 0 0 0-2-4h-1V6h-2v2H8v8h3zm-1-3h4a.5.5 0 1 1 0 1h-4v-1zm0-3h4a.5.5 0 1 1 0 1h-4v-1z" />
                </svg>
              </li>
              <li>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="32" height="32">
                  <path fill="none" d="M0 0h24v24H0z" />
                  <path d="M20.067 8.478c.492.88.556 2.014.3 3.327-.74 3.806-3.276 5.12-6.514 5.12h-.5a.805.805 0 0 0-.794.68l-.04.22-.63 3.993-.032.17a.804.804 0 0 1-.794.679H7.72a.483.483 0 0 1-.477-.558L7.418 21h1.518l.95-6.02h1.385c4.678 0 7.75-2.203 8.796-6.502zm-2.96-5.09c.762.868.983 1.81.752 3.285-.019.123-.04.24-.062.36-.735 3.773-3.089 5.446-6.956 5.446H8.957c-.63 0-1.174.414-1.354 1.002l-.014-.002-.93 5.894H3.121a.051.051 0 0 1-.05-.06l2.598-16.51A.95.95 0 0 1 6.607 2h5.976c2.183 0 3.716.469 4.523 1.388z" />
                </svg>
              </li>
            </ul>
          </div>
          <div class="layout_item w-25">
            <ul class="flex">
              <li>
                <a href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="32" height="32">
                    <path fill="none" d="M0 0h24v24H0z" />
                    <path d="M12 2C6.477 2 2 6.477 2 12c0 4.991 3.657 9.128 8.438 9.879V14.89h-2.54V12h2.54V9.797c0-2.506 1.492-3.89 3.777-3.89 1.094 0 2.238.195 2.238.195v2.46h-1.26c-1.243 0-1.63.771-1.63 1.562V12h2.773l-.443 2.89h-2.33v6.989C18.343 21.129 22 16.99 22 12c0-5.523-4.477-10-10-10z" />
                  </svg>
                </a>
              </li>
              <li>
                <a href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="32" height="32">
                    <path fill="none" d="M0 0h24v24H0z" />
                    <path d="M22.162 5.656a8.384 8.384 0 0 1-2.402.658A4.196 4.196 0 0 0 21.6 4c-.82.488-1.719.83-2.656 1.015a4.182 4.182 0 0 0-7.126 3.814 11.874 11.874 0 0 1-8.62-4.37 4.168 4.168 0 0 0-.566 2.103c0 1.45.738 2.731 1.86 3.481a4.168 4.168 0 0 1-1.894-.523v.052a4.185 4.185 0 0 0 3.355 4.101 4.21 4.21 0 0 1-1.89.072A4.185 4.185 0 0 0 7.97 16.65a8.394 8.394 0 0 1-6.191 1.732 11.83 11.83 0 0 0 6.41 1.88c7.693 0 11.9-6.373 11.9-11.9 0-.18-.005-.362-.013-.54a8.496 8.496 0 0 0 2.087-2.165z" />
                  </svg>
                </a>
              </li>
              <li>
                <a href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="32" height="32">
                    <path fill="none" d="M0 0h24v24H0z" />
                    <path d="M12 2c2.717 0 3.056.01 4.122.06 1.065.05 1.79.217 2.428.465.66.254 1.216.598 1.772 1.153a4.908 4.908 0 0 1 1.153 1.772c.247.637.415 1.363.465 2.428.047 1.066.06 1.405.06 4.122 0 2.717-.01 3.056-.06 4.122-.05 1.065-.218 1.79-.465 2.428a4.883 4.883 0 0 1-1.153 1.772 4.915 4.915 0 0 1-1.772 1.153c-.637.247-1.363.415-2.428.465-1.066.047-1.405.06-4.122.06-2.717 0-3.056-.01-4.122-.06-1.065-.05-1.79-.218-2.428-.465a4.89 4.89 0 0 1-1.772-1.153 4.904 4.904 0 0 1-1.153-1.772c-.248-.637-.415-1.363-.465-2.428C2.013 15.056 2 14.717 2 12c0-2.717.01-3.056.06-4.122.05-1.066.217-1.79.465-2.428a4.88 4.88 0 0 1 1.153-1.772A4.897 4.897 0 0 1 5.45 2.525c.638-.248 1.362-.415 2.428-.465C8.944 2.013 9.283 2 12 2zm0 5a5 5 0 1 0 0 10 5 5 0 0 0 0-10zm6.5-.25a1.25 1.25 0 0 0-2.5 0 1.25 1.25 0 0 0 2.5 0zM12 9a3 3 0 1 1 0 6 3 3 0 0 1 0-6z" />
                  </svg>
                </a>
              </li>
            </ul>
          </div>
          <div class="layout_item w-25" style="display:flex;justify-content: end;align-items: center;">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="36" height="36">
              <path fill="none" d="M0 0h24v24H0z" />
              <path d="M12 2c5.52 0 10 4.48 10 10s-4.48 10-10 10S2 17.52 2 12 6.48 2 12 2zm1 10h3l-4-4-4 4h3v4h2v-4z" />
            </svg>
          </div>
        </div>
      </div>
    </div>
    <div class="footer_copyright">
      <p>&copy; 2022 The Wildbeans Company Inc.</p>
    </div>
  </div>
</footer>
  

   </body>
   </html> 
   """
        ,
        height=600,
    )


selected = option_menu(
    menu_title = None,
    options = ["Home", "Products","Write Review"],
    icons = ["house",'bag-dash','star-half'],
    menu_icon = 'cast',
    orientation = 'horizontal'
)
main(selected)




            
            
            
            








            

