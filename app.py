import streamlit as st 
import pandas as pd
import numpy as np
from src.pipeline.prediction_pipline import CustomData , PredictPipeline

# Set up page configuration
st.set_page_config(page_title="Smartphones sales price prediction", layout="centered")

# Title and Description
st.title("Smartphones sales price prediction")
st.write("Enter the  details below to predict smartphones selling price.")

# Create input form
with st.form("prediction_form"):
    st.subheader("Smartphone Context Details")

    # Categorical inputs using dropdowns
    brand = st.selectbox("Brand", options=['OPPO' 'HTC' 'IQOO' 'Google Pixel' 'LG' 'ASUS' 'realme' 'GIONEE' 'Nokia' 
                                           'Apple' 'SAMSUNG' 'Lenovo' 'Motorola' 'POCO' 'vivo' 'Xiaomi' 'Infinix'])
    model = st.selectbox(
        "model",
        options=['A53' 'A12' 'A53s 5G' 'A33' 'A31' 'A74 5G' 'A11K' 'F17 Pro' 'A54''Reno6 5G' 'F17' 'A16' 'Reno5 Pro 5G' 'A15' 'A74 5G BLACK' 'Reno6 Pro 5G''Reno2 F' 'Reno3 Pro' 'A15s' 'A15S' 'F19 Pro' 'F19 Pro+ 5G' 'F19' 'A7''F5' 'A5s' 'Reno' 'Reno2 Z' 'F15' 'K1' 'A9 2020' 'Find X'
            'Reno4 Pro Special Edition' 'F11' 'A5' 'Reno4 Pro' 'A5 2020''F3 Deepika Padukone Limited Edition' 'F1 Plus' 'A83 2018 Edition' 'F3''A3s' 'A83' 'K3' 'F9' 'A52' 'A71 New Edition' 'Neo 7 4G' 'A71' 'A57''A1K' 'A37f' 'F9 Pro' 'F19s' 'A71k' 'A9' 'F3 Plus''Rohit Sharma Limited Edition' 'F17 PRO' 'Hardik Pandya Limited Editon''Ravichandran Ashwin Limited Edition' 'F11 Pro' 'R1 R829' 'F5 Youth' 'F7'
            'Neo 5' 'F1' 'Reno2' 'R17' 'F11 Pro Marvels Avengers Limited Edition''Reno 10x Zoom' 'F1S' 'N5111' 'U11+ ' 'Wildfire X ' '3' '4a ' '3a XL ''3 XL ' '3a ' 'XL ' '2 XL ' '2' 'W11 ' 'Velvet Dual Screen ' 'Q60 ''L90 Dual ' 'W31 ' 'W31 Plus ' 'K42 ' 'W30 Pro ' 'Candy K9 ' 'K9 4G LTE '
            'W10 ' 'K-10 ' 'W41 ' 'Q Stylus+ ' 'W10 Alpha ' 'W30 ' 'W30 Plus ''G7+ ThinQ ' 'Stylus 2 ' 'Max X160 ' 'V20 ' 'G8X ' 'V40 ThinQ ' 'K7i ''Nexus4 E960' 'G7 ThinQ ' 'Nexus 5X ' 'G5 ' 'Q6+ ' 'Spirit 4G LTE ''Velvet ' 'W41 Plus ' 'Wing ' 'W41 Pro ' 'Optimus L70 ' 'K8 ' 'G Pro 2 '
            'V30+ ' 'G2 D802 ' 'Q Stylus ' 'Q6 ' 'G6 ' 'K10 2017 ' 'V20a ''K10 K420DS ' 'Q7 ' 'X Power ' 'Stylus 2 Plus ' 'K-7 ' 'G3 Beat ''G8s ThinQ ' 'G4 ' 'Stylus 3 ' 'L Bello ' 'G4 Stylus 4G LTE ' 'X Cam ''L 80 Dual ' 'G3 Stylus ' 'Q7+ ' 'X Screen K500I ' 'Spirit ''ROG Phone 5 ' 'ROG Phone 3 ' 'Zenfone Max Pro M1 ''Zenfone 2 Laser ZE500KL ' 'ZenFone Max M2 ' 'ZenFone 5Z ' '6Z '
            'Zenfone Go 4.5 ' 'Zenfone Go 5.5 ' 'Zenfone Go 3rd Gen''Zenfone Go 2nd Gen' 'Zenfone Go ' 'ZenFone Lite L1 ' 'Zenfone 4 Selfie ''ROG Phone II ' 'Zenfone GO ' 'Zenfone Go 5.0 ' 'ZenFone Max Pro M2 ' 'Zenfone Go 5.0 LTE 2nd Gen ' 'Zenfone C ' 'Zenfone Go 5.0 LTE '
            'Zenfone Max ZC550KL ' 'Zenfone Max ' 'Zenfone Go 4.5 LTE ''ZenFone Max M1 ' 'ROG ' 'Zenfone Selfie ' 'ROG Phone 5 Pro ''ROG Phone 5 Ultimate ' 'Zenfone Live ' 'Zenfone Zoom ' 'C11 2021 '
            'Narzo 30 5G ' 'C20 ' 'C21Y ' '8i ' 'C25s ' '8s 5G ' 'C25Y ' 'C21 ''Narzo 30 ' 'Narzo 30A ' '8 5G ' '8 Pro ' 'C15 ' '8' 'C12 ' 'C11 ' '7''X7 5G ' 'Narzo 30 Pro 5G ' 'C3 ' 'C15 Qualcomm Edition ' '3i ' '5 Pro '
            '7i ' 'X3 SuperZoom ' 'X50 Pro 5G ' 'X3 ' 'GT 5G ' 'X50 Pro ' 'C2 ' '6''X7 Pro 5G ' 'GT Master Edition ' '6i ' '1' 'X2 ' '5i ' 'X ' 'Narzo 50i ''Narzo 50A ' 'C25 ' '7 Pro ' 'Narzo 20 Pro ' 'Narzo 20A ' 'Narzo 20 ''Narzo 10A ' 'Narzo 10 ' '6 Pro ' 'U1 ' 'X2 Pro ' '5s ' 'Max ' 'Max Pro '
            'F8 Neo ' 'F103 Pro ' 'F205 Pro ' 'A1 ' 'S11 Lite ' 'F11 ' 'F10 Plus ''L800' 'Pioneer P5W ' 'S96' 'L700' 'P5L ' 'P5 Mini ' 'P5_W ' 'F9 ' 'X-1 ''M5 Lite 4G ' 'P7 ' 'A1 Plus ' 'Pioneer P4 ' 'Pioneer P3S ' 'X1s ' 'F10 ''F205 ' 'A1 Lite ' 'Elife S7 ' 'Marathon M5 lite CDMA ' 'Marathon '
            'Marathon M3 ' 'M7 Power ' 'S10 Lite ' 'S6S ' 'M3 ' 'S Plus ' 'Elife E8 ''F9 Plus ' 'Pioneer P4S ' 'Marathon M5 Lite ' 'P7 Max ' 'M7 ''F103 3GB Version ' 'S6 ' 'S6 Pro ' 'F103 ' 'S11 ' 'V4S ''Marathon M5 Plus ' 'G3 ' 'F103 3Gb Version ' 'Elife E7 Mini ' 'Ctrl V4S ' 'M2 ' 'Elife S5.1 ' 'Elife E3 ' 'Pioneer P6 ' '105 DS 2020''TA-1010/105' '110 TA-1302 DS' '150 DS 2020' '150 TA-1235 DS' '6310''105 SS 2021' 'C01 Plus ' '125 DS' '5310ds' '3310 DS 2020' '3.4'
            '125 TA-1253 DS' 'C20 Plus ' 'G20 ' '5.4' 'TA-1174 / TA-1299' '2.4''G10 ' '125 DS 2020' '5310 TA-1212 DS' '216 DS 2020' '215 4G DS 2020''225 4g ds' '225 4G DS 2020' '2.3' '215 4G DS' '9' '150' '105 DS' '215''130' '108 Dual SIM' '107 Dual SIM' '150/150 DS' '3.2' '5.1 Plus '
            'Ta -1010/105' '105' '6.1' '2.1' '5' '112' '3.1' '5.1' '2.2' 'Lumia 920 ''3.1 Plus ' '110' '225' '106' '8 Sirocco ' 'RM-1172 / Nokia 230 DS' '5.3''6.1 Plus ' '8.1' '7.2' '7.1' 'X2 Dual SIM ' '7 Plus ' 'Asha 206' '222''6.2' '4.2' 'Asha 502 ' '515' 'iPhone SE ' 'iPhone XR ' 'iPhone 12 Mini ''iPhone 13 Pro ' 'iPhone 11 ' 'iPhone 12 ' 'iPhone 13 ' 'iPhone 13 Mini '
            'iPhone 12 Pro ' 'iPhone 11 Pro Max ' 'iPhone 8 Plus ' 'iPhone 11 Pro ''iPhone XS ' 'iPhone 8 ' 'iPhone 13 Pro Max ' 'iPhone 6 Plus ''iPhone XS Max ' 'iPhone 7 Plus ' 'iPhone 6s ' 'iPhone 7 ' 'iPhone 6 ''iPhone 6s Plus ' 'iPhone X ' 'iPhone 6s' 'Galaxy F22 ' 'Galaxy F12 '
            'M31s ' 'M02s ' 'Galaxy M02 ' 'Galaxy A12 ' 'Galaxy A22 ''Galaxy A22 5G ' 'Galaxy M12 ' 'M31 ' 'M21 2021 Edition ' 'Galaxy A21s ''GALAXY M31S ' 'M32 5G ' 'Galaxy M31 ' 'Galaxy Z Flip3 5G ' 'Galaxy M01 ''Galaxy A03s ' 'Galaxy F02s ' 'Galaxy A72 ' 'Galaxy M31s ' 'Galaxy M32 '
            'Galaxy A51 ' 'Galaxy F62 ' 'Galaxy A52s 5G ' 'Galaxy M21 2021 Edition ''Galaxy M11 ' 'Galaxy S10 Lite ' 'Galaxy F41 ' 'Galaxy A52 ''Galaxy A32 ' 'S20 FE 5G ' 'Galaxy S10 ' 'M01 core ' 'GALAXY M51 ''Galaxy Note 20 ' 'Galaxy Note10 Lite ' 'Galaxy A71 ' 'Galaxy A70s ''Galaxy Z Fold3 5G ' 'Galaxy M32 5G ' 'Galaxy A50s ' 'Galaxy S9 Plus '
            'Galaxy A6 ' 'Galaxy A6+ ' 'Galaxy On7 ' 'Galaxy A70 ' 'Galaxy A31 ''M21 2021 Edition' 'Galaxy Note 20 Ultra 5G ' 'Galaxy M01s ' 'Metro 350''Galaxy A20s ' 'Galaxy J7 Prime ' 'Galaxy S20 FE ' 'Galaxy A30s ''Galaxy A7 ' 'Galaxy J2 Core ' 'Galaxy J6 ' 'Galaxy A20 ' 'Galaxy A10s '
            'Galaxy M21 ' 'Galaxy J2 2018 ' 'Galaxy J7 Nxt ' 'Galaxy M30 ''Galaxy M42 ' 'Galaxy J7 Duo ' 'Galaxy A9 ' 'Galaxy J6 Plus ''Galaxy Fold 2 ' 'Galaxy J7 - 6 New 2016 Edition) ' 'Galaxy S20 Ultra ''Galaxy Note 9 ' 'Galaxy A10 ' 'Galaxy M40 ' 'Galaxy J2 ' 'Galaxy A50 '
            'Galaxy J2-2017 ' 'Galaxy Grand 2 ' 'Galaxy S10 Plus ' 'Galaxy J4 ''Galaxy A30 ' 'Galaxy M51 ' 'Galaxy J5 Prime ' 'Galaxy S6 Edge ' 'Galaxy S21 Ultra ' 'Galaxy J8 ' 'Galaxy A7-2017 ' 'Galaxy J4 Plus ''Galaxy A8 Plus ' 'Galaxy A80 ' 'Galaxy M30s ' 'Galaxy S5 ''Galaxy J7 Prime 2 ' 'Galaxy S21 Plus ' 'Galaxy S21 ' 'Galaxy M42 5G ''Galaxy On8 ' 'Galaxy S20+ ' 'Galaxy A2 Core ' 'Galaxy M10 ' 'M31 Prime '
            'Galaxy S9 ' 'Galaxy A7 2016 Edition ' 'Metro XL' 'Z2 ' 'Galaxy M10S ''Galaxy M20 ' 'Galaxy J5 ' 'Galaxy On6 ' 'Galaxy Note 5 ''Galaxy S4 Mini ' 'Galaxy A9 Pro ' 'Galaxy J2 - 2016 ' 'Galaxy J2 Pro ''Galaxy Note 10 Plus ' 'Galaxy S8 ' 'Z4 ' 'Galaxy J1 ' 'Galaxy S20 '
            'Galaxy Z Flip ' 'Galaxy Alpha ''Galaxy Core Prime G361 Dual Sim - White ' 'Galaxy E7 ' 'Galaxy Note 8 ''Galaxy A5-2017 ' 'Tizen Z3 ' 'Galaxy On Nxt ' 'Galaxy M30S ''Galaxy J2 Ace ' 'Galaxy S4 ' 'Galaxy Note 3 ' 'Galaxy A3 '
            'Galaxy Note 3 Neo ' 'Galaxy A8 ' 'Galaxy A5 2016 Edition ' 'Galaxy J7 ''Galaxy Core ' 'Galaxy Note 10 ' 'Galaxy Grand Neo ' 'Galaxy Note 4 ''On7 Pro ' 'Galaxy S10e ' 'Galaxy S7 Edge ' 'Galaxy A8 Star '
            'Galaxy S3 Neo ' 'Galaxy C7 Pro ' 'Galaxy Folder 2 ''Galaxy J5 - 6 New 2016 Edition) ' 'Metro 360' 'Z3 ' 'Grand Prime 4G ''Fold 2 5G ' 'Galaxy Core Prime ' 'On5 Pro ' 'Galaxy A5 '
            'Galaxy Grand Neo Plus ' 'Galaxy Note Edge ' 'Galaxy S6 ' 'Grand Prime ''Galaxy S8 Plus ' 'Rex 60' 'Galaxy S7 ' 'Galaxy E5 ' 'Sm-G361Hhadins ''Guru FM Plus' 'Galaxy Grand Quattro ' 'Galaxy Grand I9082 '
            'Galaxy Mega 5.8 ' 'Galaxy Note 5 Dual ' 'GURU FM PLUS' 'Galaxy S4 Zoom ''Galaxy Grand Prime 4g ' 'S7 Edge ' 'Galaxy S6 Edge+ ' 'A6600d40 ''K8 Plus ' 'K3 Note ' 'S930 ' 'A7 ' 'A5 ' 'A2010 ' 'Vibe Shot ' 'A7700 '
            'K8 Note ' 'K9 ' 'VIBE P1m ' 'Z2 Plus ' 'Vibe K5 Note ' 'B ' 'S850 ''A6600 ' 'A5000 ' 'A6600 Plus ' 'A536 ' 'A7000 ' 'K10 Note ''A7000 Turbo ' 'Vibe K5 Plus ' 'K6 Note ' 'K6 Power ' 'X2-AP '
            'K10 Plus ' 'P770 ' 'VIBE P1 ' 'K4 Note ' 'S660 ' 'Vibe Z2 Pro ''K9 Note ' 'P780 ' 'A850 ' 'P2 ' 'A328 ' 'A6 Note ' 'Vibe P1 Turbo ''A1000 ' 'A6000 Plus ' 'S560 ' 'Sisley S60 ' 'P70 ' 'S90 Or Sisley S90 '
            'A6000 Shot ' 'Vibe S1 ' 'K6 POWER ' 'K3 Note Music ' 'ZUK Z1 ''E7 Power ' 'Edge 20 Fusion ' 'G10 Power ' 'G60 ' 'G40 Fusion ''2nd Generation ' 'Edge 20 ' 'G8 Power Lite ' 'X4' 'Razr 5G ' 'E5'
            'Razr ' 'X Play' 'C Plus' 'G6' 'M' 'One Vision ' 'G30 ' 'G5' 'G9 Power ''G 2nd Generation ' 'One Fusion+ ' 'G9 ' 'E7 Plus ' 'One Action ''E 2nd Gen 3G ' 'E6s' 'C' 'E5 Plus' 'Edge+ ' 'G 3rd Generation ''One Macro ' 'G5s' 'X 2nd Generation ' 'G7' 'One' 'Z2 Play' 'One Power ''G8 Plus' 'G' 'G6 Play' 'G6 Plus ' 'G4 Plus' 'G7 Power' 'G4' 'Z Play'
            'G5 Plus' 'Z2 Force' 'X3 Pro ' 'M3 Pro 5G ' 'M2 Reloaded ' 'Y20G ' 'Y21 ''Y33s ' 'Y20A ' 'Y12s ' 'Y20G 2021 ' 'Y20A 2021 ' 'Y73 ' 'Y1s ' 'Y12G ''Y51A ' 'Y1S ' 'Y72 5G ' 'V21e ' 'Y20 ' 'V20 SE ' 'Y11 ' 'Y53s ' 'X60 ''V20 2021 ' 'V21 5G ' 'Y91i ' 'Y31 ' 'Y30 ' 'V20 Pro ' 'Mi 10 ' 'Mi 10i '
            'Mi 10T ' 'Mi 11 Lite ' 'Mi 11X ' 'MI 11X 5G ' 'Mi 11X Pro 5G ' 'Mi A2 ''Mi A3 ' 'Mi Max 2 ' 'MI3 ' 'Redmi 5 ' 'Redmi 6 ' 'Redmi 6 Pro ''Redmi 6A ' 'Redmi 7 ' 'Redmi 7A ' 'Redmi 8 ' 'Redmi 8A ''Redmi 8A Dual ' 'Redmi 9 ' 'REDMI 9 Power ' 'REDMI 9 Prime ' 'Redmi 9A '
            'REDMI 9i ' 'Redmi K20 ' 'Redmi K20 Pro ' 'REDMI Note 10 Pro ''REDMI Note 10S ' 'Redmi Note 4 ' 'Redmi Note 5 ' 'Redmi Note 5 Pro ''Redmi Note 6 Pro ' 'Redmi Note 7 ' 'Redmi Note 7 Pro ' 'Redmi Note 7S ''Redmi Note 8 ' 'REDMI Note 9 ' 'Redmi Note 9 Pro ' 'Redmi Y1 '
            'Redmi Y2 ' 'Redmi Y3 ' 'Quite Black' 'Very Silver' 'U11' 'U Ultra''Desire 630' '10' '628' 'Desire 626' 'Desire 828' 'Desire 826 DS''820 G+' 'Desire 326G DS' 'One E9+' '626G Plus' '826' '526G Plus' '820S''Desire 820' 'Desire 526G Plus' '816G' '620G' 'Desire 816G' 'Desire 616'
            'Desire 516' 'One M8 ' 'Desire 816  ' 'Desire 700' 'One 802D' 'XR 20''A55' 'F3 GT' 'M2 Pro' 'M2  ' 'X2' 'X2 Special Edition' '9i' 'GT Neo 2''X7 Max' 'Galaxy F42 5G ' 'Y21A' 'Y33T' 'Y21T' 'V23 5G' 'V23 Pro 5G''Y20 T' 'X70 Pro+' 'X70 Pro ' 'X60 Pro' 'X50 Pro' 'X50  ' 'U10' 'U12''S1' 'Z1x' 'S2' 'Z1 Pro' 'Y90' 'Y93' 'Y94' 'Y95' 'Y83' 'V9' 'X21''V9 Youth' 'V7+' 'Y69' 'Y83 Pro' 'V11 Pro' 'Y71i' 'V7  ' 'Y11T Pro'
            '11i 5G' '11i Hypercharge 5G' 'Redmi Note 11T 5G' 'Note 10 lite''9A Sport' '11 Lite NE' 'Hot 10 Play' 'Hot 11S' 'Hot 11' 'Hot 10S''Smart 5' 'Smart 6' 'Smart 7' 'Smart 8' 'Note 11 ' 'Smart 5A''Smart HD 2021' 'Note 11S' 'Note 11' 'Note 10 Pro''Note 11S Free Fire Edition' 'Smart 4' 'Note 5' 'Smart 2' 'Hot 6 Pro''Hot 8' 'Smart 3 Plus' 'Note 5 Stylus' 'S5 Pro' 'S4' 'Note 7' 'Zero 8i'
            'Hot 10' 'Smart 4 Plus' 'Note 10' 'Hot 9' 'Hot 9 Pro' 'Hot S3X' 'Note 4''Hot S3' 'S5 Lite' 'Hot 4 Pro' 'S5 ' 'Hot 7' 'Zero 5' 'Zero 5 Pro''Hot 7 Pro' 'Reno7 Pro 5G' 'C31' 'T1 5G' 'Reno7 5G' 'A16k' 'ROG 5s''ROG 5s Pro' 'iPhone 13 Pro Max' 'iPhone 13 Pro' 'iPhone 13 mini'
            'Y75 5G' '9 5G' '9 Pro 5G' '9 Pro+ 5G' '9 5G SE' 'C35' 'M4 Pro''M4 Pro 5G' 'Galaxy S21 SE 5G' 'Galaxy F23 5G' 'Galaxy A03 Core' 'M52 5G'],
            )
    color = st.selectbox("color ",
        options=['Black''White/Cream' 'Blue' 'Silver' 'Green' 'Purple' 'Gold' 'Aurora'
            'White' 'Orange' 'Red' 'Starry Night' 'Grey' 'Pink' 'Brown' 'Ice'
            'Black & Blue' 'Black & Gold' 'Illusion Sky' 'Yellow' 'Nebula'
            'Diamond Sapphire' 'Purple/Pink' 'Fjord' 'Glacier' 'Polar Night'
            'Gold/Beige' 'Dusk' 'Night' 'Copper/Bronze' 'Black & Copper'
            'Blue & Silver' 'Baltic' 'Silver/Grey' 'Cyan/Teal' 'White & Copper'
            'Silver/White' 'Black/Grey' 'Cloud Navy' 'Aura Glow' 'Sprite'
            'Frosted Pearl' 'Sapphire Gradient' 'Dark Pearl' 'Pastel Sky'
            'Smoky Sangria' 'Rich Cranberry' 'Metallic Sage' 'Midday Dream'
            'Diamond Glow' 'Diamond Flare' 'Prism Magic' 'Crystal Symphony'
            'Sunset Jazz' 'Sunset Melody' 'Sunset Dazzle' 'Fantastic Rainbow'
            'Pacific Sunrise' 'Sunrise Flare' 'Dark Nebula' 'Dark Night' 'Neon Spark'
            'Aurora Dawn' 'Celestial Magic' 'Pacific Pearl' 'Diamond Dazzle'
            'Ocean Wave' 'Celestial Snow' 'Nordic Secret' 'Heart Of Ocean'
            'Moonlight Jade' 'Rainbow Fantasy' 'Glowing Galaxy' 'Azure Glow'
            'Starry Glow'],
    )
    
    rating = st.slider("rating", 
                         options=[ 4.5 , 4.3, 4.4 ,4.2 ,0 , 4,  4.6, 3.8, 3,  4.1, 3.7, 3.1, 4.7, 3.9, 3.4, 3.3 ,3.6 ,3.53,2, 2.7, 2.8, 5, 2.3, 4.9 ,4.8])
    
    storage = st.selectbox("Storage ",
        options=[ 64, 128,  32, 256,  16,   8,   4, 512,  10, 100, 129, 130 ],
    )

    memory = st.number_input(
        "memory", options=[4, 6, 3, 8, 2, 12, 1,   1.5, 16, 18, 64, 32, 46,  0.5, 30]
    )
    
    Original_Price = st.number_input(
        "Original Price",min_value= 1149 ,max_value=179900
    )
    # Submit button for the form
    submit_button = st.form_submit_button(label="Predict Performance")

# Handle form submission and prediction
if submit_button:
    # Initialize the custom data class with form values
    data = CustomData(
        brand = brand,
        model = model,
        color = color,
        rating = rating,
        Storage = float(storage),
        memory = float(memory),
        Original_Price = float(Original_Price),
    )

    # Convert to DataFrame
    pred_df = data.get_data_as_data_frame()

    # Console logging mimicking your Flask prints
    print(pred_df)
    print("Before Prediction")

    # Run the prediction pipeline
    predict_pipeline = PredictPipeline() 
    print("Mid Prediction")
    results = np.expm1(predict_pipeline.predict(pred_df))
    print("After Prediction")

    # Display results to the user
    st.success(f"### Predicted Score: {results[0]:.2f}")
