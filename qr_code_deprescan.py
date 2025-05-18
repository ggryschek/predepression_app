## Create QR code for DepreScan ##

import qrcode
from PIL import Image

# Data to encode in the QR code
data = "https://deprescan.streamlit.app"

# Create a QR Code instance
qr = qrcode.QRCode(
    version=5,  # Adjust size (1 to 40)
    error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction for embedding images
    box_size=10,
    border=4,
)

# Add data to the QR code
qr.add_data(data)
qr.make(fit=True)

# Create QR code image
qr_img = qr.make_image(fill="black", back_color="white").convert("RGB")

# Load the logo image
logo = Image.open("C:/Thesis/predepression_app/logo_app.jpg")  # Replace with your image path

# Convert to RGB if it has an alpha channel (transparency)
if logo.mode in ("RGBA", "LA"):
    logo = logo.convert("RGB")

# Resize the logo
logo_size = min(qr_img.size) // 5  # Adjust logo size relative to QR code
logo = logo.resize((logo_size, logo_size), Image.LANCZOS)

# Calculate positioning
pos = ((qr_img.size[0] - logo.size[0]) // 2, (qr_img.size[1] - logo.size[1]) // 2)

# Paste logo onto QR code **without a mask**
qr_img.paste(logo, pos)

# Save and show the final QR code with the logo
qr_img.show()
qr_img.save("C:/Thesis/predepression_app/qrcode_with_logo.png")