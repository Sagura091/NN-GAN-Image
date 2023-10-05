from selenium import webdriver
from selenium.webdriver.common.by import By  # Import the necessary class
import os
import time
import requests
# URL of the page containing clip art images
base_url = "https://www.clipartmax.com/"

# Create a directory to save images
os.makedirs("D:/clipart_images3", exist_ok=True)

# Initialize a WebDriver (you'll need to specify the path to your webdriver executable)
# For example, if using Chrome:
# driver = webdriver.Chrome("path_to_chromedriver")

# Replace "path_to_chromedriver" with the actual path to your chromedriver executable
driver = webdriver.Chrome("path_to_chromedriver")

# Function to download images from a page
def download_images_on_page(driver):
    image_elements = driver.find_elements(By.CSS_SELECTOR, "img.lazy2")

    # Download and save the images
    for img in image_elements:
        img_url = img.get_attribute("data-original")
        if img_url and img_url.startswith("https://"):
            try:
                img_data = requests.get(img_url).content
                img_name = img_url.split("/")[-1]
                img_path = os.path.join("D:/clipart_images3", img_name)
                with open(img_path, "wb") as img_file:
                    img_file.write(img_data)
                print(f"Downloaded: {img_name}")
            except Exception as e:
                print(f"Error downloading: {img_url}, {e}")

# Start with the base URL
current_url = base_url
while True:
    # Navigate to the current URL
    driver.get(current_url)
    
    # Download images from the current page
    download_images_on_page(driver)

      # Find the "Next" button by its text
    next_button = driver.find_element(By.XPATH, "//a[@class='item' and contains(text(), 'Next')]")
    next_page_href = next_button.get_attribute("href")
    if next_page_href:
        current_url = next_page_href
        time.sleep(2)  # Add a delay to ensure the page loads completely
    else:
        break  # No more pages left, exit the loop

print("All images downloaded successfully.")

# Close the browser
driver.quit()