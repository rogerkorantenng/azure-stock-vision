import os
import json
import torch
import sqlite3
import gradio as gr
import faiss
import pandas as pd
from datetime import datetime
from PIL import Image
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from threading import Thread
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoTokenizer, AutoModel, \
    AutoModelForCausalLM
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn, optim
from torch.optim import lr_scheduler
from sklearn.preprocessing import LabelEncoder


# Initialize Flask app and models
app = Flask(__name__, static_folder="static")

# Constants
LOW_STOCK_THRESHOLD = 5  # Customize threshold as needed
DATABASE = 'uploaded_images.db'

# Available model options
MODEL_OPTIONS = {
    "Google ViT (Base)": "google/vit-base-patch16-224",
    "Google ViT (Large)": "google/vit-large-patch16-224",
    "Microsoft ResNet50": "microsoft/resnet-50",
    "Facebook ConvNeXt Tiny": "facebook/convnext-tiny-224",
    "Microsoft Swin": "microsoft/swin-tiny-patch4-window7-224",
}

# Set default model
selected_model_name = MODEL_OPTIONS["Google ViT (Base)"]
feature_extractor = AutoFeatureExtractor.from_pretrained(selected_model_name)
model = AutoModelForImageClassification.from_pretrained(selected_model_name)
class_names = model.config.id2label

# Initialize inventory data
inventory_data = {}


# Initialize database
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            upload_time TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


def plot_bar_chart(inventory_data):
    # Get the list of items and their counts
    items = list(inventory_data.keys())
    counts = [inventory_data[item]["count"] for item in items]

    # Extract only the first word from each item name
    first_words = [item.split()[0] for item in items]

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(first_words, counts, color="skyblue")

    # Set titles and labels
    ax.set_title("Inventory Counts")
    ax.set_xlabel("Items")
    ax.set_ylabel("Count")

    # Save the chart as a PNG file
    chart_path = "bar_chart.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return chart_path



def plot_line_chart(inventory_data):
    # Extract items, counts, and dates from inventory data
    items = list(inventory_data.keys())
    counts = [inventory_data[item]["count"] for item in items]
    dates = [inventory_data[item]["last_detected"] for item in items]

    # Convert string dates to pandas datetime objects
    dates = pd.to_datetime(dates)

    # Create a DataFrame with dates and counts
    counts_df = pd.DataFrame({"Date": dates, "Count": counts})
    counts_df.sort_values("Date", inplace=True)

    # Format the 'Date' column into a human-readable format
    counts_df["Date"] = counts_df["Date"].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(counts_df["Date"], counts_df["Count"], marker="o", color="orange")

    # Set the title and labels for the plot
    ax.set_title("Stock Changes Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Level")

    # Save the plot as a PNG file
    chart_path = "line_chart.png"
    plt.xticks(rotation=45)  # Rotate x-ticks for better readability
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return chart_path


def plot_pie_chart(inventory_data):
    # Get the list of items and their counts
    items = list(inventory_data.keys())
    counts = [inventory_data[item]["count"] for item in items]

    # Extract only the first word from each item name
    first_words = [item.split()[0] for item in items]

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(counts, labels=first_words, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)

    # Set the title
    ax.set_title("Product Category Breakdown")

    # Save the chart as a PNG file
    chart_path = "pie_chart.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return chart_path



def plot_heatmap(inventory_data):
    # Get the list of items
    items = list(inventory_data.keys())

    # Extract only the first word from each item name
    first_words = [item.split()[0] for item in items]

    # Create a matrix where the count is placed at the correct location
    change_matrix = [[inventory_data[item]["count"] if item == other else 0 for other in items] for item in items]

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(change_matrix, cmap="YlOrRd", interpolation='nearest')
    # Set the labels on the x and y axes to the first words of the items
    ax.set_xticks(range(len(items)))
    ax.set_yticks(range(len(items)))
    ax.set_xticklabels(first_words)
    ax.set_yticklabels(first_words)
    # Add a colorbar
    fig.colorbar(cax)
    # Set the title of the heatmap
    ax.set_title("Stock Change Heatmap")
    # Save the chart as a PNG file
    chart_path = "heatmap.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return chart_path



# Function to return the paths of the images
def generate_bar_chart():
    return plot_bar_chart(inventory_data)


def generate_line_chart():
    return plot_line_chart(inventory_data)


def generate_pie_chart():
    return plot_pie_chart(inventory_data)


def generate_heatmap():
    return plot_heatmap(inventory_data)


# Utility functions
def log_inventory_data(item_class):
    try:
        timestamp = datetime.now().isoformat()

        # Initialize the inventory data for the item class if it's not present
        if item_class not in inventory_data:
            inventory_data[item_class] = {
                "category": item_class,
                "count": 0,  # Initial count
                "last_detected": None,  # Timestamp of last detection
                "history": []  # Track the historical counts over time
            }

        # Log the count and timestamp to the history
        inventory_data[item_class]["history"].append({
            "timestamp": timestamp,  # Store the timestamp as a string
            "count": inventory_data[item_class]["count"]
        })

        # Update the count of the item class
        inventory_data[item_class]["count"] += 1

        # Update the last detected timestamp
        inventory_data[item_class]["last_detected"] = timestamp

        # Optionally: Save the data to a file for persistence
        with open("inventory_log.json", "w") as f:
            json.dump(inventory_data, f, indent=4)

        print(f"Inventory data logged for item class: {item_class}")

    except Exception as e:
        print(f"Error logging inventory data: {e}")


def check_stock_levels():
    try:
        for item, details in inventory_data.items():
            if details["count"] < LOW_STOCK_THRESHOLD:
                print(f"Low stock detected for {item}: {details['count']} items.")
    except Exception as e:
        print(f"Error checking stock levels: {e}")


def save_image_to_db(file):
    filename = secure_filename(file.filename)
    upload_time = datetime.now().isoformat()
    file_path = os.path.join("uploads", filename)
    file.save(file_path)
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO images (filename, upload_time) VALUES (?, ?)", (filename, upload_time))
    conn.commit()
    conn.close()


def batch_predict(images):
    results = []
    try:
        for image_file in images:
            with Image.open(image_file.name) as image:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                inputs = feature_extractor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
                item_class = class_names[predicted_class_id]
                log_inventory_data(item_class)  # Pass only the item_class argument
                results.append({"Image": os.path.basename(image_file.name), "Classification": item_class})
    except Exception as e:
        return [{"Image": "Error", "Classification": str(e)}]
    return pd.DataFrame(results)


def forecast_inventory(item_class, days=7):
    """Use linear regression for basic inventory forecasting."""
    try:
        # Get the historical data for the item class
        if item_class not in inventory_data or not inventory_data[item_class]["history"]:
            return {"error": f"No historical data for {item_class}."}

        history = inventory_data[item_class]["history"]
        timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in history]
        counts = [entry["count"] for entry in history]

        # Convert timestamps to days since the first entry
        days_since_first_entry = [(timestamp - timestamps[0]).days for timestamp in timestamps]

        # Apply linear regression to forecast the next `days` values
        model = LinearRegression()
        model.fit(np.array(days_since_first_entry).reshape(-1, 1), counts)

        # Predict future inventory counts
        future_days = np.array([days_since_first_entry[-1] + i for i in range(1, days + 1)]).reshape(-1, 1)
        predictions = model.predict(future_days)

        forecast = [{"day": i + 1, "predicted_count": int(predictions[i])} for i in range(days)]
        return forecast
        print(f"Inventory data logged for item class: {forecast}")

    except Exception as e:
        return {"error": f"Error in forecasting: {str(e)}"}


def change_model(selected_model_key):
    try:
        global feature_extractor, model, class_names

        if selected_model_key not in MODEL_OPTIONS:
            return "Error: Invalid model selection"

        # Retrieve the model path (either pre-trained or fine-tuned)
        selected_model_path = MODEL_OPTIONS[selected_model_key]

        # Load the model from the specified path
        feature_extractor = AutoFeatureExtractor.from_pretrained(selected_model_path)
        model = AutoModelForImageClassification.from_pretrained(selected_model_path)
        class_names = model.config.id2label

        return f"Model changed to {selected_model_key}"
    except Exception as e:
        return f"Error changing model: {str(e)}"


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to preprocess and load data
def preprocess_data(train_images, train_labels):
    dataset = CustomDataset(train_images, train_labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader


# Fine-tune the model with a custom dataset
def fine_tune_model(train_images, train_labels, model_name="custom_model"):
    try:
        # Convert image paths to a suitable format
        train_images = [image.name for image in train_images]  # list of image paths

        # Check if labels need encoding
        if isinstance(train_labels, str):
            train_labels = train_labels.split(",")  # Convert the comma-separated string to a list

        # Encode labels if they are not integers
        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(train_labels)

        # Load the model and prepare for fine-tuning
        global model
        model = AutoModelForImageClassification.from_pretrained(selected_model_name)

        # Modify the classifier layer for the new dataset
        num_labels = len(set(train_labels))  # Number of unique labels
        model.classifier = nn.Linear(model.config.hidden_size, num_labels)

        # Prepare data loader
        dataloader = preprocess_data(train_images, train_labels)

        # Define loss function, optimizer, and learning rate scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Training loop
        num_epochs = 50  # Set the number of epochs
        model.train()  # Set model to training mode
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs  # Handle outputs

                # Calculate loss
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            scheduler.step()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")
            # return f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}"

        # Save the fine-tuned model
        fine_tuned_model_path = f"models/{model_name}"  # Save path
        model.save_pretrained(fine_tuned_model_path)
        print(f"Fine-tuned model saved to {fine_tuned_model_path}")

        # Dynamically add the fine-tuned model to the model selection
        MODEL_OPTIONS[model_name] = fine_tuned_model_path
        return f"Fine-tuned model {model_name} has been added successfully!"
    except Exception as e:
        return f"Error fine-tuning the model: {str(e)}"



proxy_prefix = os.environ.get("PROXY_PREFIX")

# Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown("## StockVision – AI-Driven Inventory Management")

    with gr.Tab("Model Selection/Fine-Tuning"):
        with gr.Tab("Select Model"):
            gr.Markdown("Choose a model for image classification.")
            model_dropdown = gr.Dropdown(choices=list(MODEL_OPTIONS.keys()), label="Select Model",
                                         value="Google ViT (Base)")
            model_dropdown.change(change_model, inputs=model_dropdown, outputs=gr.Textbox(label="status"))
        with gr.Tab("Model Fine-Tuning"):
            gr.Markdown("Upload your custom dataset for model fine-tuning.")
            train_image_input = gr.File(label="Upload Training Images", file_count="multiple", type="filepath")
            train_label_input = gr.Textbox(label="Enter Corresponding Labels (comma-separated)")
            fine_tune_button = gr.Button("Fine-Tune Model")
            fine_tune_output = gr.Text(label="Training Status")
            fine_tune_button.click(fine_tune_model, inputs=[train_image_input, train_label_input],
                                   outputs=fine_tune_output)

    with gr.Tab("Image Classification"):
        gr.Markdown("Upload images for classification.")
        image_input = gr.File(label="Upload Images", file_count="multiple", type="filepath")
        output = gr.DataFrame(label="Classification Results", interactive=True)
        image_input.upload(batch_predict, inputs=image_input, outputs=output)

    with gr.Tab("Inventory"):
        gr.Markdown("Check current inventory.")
        inventory_display = gr.DataFrame(value=pd.DataFrame(columns=["Item", "Count", "Last Detected"]))
        gr.Button("Fetch Inventory").click(lambda: pd.DataFrame(inventory_data).T, outputs=inventory_display)

    with gr.Tab("Forecasting"):
        gr.Markdown("Forecast inventory levels for the next 7 days.")
        item_class_input = gr.Textbox(label="Enter Item Class")
        forecast_button = gr.Button("Forecast")
        forecast_output = gr.Text(label="Predicted Inventory Forecast")
        forecast_button.click(forecast_inventory, inputs=item_class_input, outputs=forecast_output)

    with gr.Tab("Inventory Dashboard"):
        with gr.Tab("Bar Chart"):
            bar_chart_image = gr.Image(type="filepath", label="Bar Chart")
            gr.Button("Generate Bar Chart").click(generate_bar_chart, outputs=bar_chart_image)

        with gr.Tab("Line Chart"):
            line_chart_image = gr.Image(type="filepath", label="Line Chart")
            gr.Button("Generate Line Chart").click(generate_line_chart, outputs=line_chart_image)

        with gr.Tab("Pie Chart"):
            pie_chart_image = gr.Image(type="filepath", label="Pie Chart")
            gr.Button("Generate Pie Chart").click(generate_pie_chart, outputs=pie_chart_image)

        with gr.Tab("Heatmap"):
            heatmap_image = gr.Image(type="filepath", label="Heatmap")
            gr.Button("Generate Heatmap").click(generate_heatmap, outputs=heatmap_image)

if __name__ == '__main__':
    app_thread = Thread(target=lambda: app.run(debug=True, use_reloader=False))
    app_thread.start()
    interface.launch(server_name="0.0.0.0" ,root_path=proxy_prefix, share=True)