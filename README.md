# Estimate_Speed
Estimate_Speed
# Object Detection
To perform object detection using a `pre-trained model`, we need to load the model and run inference on an input image. We can achieve this by defining the path to the model file using the `MODEL_PATH` variable. If we're using a `pre-trained model`, we can simply specify the version we want, such as yolov8n.pt.

Once we have the model loaded, we can process an image by converting it into a numpy array and passing it as input to the model's inference function. This will produce a set of predictions that identify the objects present in the image, along with their respective bounding boxes and confidence scores.

Overall, the process of performing object detection using a pre-trained model involves loading the model, processing an input image, and obtaining the resulting predictions. By following these steps, we can quickly and accurately detect objects in a wide variety of visual contexts.

Install Yolov8
```ruby
pip install ultralytics
```
# Object Tracking
Install ByteTRACK
```ruby
pip install bytetracker
```
