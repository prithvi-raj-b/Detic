#! /home/asl1/miniconda3/envs/detic3.9/bin/python3.9
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from torch import uint8 as torch_uint8
import tkinter as tk
import rospkg

import os
import sys
from shape_detection.srv import StringSrv

import pathlib as pl
package_path = pl.Path(rospkg.RosPack().get_path('detic'))
os.chdir(package_path)

from detic.predictor import DefaultPredictor
import detic.predictor as detic_predictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detic.config import add_detic_config


sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config

from google import genai
from google.genai import types
from dotenv import load_dotenv

config_file = package_path / "configs/Detic_LbaseI_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
model_weights_file = package_path / "models/Detic_LbaseI_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
if not config_file.exists():
    raise FileNotFoundError(f"Config file not found: {config_file}")
if not model_weights_file.exists():
    raise FileNotFoundError(f"Model weights not found: {model_weights_file}")

# camera_input_topic = "/dip/camera/color/image_rect_color"
camera_input_topic = "color_image"
camera_output_topic = "/dip/filtered_depth_image"
camera_input_topic_depth = "depth_image"
# camera_input_topic_depth = "/dip/camera/aligned_depth_to_color/image_raw"


class ImageProcessor:
    selected_object = None
    def __init__(self):
        conf = 0.5
        self.bridge = CvBridge()
        self.cfg = get_cfg()
        add_centernet_config(self.cfg)
        add_detic_config(self.cfg)
        self.cfg.merge_from_file(str(config_file))  # Path to SwinBase config file
        self.cfg.MODEL.WEIGHTS = str(model_weights_file) # Path to locally downloaded weights
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf  # Set threshold for this model
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = conf
        self.cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        self.cfg.MODEL.DEVICE = "cuda"  # Use GPU for inference
        self.cfg.MODEL.VOCAB = "lvis"  # Use LVIS vocabulary
        self.cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        self.metadata = MetadataCatalog.get(detic_predictor.BUILDIN_METADATA_PATH[self.cfg.MODEL.VOCAB])
        classifier = detic_predictor.BUILDIN_CLASSIFIER[self.cfg.MODEL.VOCAB]
        num_classes = len(self.metadata.thing_classes)
        self.all_class_names = self.metadata.get("thing_classes", None)

        self.cfg.freeze()
        self.predictor = DefaultPredictor(self.cfg)
        rospy.Subscriber(camera_input_topic, Image, self.callback, queue_size=1)
        # self.pub = rospy.Publisher(camera_output_topic, Image, queue_size=10)
        self.identified_objects = []
        detic_predictor.reset_cls_test(self.predictor.model, classifier, num_classes)

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            outputs = self.predictor(cv_image)

            # Extract instances and draw bounding boxes
            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes if instances.has("pred_boxes") else None
            classes = instances.pred_classes if instances.has("pred_classes") else None
            # class_names = instances.pred_classes_names if instances.has("pred_classes_names") else None
            class_names = [self.all_class_names[i] for i in classes] if classes is not None else None

            # Draw bounding boxes and labels on the image
            flag = 0
            if boxes is not None and class_names is not None:
                for box, class_name in zip(boxes, class_names):
                    x1, y1, x2, y2 = box.int().tolist()
                    if self.selected_object is not None and class_name == self.selected_object and flag == 0:
                        # Draw a red rectangle for the selected object
                        flag = 1
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(cv_image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else:
                        # Draw a gray rectangle for other objects
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (100, 100, 100), 2)
                        cv2.putText(cv_image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)

            # Display the image with bounding boxes
            cv2.imshow("Detic Output", cv_image)
            cv2.waitKey(1)

            # Create a list of identified object names
            if class_names is not None:
                self.identified_objects = tuple(zip(class_names, instances.pred_masks)) 

            # rospy.loginfo("Processed image with Detic")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

class PickFunction:
    def __init__(self):
        self.object_name = None
        self.mask = None
        self.shape_name = None
        self.pub = rospy.Publisher(camera_output_topic, Image, queue_size=10)
        self.message = None
        self.bridge = CvBridge()
        
        self.sub = rospy.Subscriber(camera_input_topic_depth, Image, self.callback, queue_size=1)
        self.serviceClient = rospy.ServiceProxy("/pick_and_place", StringSrv)

    def pick_fn(self, object_name, mask, shape_name):
        self.object_name = object_name
        self.mask = mask
        self.shape_name = shape_name
        ImageProcessor.selected_object = object_name

        self.serviceClient.call(self.shape_name)
    
    def callback(self, msg):
        try:
            # Convert the depth image to a CV2 format
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Apply the segmentation mask
            if self.mask is not None:
                segmented_depth = cv2.bitwise_and(depth_image, depth_image, mask=self.mask.numpy().astype("uint8"))
            else:
                # rospy.logwarn("No mask available to apply on the depth image.")
                return

            # Convert the segmented depth image back to a ROS Image message
            segmented_msg = self.bridge.cv2_to_imgmsg(segmented_depth, encoding="16UC1")
            segmented_msg.header = msg.header

            # Publish the segmented depth image
            self.pub.publish(segmented_msg)
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")

class ChatBoxGUI:
    def __init__(self, client, processor, pick_obj, config):
        self.window = tk.Tk()
        self.window.title("Chat Box")
        self.text_area = tk.Text(self.window, state="disabled", wrap="word")
        self.text_area.pack(expand=True, fill="both")
        self.entry = tk.Entry(self.window)
        self.entry.pack(fill="x")
        self.entry.bind("<Return>", lambda event: self.send_message())  # Bind Enter key to send_message
        self.send_button = tk.Button(self.window, text="Send", command=self.send_message)
        self.send_button.pack()

        # Store message history
        self.message_history = []

        # Gemini LLM client and other dependencies
        self.client = client
        self.chat = client.chats.create(model="gemini-2.5-flash", config=config)
        self.processor = processor
        self.pick_obj: PickFunction = pick_obj
        self.config = config

    def clear_text_area(self):
        self.text_area.config(state="normal")
        self.text_area.delete(1.0, tk.END)
        self.text_area.config(state="disabled")
        self.text_area.see(tk.END)
        self.message_history = []  # Clear message history
        self.update_text_area()
        self.entry.delete(0, tk.END)  # Clear the entry field
        ImageProcessor.selected_object = None  # Reset selected object

    def send_message(self):
        user_message = self.entry.get()
        if user_message.lower() == "clear":
            self.clear_text_area()
        elif user_message:
            self.message_history.append(f"You: {user_message}")
            self.update_text_area()
            self.entry.delete(0, tk.END)

            # Process the message with Gemini LLM
            self.process_with_gemini(user_message)

    def process_with_gemini(self, user_message):
        try:
            table_objects_str = ", ".join([x[0] for x in self.processor.identified_objects]) if self.processor.identified_objects else "None"
            actual_prompt = "Available Objects: " + table_objects_str + ".\nMy Response: " + user_message
            self.message_history.append(f"Actual Prompt:\n{actual_prompt}")
            self.update_text_area()

            # response = self.client.models.generate_content(
            #     model="gemini-2.5-flash-preview-04-17",
            #     contents=actual_prompt,
            #     config=self.config,
            # )

            response = self.chat.send_message(actual_prompt)
            # import pdb; pdb.set_trace()
            gemini_response = response.text
            self.message_history.append(f"Gemini: {gemini_response}")
            self.update_text_area()

            # Handle function calls from Gemini
            if response.function_calls:
                for function_call in response.function_calls:
                    if function_call.name == "pick":
                        object_name = function_call.args.get("object")
                        shape_name = function_call.args.get("shape")
                        object_mask = None
                        for obj_name, mask in self.processor.identified_objects:
                            if obj_name == object_name:
                                object_mask = mask
                                break
                        if object_mask is not None:
                            self.message_history.append(f"Picked object: {object_name}, Shape: {shape_name}")
                            self.update_text_area()
                            self.pick_obj.pick_fn(object_name, object_mask, shape_name)
                        else:
                            self.message_history.append(f"Object {object_name} not found in identified objects.")
                            self.update_text_area()
        except Exception as e:
            # error_message = f"Error during conversation: {e}"
            # self.message_history.append(error_message)
            self.update_text_area()

    def update_text_area(self):
        self.text_area.config(state="normal")
        self.text_area.delete(1.0, tk.END)
        for msg in self.message_history:
            self.text_area.insert(tk.END, f"{msg}\n")
            self.text_area.insert(tk.END, "---------------------------------\n")
        # self.text_area.insert(tk.END, "---------------------------------\n")
        self.text_area.config(state="disabled")
        self.text_area.see(tk.END)  # Scroll to the end of the text area
        self.window.update_idletasks()  # Update the GUI to reflect changes


if __name__ == "__main__":
    rospy.init_node("detic_image_processor")
    processor:ImageProcessor = ImageProcessor()
    pick_obj:PickFunction = PickFunction()

    # Load environment variables
    load_dotenv("gemini/.env")

    # Initialize the GenAI LLM model
    pick_fn_declaration = {
        "name": "pick",
        "description": "The prompt contains a list of objects on the table. Pick one of them according to the instruction from the user. " \
            "Use your knowledge to identify whether the object is a cylinder, cuboid, sphere or thin plate.",
        "parameters": {
            "type": "object",
            "properties": {
                "object": {
                    "type": "string",
                    "description": "The object to pick from the table. The object has to be chosen from the list of objects on the table."
                },
                "shape": {
                    "type": "string",
                    "description": "The shape of the object. The shape can be a cylinder, cuboid, sphere or thin plate."
                }
            },
            "required": ["object", "shape"],
        },
    }
    reset_fn_declaration = {
        "name": "reset",
        "description": "Reset the robot arm to the initial/home position.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
    # Define client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    tools = types.Tool(function_declarations=[pick_fn_declaration, reset_fn_declaration])
    config = types.GenerateContentConfig(tools=[tools])

    # Conversation loop
    chat_box = ChatBoxGUI(client, processor, pick_obj, config)
    chat_box.window.mainloop()
    
    # rospy.spin()
