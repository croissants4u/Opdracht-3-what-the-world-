import cv2
import torch
import openai
from diffusers import StableDiffusionPipeline
import os

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Set your OpenAI API key here
openai.api_key = '#'

# Custom class for object detection
class DetectionAgent:
    def detect_objects(self, frame):
        results = model(frame)
        detected_objects = results.pandas().xyxy[0]['name'].tolist()
        return detected_objects, results

# Custom class for text comprehension
class ComprehensionAgent:
    def comprehend_objects(self, detected_objects):
        if not detected_objects:
            return "No objects detected."
        
        detected_text = ', '.join(detected_objects)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": f"Detected objects: {detected_text}. Please describe the behavior of the visitors."
                }
            ]
        )
        generated_prompt = response['choices'][0]['message']['content'].strip()
        return generated_prompt

# Custom class for contextual analysis
class CommentatorAgent:
    def provide_context(self, comprehension_text):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable commentator providing context."
                },
                {
                    "role": "user",
                    "content": comprehension_text
                }
            ]
        )
        context_analysis = response['choices'][0]['message']['content'].strip()
        return context_analysis

# Custom class for knowledge aggregation
class KnowledgeAggregatorAgent:
    def aggregate_knowledge(self, vision_description, context_analysis):
        aggregated_text = f"Vision: {vision_description} | Context: {context_analysis}"
        return aggregated_text

# Custom class for image generation
class StableDiffusionGenerationAgent:
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
    
    def generate_image(self, prompt):
        print(f"Generating image with prompt: {prompt}")
        results = self.pipe(prompt=prompt, num_inference_steps=50, guidance_scale=7.5)
        imga = results.images[0]
        imga.save("generated_image.png")
        self.open_image_in_vscode("generated_image.png")

    def open_image_in_vscode(self, image_path):
        os.system(f'code -r {image_path}')

# Custom pipeline for real-time processing
class RealTimeProcessingPipeline:
    def __init__(self, detection_agent, comprehension_agent, commentator_agent, aggregator_agent, generation_agent):
        self.detection_agent = detection_agent
        self.comprehension_agent = comprehension_agent
        self.commentator_agent = commentator_agent
        self.aggregator_agent = aggregator_agent
        self.generation_agent = generation_agent

    def process_frame(self, frame):
        print("Processing frame for object detection...")
        detected_objects, results = self.detection_agent.detect_objects(frame)
        print(f"Detected objects: {detected_objects}")
        
        vision_description = self.comprehension_agent.comprehend_objects(detected_objects)
        print(f"Vision description: {vision_description}")
        
        context_analysis = self.commentator_agent.provide_context(vision_description)
        print(f"Context analysis: {context_analysis}")
        
        aggregated_knowledge = self.aggregator_agent.aggregate_knowledge(vision_description, context_analysis)
        print(f"Aggregated knowledge: {aggregated_knowledge}")
        
        self.generation_agent.generate_image(aggregated_knowledge)
        return results

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    detection_agent = DetectionAgent()
    comprehension_agent = ComprehensionAgent()
    commentator_agent = CommentatorAgent()
    aggregator_agent = KnowledgeAggregatorAgent()
    generation_agent = StableDiffusionGenerationAgent()
    pipeline = RealTimeProcessingPipeline(detection_agent, comprehension_agent, commentator_agent, aggregator_agent, generation_agent)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame and perform object detection
            results = pipeline.process_frame(frame)

            # Render the results on the frame
            annotated_frame = results.render()[0]
            cv2.imshow('YOLOv5 Detection', annotated_frame)

            # Check for 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    main()
