import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part

class MultimodalAgent:
    def __init__(self, project_id: str, location: str, model_name: str):
        """
        Initializes the Multimodal Agent.

        Args:
            project_id: The Google Cloud project ID.
            location: The Google Cloud location.
            model_name: The name of the Generative Model to use.
        """
        if not project_id:
            raise ValueError("Google Cloud Project ID is required.")
        if not location:
            raise ValueError("Google Cloud Location is required.")
        if not model_name:
            raise ValueError("Model name is required.")

        self.project_id = project_id
        self.location = location
        self.model_name = model_name

        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.location)

        # Load the generative model
        self.model = GenerativeModel(self.model_name)
        self.chat = None # Will be initialized when starting a new chat

        print(f"MultimodalAgent initialized with project: {self.project_id}, location: {self.location}, model: {self.model_name}")

    def start_chat(self):
        """Starts a new chat session with the model."""
        self.chat = self.model.start_chat(response_validation=False) # allow function calling without strict validation for now
        print("New chat session started.")
        # You could send an initial system message or context here if needed
        # self.chat.send_message("You are a helpful assistant.")
        return self.chat

    def send_message(self, text_prompt: str = None, image_parts: list = None, audio_parts: list = None, video_parts: list = None) -> str:
        """
        Sends a message (which can be a combination of text, image, audio, video) to the model.

        Args:
            text_prompt: The text prompt to send to the model.
            image_parts: A list of image Parts (e.g., Part.from_uri, Part.from_data).
            audio_parts: A list of audio Parts.
            video_parts: A list of video Parts.

        Returns:
            The model's text response.
        """
        if not self.chat:
            print("Chat not started. Starting a new chat session.")
            self.start_chat()

        if not text_prompt and not image_parts and not audio_parts and not video_parts:
            raise ValueError("At least one input type (text, image, audio, video) must be provided.")

        prompt_parts = []
        if text_prompt:
            prompt_parts.append(Part.from_text(text_prompt))

        # Placeholder for how other modalities might be added
        # In subsequent steps, we will create actual Part objects from data/files
        if image_parts:
            prompt_parts.extend(image_parts) # Expecting list of Part objects
        if audio_parts:
            prompt_parts.extend(audio_parts) # Expecting list of Part objects
        if video_parts:
            prompt_parts.extend(video_parts) # Expecting list of Part objects

        print(f"Sending parts to model: {[(type(p.to_dict().get('inline_data', {}).get('mime_type', 'text')) if not p.to_dict().get('text', None) else 'text') for p in prompt_parts]}")


        try:
            response = self.chat.send_message(prompt_parts, stream=False) # Using stream=False for now
            # For streaming text:
            # responses = self.chat.send_message(prompt_parts, stream=True)
            # collected_response = []
            # for res in responses:
            #     collected_response.append(res.text)
            # return "".join(collected_response)

            # For function calling, one might inspect response.candidates[0].content.parts[0].function_call
            # For now, just returning text
            return response.text
        except Exception as e:
            print(f"Error sending message to model: {e}")
            # Potentially re-raise or handle more gracefully
            raise

    # Placeholder for future methods related to specific modalities
    def process_image_input(self, image_data):
        # This will be developed to convert raw image data to a Part object
        # For example: return Part.from_data(image_data, mime_type="image/jpeg")
        pass

    def process_audio_input(self, audio_data):
        # This will be developed to convert raw audio data to a Part object
        # For example: return Part.from_data(audio_data, mime_type="audio/wav")
        pass

    def process_video_input(self, video_frame_data):
        # This will be developed to convert raw video data to a Part object
        # For example: return Part.from_data(video_frame_data, mime_type="video/mp4")
        pass

if __name__ == '__main__':
    # This is for basic testing of the agent class.
    # It requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION to be set in the environment.
    # And GOOGLE_GENAI_USE_VERTEXAI=True

    print("Attempting to run agent.py for a basic test...")
    print(f"GOOGLE_GENAI_USE_VERTEXAI: {os.getenv('GOOGLE_GENAI_USE_VERTEXAI')}")
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
    print(f"GOOGLE_CLOUD_PROJECT: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
    print(f"GOOGLE_CLOUD_LOCATION: {os.getenv('GOOGLE_CLOUD_LOCATION')}")

    use_vertex_env = os.getenv('GOOGLE_GENAI_USE_VERTEXAI', 'False').lower() == 'true'
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    location = os.getenv('GOOGLE_CLOUD_LOCATION')

    if use_vertex_env and project and location:
        print(f"Initializing agent with project='{project}', location='{location}'")
        try:
            agent = MultimodalAgent(
                project_id=project,
                location=location,
                model_name="gemini-2.0-flash-live-preview-04-09" # Using the specified model
            )
            agent.start_chat()
            # Test sending a simple text message
            response_text = agent.send_message(text_prompt="Hello! What can you do?")
            print(f"Model Response: {response_text}")

            # Example of how image part would be constructed (actual image data needed)
            # from vertexai.generative_models import Image
            # image_part = Part.from_image(Image.load_from_file("path_to_your_image.jpg")) # Example
            # response_with_image = agent.send_message(text_prompt="Describe this image", image_parts=[image_part])
            # print(f"Model Response to image: {response_with_image}")

        except Exception as e:
            print(f"Error during agent test: {e}")
    else:
        print("Environment variables for Vertex AI not set. Skipping agent test.")
        print("Please set GOOGLE_GENAI_USE_VERTEXAI=True, GOOGLE_CLOUD_PROJECT, and GOOGLE_CLOUD_LOCATION.")
