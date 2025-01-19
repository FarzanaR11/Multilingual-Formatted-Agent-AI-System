import langdetect
from deep_translator import GoogleTranslator
from transformers import pipeline
import logging

# Set up logging for the agent system
logging.basicConfig(filename='multilingual_agent.log', level=logging.INFO)

# Language Detection Class
class LanguageDetector:
    def detect_language(self, text):
        """Detect the language of the input text."""
        try:
            lang = langdetect.detect(text)
            logging.info(f"Detected language: {lang}")
            return lang
        except Exception as e:
            logging.error(f"Error in detecting language: {str(e)}")
            return "en"  # Default to English in case of error

# Translator Class using deep_translator for better accuracy
class TranslatorService:
    def __init__(self):
        pass
    
    def translate_text(self, text, target_lang):
        """Translate text into the target language."""
        try:
            translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
            logging.info(f"Translated '{text}' to {target_lang}: {translated}")
            return translated
        except Exception as e:
            logging.error(f"Error in translation: {str(e)}")
            return text  # Return the text as-is if translation fails

# NLP Task Handler using Hugging Face pipeline
class TaskHandler:
    def __init__(self):
        # Load a pre-trained model for question-answering (or any task you need)
        self.qa_pipeline = pipeline("question-answering")
    
    def process_query(self, query, context=""):
        """Process user query using a pre-trained NLP model."""
        try:
            answer = self.qa_pipeline(question=query, context=context)
            logging.info(f"Processed query: {query} -> Answer: {answer['answer']}")
            return answer['answer']
        except Exception as e:
            logging.error(f"Error in processing query: {str(e)}")
            return "Sorry, I could not process your query."

# Multilingual Agent Class
class MultilingualAgent:
    def __init__(self):
        self.lang_detector = LanguageDetector()
        self.translator = TranslatorService()
        self.task_handler = TaskHandler()
    
    def process_user_query(self, user_input, context=""):
        """Process user query in any language."""
        # Step 1: Detect Language
        detected_language = self.lang_detector.detect_language(user_input)
        
        # Step 2: Translate if needed (Assuming English is the preferred response language)
        if detected_language != 'en':
            user_input = self.translator.translate_text(user_input, 'en')
        
        # Step 3: Process the query using NLP model
        response = self.task_handler.process_query(user_input, context)
        
        # Step 4: Translate the response back to User's Language
        if detected_language != 'en':
            response = self.translator.translate_text(response, detected_language)
        
        # Return the agent's response in the correct language
        return response

# Example Usage:
if __name__ == "__main__":
    agent = MultilingualAgent()

    # Simulate a customer query in Spanish
    user_input = "¿Cómo puedo hacer una devolución?"
    print(f"User Input: {user_input}")
    
    response = agent.process_user_query(user_input, context="You can return items within 30 days of purchase.")
    print(f"Agent Response: {response}")

    # Log interaction for feedback loop
    logging.info(f"User Query: {user_input}")
    logging.info(f"Agent Response: {response}")
