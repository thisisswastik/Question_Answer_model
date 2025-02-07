# Question_Answer_model
This project implements a Question Answering (QA) system using a Fine-Tuned Language Model (FLAN-T5-Large). The user inputs a paragraph, and the model generates answers to questions based on that paragraph.
We use the FLAN-T5-Large model, a fine-tuned version of Google's T5 (Text-to-Text Transfer Transformer).
Transformer-Based: The model follows the encoder-decoder architecture.
Fine-Tuned for Instruction Following: Unlike standard T5, FLAN-T5 is optimized for question-answering and task-specific prompts.
Sequence-to-Sequence Learning: It takes the question + context as input and generates an answer as output.
The following steps ensure that the input is properly formatted before passing it to the model:

📌 Text Cleaning & Formatting
Converts the question and passage into a structured input for FLAN-T5.
Strips unnecessary whitespaces.
📌 Tokenization
Uses Hugging Face's AutoTokenizer (google/flan-t5-large).
Converts text into input tokens for model processing.
📌 GPU Acceleration
Moves the model to cuda if a GPU is available, significantly improving speed.
 Implementation Details
📌 Streamlit UI Components
Text Area: Users enter a paragraph.
Text Input: Users type a question.
Model Execution: The system tokenizes the input, passes it to FLAN-T5, and decodes the output.
Display Section: Shows the generated answer in real time.
Evaluation Methodology
To evaluate the model’s accuracy and response quality, we use manual validation and benchmark datasets.

📌 Evaluation Metrics
✅ BLEU Score - Measures how well the generated answer matches the expected output.
✅ ROUGE Score - Evaluates text similarity by comparing generated and reference answers.
✅ Human Evaluation - Checks if the answers make sense and are factually correct.
📌 Sample Evaluation
Input Passage
"The Eiffel Tower was completed in 1889 and stands in Paris, France. It was originally criticized but later became a global icon."

Question
"When was the Eiffel Tower completed?"

Expected Output
"1889"

Generated Model Output
✅ Correct Answer: "The Eiffel Tower was completed in 1889."
