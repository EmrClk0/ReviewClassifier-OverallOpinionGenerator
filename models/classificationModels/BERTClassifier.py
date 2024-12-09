from transformers import BertForSequenceClassification, BertTokenizer
import torch

class BERTClassifier:

    def __init__(self, modelPath, tokenizerPath="bert-base-uncased", workOnCuda=False):
        self.modelPath = modelPath
        self.tokenizerPath =tokenizerPath
        self.workOnCuda = workOnCuda
        self.loadModel()
    

    def loadModel(self):
        try:

            self.tokenizer = BertTokenizer.from_pretrained(self.tokenizerPath)  
            self.model = BertForSequenceClassification.from_pretrained(self.modelPath)

            if self.workOnCuda:
                self.model = self.model.cuda()   #inputlar cuda ya taşınırsa
                
            self.model.eval()
            print("Model loaded successfully and set to evaluation mode.")
    
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading bert the model: {e}")
        


    def predict(self,text):
        try:
            inputs =self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)

            if self.workOnCuda:
                inputs = {key: value.cuda() for key, value in inputs.items()}

            with torch.no_grad():  
                outputs = self.model(**inputs)

            logits = outputs.logits

            predicted_class = torch.argmax(logits, dim=1).item()
            return predicted_class
        
        except Exception as e:
            raise RuntimeError(f"An error occurred while making predictions with the BERT model: {e}")
        
        