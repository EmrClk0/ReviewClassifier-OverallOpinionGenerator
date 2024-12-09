import torch
from huggingface_hub import login
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          pipeline)



class HGModel:

    def __init__(self, modelName, modelTokenLimit):

        self.modelName =modelName
        self.modelTokenLimit =modelTokenLimit
        
        
        try:
            #login(HF_TOKEN)
            self.loadModel()
        except Exception as e:
            raise Exception(f"Error occurred: {str(e)}")


    def loadModel(self):

        # Configure model loading parameters
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Initialize Tokenizer and Model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                                                            self.modelName,
                                                            device_map="auto",
                                                            quantization_config=self.bnb_config
                                                            )
            
            self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        except Exception as e:
            raise Exception(f"Error initializing model or tokenizer: {str(e)}")
        

    def __getNumTokens(self,input):
        tokens = self.tokenizer(input)
        return len(tokens['input_ids'])
    
    def __clearCache(self):
        torch.cuda.empty_cache()

    
    def request(self, input,maxNewToken=256):
        try:
           
            token_count = self.__getNumTokens(input)

            if token_count > self.modelTokenLimit:
                raise Exception(f"Token size is too large: {token_count} tokens, max allowed: {self.modelTokenLimit}")

            # Generate response
           
            self.__clearCache()
            
            sequences = self.pipe(input, return_full_text=False, max_new_tokens=maxNewToken)
            generated_text = sequences[0]["generated_text"]
            return generated_text

        except Exception as e:
            return f"Error occurred: {str(e)}"