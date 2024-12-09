from langchain_openai import ChatOpenAI


class GPT:

    def __init__(self, API_KEY, GPT_TOKEN_LIMIT, MODEL_NAME="gpt-4o-mini"):
      
        self.API_KEY = API_KEY
        self.GPT_TOKEN_LIMIT = GPT_TOKEN_LIMIT
        self.model = ChatOpenAI(model=MODEL_NAME,
                                openai_api_key=API_KEY)
 
 
    def __getNumTokens(self,input):
        return self.model.get_num_tokens(input)
    

    def request(self, input):
        tokenNum = self.__getNumTokens(input)

        if tokenNum < self.GPT_TOKEN_LIMIT:
            response = self.model.invoke(input)
            return response.content
        else:
            raise Exception(f"TOKEN LIMIT EXCEEDED! INPUT CONTEXT LENGTH --> {tokenNum}, ALLOWED CONTEXT LENGTH IS {self.GPT_TOKEN_LIMIT}")

