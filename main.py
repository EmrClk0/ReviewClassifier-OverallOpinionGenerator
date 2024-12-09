# TO START API ---> uvicorn main:app --reload
# SWAGGER UI ---> http://127.0.0.1:8000/docs 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import json
import re

from models.classificationModels.BERTClassifier import BERTClassifier
from models.llmModels.HGModel import HGModel
from huggingface_hub import login
from models.llmModels.GPT import GPT
from langchain import PromptTemplate


#verileri yükle
data =pd.read_csv("data/Clothing/clothingClear.csv")
config_data = json.load(open("config.json"))


with open("models/ratingDict.json", "r") as json_file:
        ratingDict = json.load(json_file)



#----------YARDIMCI FONKSİYONLAR -----------
def decodePredictedClass(predictedClass):
    return int(ratingDict[str(predictedClass)])

def clean_text(text):
    text = re.sub(r"[\"\\]", "", text) 
    text = re.sub(r"\s+", " ", text)   
    text = text.strip()                
    return text

def reviewListGenarator(data,targetID):
    filteredDF = data[data["clothingID"]==targetID]
    reviewList = filteredDF["cleanReview"].tolist()
    return reviewList

def inputGenerator(reviewList):
    template = """ Analyze the user reviews provided below and create a concise paragraph summarizing the overall sentiment. Your summary should answer the following questions: Do users generally like or dislike the dress? What features are most appreciated or criticized? Do users recommend the dress? Include any additional informations observed, but keep the output as a single cohesive paragraph. Here are the user reviews:
    {revs}
    """

    prompt = PromptTemplate(
        input_variables=["revs"],
        template=template
    )
    
    text = "Review: "+" Review: ".join(reviewList)
    inputText = prompt.format(revs=text)
    return inputText




#----------------MODELLERİ AÇ ------------------------

bertModelPath="models/classificationModels/bert"
bertModel = BERTClassifier(modelPath=bertModelPath)


API_KEY=config_data["GPT_API_KEY"]
GPT_TOKEN_LIMIT=128000
gptModel = GPT(API_KEY,GPT_TOKEN_LIMIT)



LLAMA_TOKEN_LIMIT=128000
LLAMA_MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"

HF_TOKEN = config_data["HF_TOKEN"]
login(HF_TOKEN)

LLAMAmodel = HGModel(LLAMA_MODEL_NAME,LLAMA_TOKEN_LIMIT)







#----------- FASTAPI ------------------
class Review(BaseModel):
    reviewText: str  

app = FastAPI()

# CORS Middleware localde istek atabilmek için.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"]   
)


@app.get("/dressInformations")
def  getDressRatingMeans():
    dressRatingSummary= (
                        data.groupby("clothingID")["rating"]
                        .agg(["count", "mean"])
                        .reset_index()
                        .sort_values(by="count", ascending=False)
                    )
    dressRatingSummary["mean"] = dressRatingSummary["mean"].round(1)
    return dressRatingSummary.to_dict(orient='records')
    



@app.post("/classifyReview")
def classify_review(review: Review):

    predictedClass=bertModel.predict(review.reviewText)
    decodedClass = decodePredictedClass(predictedClass)

    return {"class":decodedClass}


@app.get("/exampleReviews")
def getExampleReviews(dressID:int):
    fdata = data[data["clothingID"]==dressID]
    if(len(fdata))>=3:
        return data[data["clothingID"]==dressID].sample(3).to_dict(orient='records')
    else:
        return data[data["clothingID"]==dressID].to_dict(orient='records')


@app.get("/gpt/conclusionGenerate")
def conclusionGenerate(dressID:int):

    reviewList = reviewListGenarator(data,dressID)
    input = inputGenerator(reviewList)

    gptResponse=gptModel.request(input)

    return {"LLMresponse":gptResponse}



@app.get("/llama/conclusionGenerate")
def conclusionGenerate(dressID:int):

    reviewList = reviewListGenarator(data,dressID)
    input = inputGenerator(reviewList)

    llmaResponse=LLAMAmodel.request(input,256)
    llmaResponse=clean_text(llmaResponse)
    return {"LLMresponse":llmaResponse}

