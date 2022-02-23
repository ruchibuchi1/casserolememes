import urllib.request as req
import requests
import json
import jsonpath
from nltk import pos_tag
import re
from PIL import Image
import spacy


def getWordToReplace(memeText):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(memeText)

    ranking = {}
    tagScore = {'NN': 6, 'NNS': 6, 'NNP': 4, 'NNPS': 4, 'PRP': 2}
    depScore = {'compound': 8, 'dobj': 7, 'npadvmod': 6, 'attr': 5, 'amod': 4, 'nsubj': 3, 'pobj': 2, 'conj': 1}
    for token in doc:
        print(token.text, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
        if token.is_stop:
            ranking[token] = -1
        elif not token.is_alpha:
            ranking[token] = -1
        else:
            value = 0
            if token.tag_ in tagScore:
                value += tagScore[token.tag_]
            if token.dep_ in depScore:
                value += depScore[token.dep_]
            ranking[token] = value

    print(ranking)
    return str(max(ranking, key=ranking.get))


    
    
    
imgurl = "https://i.redd.it/qj0j9t5xpej81.jpg" 
#imgurl = "https://i.imgur.com/DUus51W.jpg"
apiKey = "";


'''
download and save the uri of the latest meme
'''
req.urlretrieve(imgurl, "meme.jpg")



'''
call vision api to parse words from the meme
'''
url = "https://vision.googleapis.com/v1/images:annotate"
querystring = {"key": apiKey}
payload = "{\"requests\": [  { \"image\": {  \"source\": {    \"imageUri\": \"" + imgurl + "\" }  },  \"features\": [ { \"type\": \"TEXT_DETECTION\"  } ] }  ]}"
headers = {
    'cache-control': "no-cache",
    'Content-Type': "application/json"
}

response = requests.request("POST", url, data=payload, headers=headers, params=querystring)
visionOutput = json.loads(response.text)

with open('response.txt', 'w') as outfile:
    json.dump(visionOutput, outfile)

#with open('vision_output.json') as f:  #loading from json file to limit request rate - tied to billing account
  #visionOutput = json.load(f)

wordsOnImage = jsonpath.jsonpath(visionOutput, "responses[0].textAnnotations[*].description")
print(wordsOnImage)



'''
use Spacy to get the word to replace
'''
memeText = "";
for word in wordsOnImage[1:]:
    memeText += word + " "
print(memeText)

wordToReplace = getWordToReplace(memeText)
print(wordToReplace)



'''
parse the vision api response to get the coordinates of each instance of the word and replace with the casserole image)
'''
replaceCoordinates = jsonpath.jsonpath(visionOutput, "$.responses[0].textAnnotations[?(@.description=='" +  wordToReplace + "')].boundingPoly.vertices")
print(replaceCoordinates)

#coordinate (0,0) is top-left of the image
#get width/height required for casserole image in order to scale it
for coordinate in replaceCoordinates:  #vision api returns coordinates of top-left corner rotating clockwise in sequence (ending in bottom-left corner)
    '''
    resize the casserole image based on dimensions of the word to replace
    '''
    xCoordinates = [coordinate[0]['x'], coordinate[1]['x'], coordinate[2]['x'], coordinate[3]['x']]
    yCoordinates = [coordinate[0]['y'], coordinate[1]['y'], coordinate[2]['y'], coordinate[3]['y']]
    
    xCoordinateMin = min(xCoordinates)
    xCoordinateMax = max(xCoordinates)
    yCoordinateMin = min(yCoordinates)
    yCoordinateMax = max(yCoordinates)
    
    width = xCoordinateMax - xCoordinateMin
    height = yCoordinateMax - yCoordinateMin
    
    casserole = Image.open('casserole.png')
    print('width: ' + str(width) + ' height: ' + str(height))
    casseroleTmp = casserole.resize((width, height))
    
    
    '''
    paste the new casserole image on top of the first image
    '''
    if(replaceCoordinates.index(coordinate) == 0):
        meme = Image.open('meme.jpg')
        memeCopy = meme.copy()
    else:
        memeCopy = Image.open('meme_copy.jpg')
    
    memeCopy.paste(casseroleTmp, (coordinate[0]['x'], coordinate[0]['y']))  #give the top-left coordinate of the replacement area as the position to paste
    memeCopy.save('meme_copy.jpg')


