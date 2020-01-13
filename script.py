import requests
import json
import jsonpath
from nltk import pos_tag
from PIL import Image



'''
download and save the uri of the latest meme
'''




'''
call vision api to parse words from the meme
'''
url = "https://vision.googleapis.com/v1/images:annotate"
querystring = {"key": "<key>"}
payload = "{\"requests\": [  { \"image\": {  \"source\": {    \"imageUri\": \"https://i.redd.it/i4el1q6ql7841.jpg\" }  },  \"features\": [ { \"type\": \"TEXT_DETECTION\"  } ] }  ]}"
headers = {
    'cache-control': "no-cache",
    'Content-Type': "application/json"
}

#response = requests.request("POST", url, data=payload, headers=headers, params=querystring)
#visionOutput = json.loads(response.text)

with open('vision_output.json') as f:  #loading from json file to limit request rate - tied to billing account
  visionOutput = json.load(f)

wordsOnImage = jsonpath.jsonpath(visionOutput, "responses[0].textAnnotations[*].description")
print(wordsOnImage)


'''
use nltk to get the nouns on the image and pick the word to replace
'''
nouns = []
for word in wordsOnImage[1:]:  #skip first iteration because vision api returns the full text block in the first element of the response.  the words start from the second element
    print(pos_tag([word]))
    if [x[1] for x in pos_tag([word])][0] == 'NN':
        nouns.append(word)
print(nouns)

wordToReplace = nouns[0]  #can choose a different algorithm for this
print('word to replace: ' + wordToReplace)


'''
parse the vision api response to get the coordinates of each instance of the word and replace with the casserole image)
'''
replaceCoordinates = jsonpath.jsonpath(visionOutput, "$.responses[0].textAnnotations[?(@.description=='" + wordToReplace + "')].boundingPoly.vertices")
print(replaceCoordinates)

#coordinate (0,0) is top-left of the image
#get width/height required for casserole image in order to scale it
for coordinate in replaceCoordinates:  #vision api returns coordinates of top-left corner rotating clockwise in sequence (ending in bottom-left corner)
    '''
    resize the casserole image based on dimensions of the word to replace
    '''
    width = coordinate[1]['x'] - coordinate[0]['x']
    height = coordinate[3]['y'] - coordinate[0]['y']
    
    casserole = Image.open('casserole.png')
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

