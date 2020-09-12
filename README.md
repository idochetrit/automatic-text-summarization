# automatic-text-summarization 

# Intro
Abstractive Summarization is a method, which aims to automatically generate summaries of documents through the extraction of sentences in the text. The specific model we implemented based on 'The Daily Mail' dataset of stories, taking each and generating summary. The architecture of this method consists of a encoder-decoder design using LSTM layers.

## Data Preprocessing: ##

The file ```data_util``` responsible for reorganizing the files into more python like data structures, then in ```utils.py``` we have the preprocessing method that vectorizing and divideing our data into train and val 

## Training: ##
The file ```utils``` contains the essential function to perform preprocessing and running the layers to predict a sentence``

```
python main.py
```

## Results: ##
```json
{
    "rouge-1": {
        "1": 0.03549538099112015,
        "p": 0.5214285714285715,
        "r": 0.018400363478592384
    },
    "rouge-2": {
        "f": 0.003928072845294563,
        "p": 0.059259259259259255,
        "r": 0.002033130043667561
    },
    "rouge-l": {
        "f": 0.03239614494531364,
        "p": 0.5,
        "r": 0.01674789167308075
    }
}
```

```
| Model           | ROUGE-1       | ROUGE-2  |
| ----------------|:-------------:| --------:|
| Our Implementation| 52.14%        |   5.92% |
| Random Baseline | 32.14%        |   11.39% |

## Example Summaries: ##
| Ground Truth Summary           | Model Generated Summary       | 
| ----------------|:-------------:| 
|the duchess of cornwall has spoken of her pride at opening a new cancer support centre in aberdeen today. camilla was without her husband prince charles but was still in royal company as the queen of norway, queen sonja, also attended the opening of the elizabeth montgomerie building, next to the city's royal infirmary hospital. the building is named after golfer colin montgomerie's mother and is a branch of the maggie's cancer care charity which he set up in her memory after she died of cancer in 1991. apt choice: the duchess of cornwall arrives in tartan for the opening of a new maggie's cancer centre in aberdeen royal protocol: the duchess curtseys as she greets the queen of norway similar taste in clothes: the royal pair, both in dresses and blazers, were given a tour of the centre known as the duchess of rothesay in scotland, camilla donned a green roy allen lord of the isles tartan dress and blazer for her visit. while it's usually the duchess of cambridge who is known for her style credentials, camilla's look is on trend with numerous designers featuring tartan and window-pane check in their london fashion week shows recently. camilla teamed her patriotic outfit with beige shoes and a matching handbag and accessorised with a string of pearls round her neck. mother's legacy: golfer colin montgomerie, who set up maggie's cancer support in memory of his late mother helped to show the duchess around warm welcome: the duchess meets children who attended the opening gifts: flowers were given to the two royals by leigh bonthrone, 15, and jemma findlay, eight, whose families have been affected by cancer meanwhile, queen sonja wore a similar outfit of a blazer and dress but opted for less colourful shades of grey and pale blue with stripes instead of checks, accessorised with a chunky silver necklace. camilla curtsied to the queen as they greeted one another before being given a tour of the new building, which they first saw plans for during a previous meeting in oslo. two girls, leigh bonthrone, 15, and jemma findlay, eight, whose families have been affected by cancer presented bouquets to the duchess and queen. proud patron: camilla said she was delighted with what the charity have achieved with the new centre camilla said of the charities support centres: 'these places are the most uplifting places you could ever be, and you come out feeling better' camilla has been president of the cancer charity since 2008 and has visited many of its 17 centres. today she met maggie’s staff, supporters and fundraisers, and unveiled a commemorative stone to mark the event. they also broke a kransekake, a norwegian celebration cake, to mark the opening. addressing guests, camilla, said: 'as a very, very proud patron of maggie’s, i just want to thank everyone here today for everything they have done to make this building so special. 'these places are the most uplifting places you could ever be, and you come out feeling better. 'that is surely the point of maggie’s. you see so many smiling faces and that is what you want if you’re facing this terrible disease.' commemorative: the duchess made a speech as she unveiled a plaque montgomerie and his family members also attended the event. montgomerie said: 'the opening of the centre is a very special day for myself and my family and i want to thank everyone involved in making this dream a reality.' earlier, the duchess and queen sonja spent time talking with leigh, from kirkcaldy, fife, whose father recovered from cancer, and jemma, of aberdeen, who lost her mother to the disease. leigh said: 'the duchess was asking about how maggie’s can support people. i told her how the centres had helped my family and how much one was needed here in aberdeen.' | 'as can be seen from the photos the conditions in the store were unacceptable.'as can be seen from the photos the conditions in the store were unacceptable.| 
| | 'as can be seen from the photos the conditions in the store were unacceptable.'as can be seen from the photos the conditions in the store were unacceptable.| 
