# Automatic Text Summarization

## Intro

Abstractive Summarization is a method, which aims to automatically generate summaries of documents through the extraction of sentences in the text. The specific model we implemented based on 'The Daily Mail' dataset of stories, taking each and generating summary. The architecture of this method consists of a encoder-decoder design using LSTM layers.

## Data Preprocessing

The file ```data_util``` responsible for reorganizing the files into more python like data structures, then in ```utils.py``` we have the preprocessing method that vectorizing and divideing our data into train and val

## Training

The file ```utils``` contains the essential function to perform preprocessing and running the layers to predict a sentence``

```bash
python main.py
```

## Results


**Ouput model loss**

```bash
Loaded Stories 250
Train input size 175
Test input size 75

Train
Number of unique input tokens: 5613
Number of unique output tokens: 678

Epoch 1/25
3/3 [==============================] - 9s 3s/step - loss: 1.6426 - val_loss: 1.9482
Epoch 2/25
3/3 [==============================] - 9s 3s/step - loss: 1.6379 - val_loss: 1.9495
Epoch 3/25
3/3 [==============================] - 9s 3s/step - loss: 1.6329 - val_loss: 1.9631
Epoch 4/25
3/3 [==============================] - 9s 3s/step - loss: 1.6130 - val_loss: 2.1303
Epoch 5/25
3/3 [==============================] - 9s 3s/step - loss: 1.5702 - val_loss: 2.2681
Epoch 6/25
3/3 [==============================] - 9s 3s/step - loss: 1.5316 - val_loss: 2.4084
Epoch 7/25
3/3 [==============================] - 9s 3s/step - loss: 1.4977 - val_loss: 2.5793
Epoch 8/25
3/3 [==============================] - 9s 3s/step - loss: 1.4648 - val_loss: 2.7414
Epoch 9/25
3/3 [==============================] - 9s 3s/step - loss: 1.4324 - val_loss: 2.8845
Epoch 10/25
3/3 [==============================] - 9s 3s/step - loss: 1.4033 - val_loss: 3.0378
Epoch 11/25
3/3 [==============================] - 9s 3s/step - loss: 1.3766 - val_loss: 3.1628
Epoch 12/25
3/3 [==============================] - 9s 3s/step - loss: 1.3491 - val_loss: 3.2501
Epoch 13/25
3/3 [==============================] - 10s 3s/step - loss: 1.3261 - val_loss: 3.3194
Epoch 14/25
3/3 [==============================] - 9s 3s/step - loss: 1.3101 - val_loss: 3.3772
Epoch 15/25
3/3 [==============================] - 9s 3s/step - loss: 1.2884 - val_loss: 3.4208
Epoch 16/25
3/3 [==============================] - 11s 4s/step - loss: 1.2742 - val_loss: 3.4442
Epoch 17/25
3/3 [==============================] - 10s 3s/step - loss: 1.2610 - val_loss: 3.4650
Epoch 18/25
3/3 [==============================] - 9s 3s/step - loss: 1.2436 - val_loss: 3.4949
Epoch 19/25
3/3 [==============================] - 10s 3s/step - loss: 1.2307 - val_loss: 3.5108
Epoch 20/25
3/3 [==============================] - 9s 3s/step - loss: 1.2206 - val_loss: 3.5259
Epoch 21/25
3/3 [==============================] - 9s 3s/step - loss: 1.2070 - val_loss: 3.5378
Epoch 22/25
3/3 [==============================] - 9s 3s/step - loss: 1.1935 - val_loss: 3.5438
Epoch 23/25
3/3 [==============================] - 9s 3s/step - loss: 1.1865 - val_loss: 3.5399
Epoch 24/25
3/3 [==============================] - 8s 3s/step - loss: 1.1783 - val_loss: 3.5414
Epoch 25/25
3/3 [==============================] - 8s 3s/step - loss: 1.1719 - val_loss: 3.5375
```

```json
{
    "rouge-1": {
        "f": 0.054421289314752465,
        "p": 0.4231884057971014,
        "r": 0.02974606795512569
    },
    "rouge-2": {
        "f": 0.0028187238995702316,
        "p": 0.03259259259259259,
        "r": 0.001495800983364506
    },
    "rouge-l": {
        "f": 0.08334677411153489,
        "p": 0.3751937984496124,
        "r": 0.04805841277828462
    }
}
```

|       Model        | ROUGE-1 | ROUGE-2 | ROUGE-l |
| ------------------ | :-----: | ------: | ------: |
| Our Implementation | 42.31%  |   3.25% |  37.51% |
| Random Baseline    | 32.14%  |  11.39% |  22.11% |

## Example Summaries

| Ground Truth Summary | Model Generated Summary |
| -------------------- | :---------------------: |
| a                    |            b            |


---

## Example prediction

__Original text:__

the duchess of cornwall has spoken of her pride at opening a new cancer support centre in aberdeen today. camilla was without her husband prince charles but was still in royal company as the queen of norway, queen sonja, also attended the opening of the elizabeth montgomerie building, next to the city's royal infirmary hospital. the building is named after golfer colin montgomerie's mother and is a branch of the maggie's cancer care charity which he set up in her memory after she died of cancer in 1991. apt choice: the duchess of cornwall arrives in tartan for the opening of a new maggie's cancer centre in aberdeen royal protocol: the duchess curtseys as she greets the queen of norway similar taste in clothes: the royal pair, both in dresses and blazers, were given a tour of the centre known as the duchess of rothesay in scotland, camilla donned a green roy allen lord of the isles tartan dress and blazer for her visit. while it's usually the duchess of cambridge who is known for her style credentials, camilla's look is on trend with numerous designers featuring tartan and window-pane check in their london fashion week shows recently. camilla teamed her patriotic outfit with beige shoes and a matching handbag and accessorised with a string of pearls round her neck. mother's legacy: golfer colin montgomerie, who set up maggie's cancer support in memory of his late mother helped to show the duchess around warm welcome: the duchess meets children who attended the opening gifts: flowers were given to the two royals by leigh bonthrone, 15, and jemma findlay, eight, whose families have been affected by cancer meanwhile, queen sonja wore a similar outfit of a blazer and dress but opted for less colourful shades of grey and pale blue with stripes instead of checks, accessorised with a chunky silver necklace. camilla curtsied to the queen as they greeted one another before being given a tour of the new building, which they first saw plans for during a previous meeting in oslo. two girls, leigh bonthrone, 15, and jemma findlay, eight, whose families have been affected by cancer presented bouquets to the duchess and queen. proud patron: camilla said she was delighted with what the charity have achieved with the new centre camilla said of the charities support centres: 'these places are the most uplifting places you could ever be, and you come out feeling better' camilla has been president of the cancer charity since 2008 and has visited many of its 17 centres. today she met maggie’s staff, supporters and fundraisers, and unveiled a commemorative stone to mark the event. they also broke a kransekake, a norwegian celebration cake, to mark the opening. addressing guests, camilla, said: 'as a very, very proud patron of maggie’s, i just want to thank everyone here today for everything they have done to make this building so special. 'these places are the most uplifting places you could ever be, and you come out feeling better. 'that is surely the point of maggie’s. you see so many smiling faces and that is what you want if you’re facing this terrible disease.' commemorative: the duchess made a speech as she unveiled a plaque montgomerie and his family members also attended the event. montgomerie said: 'the opening of the centre is a very special day for myself and my family and i want to thank everyone involved in making this dream a reality.' earlier, the duchess and queen sonja spent time talking with leigh, from kirkcaldy, fife, whose father recovered from cancer, and jemma, of aberdeen, who lost her mother to the disease. leigh said: 'the duchess was asking about how maggie’s can support people. i told her how the centres had helped my family and how much one was needed here in aberdeen.'

__Summary generated:__

'as can be seen from the photos the conditions in the store were unacceptable.'as can be seen from the photos the conditions in the store were unacceptable.
